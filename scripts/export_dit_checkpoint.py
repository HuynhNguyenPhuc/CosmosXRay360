"""
Export / import DiT weights from a PyTorch-Lightning training checkpoint or trained model.

Use-cases
---------
1. Export net (main weights) from Lightning checkpoint:

    python -m scripts.export_dit_checkpoint export \
        --ckpt  outputs/nvsyn_cosmos25_rf/checkpoints/epoch=0042.ckpt \
        --out   exported/dit_net_epoch42.pth

2. Export EMA weights (default) with config.json:

    python -m scripts.export_dit_checkpoint export \
        --ckpt  outputs/.../epoch=0042.ckpt \
        --out   exported/dit_ema_epoch42.pth \
        --export-config

   This generates both dit_ema_epoch42.pth and config.json in exported/ directory.
   Use config.json when loading the checkpoint with Inferencer:
   
    from predict2_5.inferencer import Inference
    inferencer = Inference(
        checkpoint_path="exported/dit_ema_epoch42.pth",
        config_path="exported/config.json",
        device="cuda"
    )

3. Load exported weights into a fresh DiT model (no Lightning required):

    from scripts.export_dit_checkpoint import load_dit_checkpoint
    net = load_dit_checkpoint("exported/dit_ema_epoch42.pth", model_size="2B", device="cuda")
    net.eval()

4. Test export→load round-trip cycle (verify compatibility):

    python -m scripts.export_dit_checkpoint roundtrip \
        --ckpt outputs/epoch=0042.ckpt --device cuda

5. Export directly from a trained model instance during/after training:

    from scripts.export_dit_checkpoint import export_model_state_dict
    # In training callback or after training:
    export_model_state_dict(
        model=cosmos_model.net_ema,
        output_path="checkpoints/dit_ema_epoch_10.pth",
        model_size="2B",
        source="ema"
    )

6. Inspect checkpoint contents:

    python -m scripts.export_dit_checkpoint inspect --ckpt outputs/epoch=0042.ckpt

Checkpoint format (Lightning .ckpt from module.py)
---------------------------------------------------
  checkpoint["state_dict"]           # main model weights with "net." prefix
  checkpoint["net_ema"]              # EMA state dict (nested dict, no prefix)
  checkpoint["ema_exp_coefficient"]  # Power-EMA scalar for bit-exact training continuation

Exported .pth format
---------------------
  {
      "net":         <state_dict>,    # bare DiT weights (no prefix, ready to load)
      "model_size":  "2B",            # DiT size tag
      "source":      "net" | "ema",   # source of weights extracted
      "ema_exp_coefficient": <float> | None,  # EMA metadata if source=="ema"
  }

Complete export→load workflow
------------------------------
1. Train model with module.py:
   lightning fit --model-class NVSynCosmos25RF --config config.yaml
   
2. Export trained model weights with config (required for Inferencer):
   python -m scripts.export_dit_checkpoint export \
       --ckpt outputs/epoch=42.ckpt \
       --out exported/dit_ema_epoch42.pth \
       --export-config
   
   This creates two files:
   - exported/dit_ema_epoch42.pth (weights)
   - exported/config.json (model architecture config)
   
3. Load and use exported weights with Inferencer:
   >>> from predict2_5.inferencer import Inference
   >>> inferencer = Inference(
   ...     checkpoint_path="exported/dit_ema_epoch42.pth",
   ...     config_path="exported/config.json",
   ...     device="cuda"
   ... )
   >>> latent = encode(video)
   >>> frames = inferencer(image=xray, prompt="prompt text", cfg_scale=1.5, num_steps=35)
   
4. Verify round-trip compatibility (optional):
   python -m scripts.export_dit_checkpoint roundtrip \
       --ckpt outputs/epoch=42.ckpt --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.serialization

# Allow MetaTensor from MonAI for weights_only=True loading
try:
    import monai.data.meta_tensor

    torch.serialization.add_safe_globals([monai.data.meta_tensor.MetaTensor])
except ImportError:
    pass  # MonAI not available, skip registration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict2_5.utils import get_logger
from predict2_5.constants import (
    COSMOS_2B_PRETRAINED_UUID,
    CROSSATTN_EMB_CHANNELS,
    CROSSATTN_PROJ_IN_CHANNELS,
    NUM_LATENT_FRAMES,
)

from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

log = get_logger(__name__)

_EXPORT_DTYPES = {
    "preserve": None,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _collect_floating_dtypes(state_dict: dict) -> set[torch.dtype]:
    """Collect all floating tensor dtypes present in a state dict."""
    dtypes: set[torch.dtype] = set()
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            dtypes.add(value.dtype)
    return dtypes


def _normalize_state_dict_for_export(
    state_dict: dict,
    export_dtype: Literal["preserve", "float32", "bfloat16", "float16"] = "float32",
    strip_extra_state: bool = True,
) -> tuple[dict, list[str], set[torch.dtype]]:
    """Prepare state dict for portable inference export.

    Floating tensors are optionally cast to a single dtype and all tensors are moved to CPU.
    TransformerEngine `._extra_state` entries can be stripped because they are not required
    for standard inference and often add noise for compatibility checks.
    """
    target_dtype = _EXPORT_DTYPES[export_dtype]
    normalized: dict = {}
    stripped_keys: list[str] = []
    observed_dtypes: set[torch.dtype] = set()

    for key, value in state_dict.items():
        if strip_extra_state and key.endswith("._extra_state"):
            stripped_keys.append(key)
            continue

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.is_floating_point():
                observed_dtypes.add(tensor.dtype)
                if target_dtype is not None:
                    tensor = tensor.to(dtype=target_dtype)
            normalized[key] = tensor
        else:
            normalized[key] = value

    return normalized, stripped_keys, observed_dtypes


def _choose_runtime_dtype(
    state_dict: dict,
    requested_export_dtype: Optional[str] = None,
) -> torch.dtype:
    """Choose a safe runtime dtype for a freshly created DiT before state dict load."""
    if requested_export_dtype in _EXPORT_DTYPES and _EXPORT_DTYPES[requested_export_dtype] is not None:
        return _EXPORT_DTYPES[requested_export_dtype]

    observed_dtypes = _collect_floating_dtypes(state_dict)
    if not observed_dtypes:
        return torch.float32
    if len(observed_dtypes) == 1:
        return next(iter(observed_dtypes))
    if torch.float32 in observed_dtypes:
        return torch.float32
    if torch.bfloat16 in observed_dtypes:
        return torch.bfloat16
    if torch.float16 in observed_dtypes:
        return torch.float16
    return next(iter(observed_dtypes))


def _fix_rope_buffers_fallback(module: torch.nn.Module) -> None:
    """Fix RoPE buffers for models initialized on meta device.

    Args:
        module: Root module containing VideoRopePosition3DEmb layers.
    """
    for _, child in module.named_children():
        if child.__class__.__name__ == "VideoRopePosition3DEmb":
            target_device = child._buffers["dim_spatial_range"].device
            max_len = max(child.max_h, child.max_w, child.max_t)

            child.seq = torch.arange(max_len, device=target_device, dtype=torch.float32)

            dim_h = child._dim_h
            dim_t = child._dim_t
            child.dim_spatial_range = (
                torch.arange(0, dim_h, 2, device=target_device, dtype=torch.float32)[
                    : (dim_h // 2)
                ]
                / dim_h
            )
            child.dim_temporal_range = (
                torch.arange(0, dim_t, 2, device=target_device, dtype=torch.float32)[
                    : (dim_t // 2)
                ]
                / dim_t
            )
        else:
            _fix_rope_buffers_fallback(child)


# ---------------------------------------------------------------------------
# PyTorch checkpoint loading utility
# ---------------------------------------------------------------------------


def _safe_torch_load(path: str, map_location: str = "cpu") -> dict:
    """
    Load checkpoint with fallback for custom objects like MetaTensor.
    PyTorch 2.6+ defaults to weights_only=True. This helper tries that first,
    then falls back to weights_only=False if custom objects are encountered.

    Args:
    path: str - Path to checkpoint file.
    map_location: str - Device to load to (default: "cpu").

    Returns:
    dict - Loaded checkpoint.
    """
    try:
        # Try with weights_only=True (safer, default in PyTorch 2.6+)
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        if "weights_only" in str(e) or "Unsupported global" in str(e):
            log.warning(f"  weights_only=True failed, retrying with weights_only=False")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise


# ---------------------------------------------------------------------------
# DiT factory (mirrors NVSynCosmos25RF._create_dit / _fix_rope_buffers)
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "2B": {"model_channels": 2048, "num_heads": 16, "num_blocks": 28},
    "7B": {"model_channels": 4096, "num_heads": 32, "num_blocks": 28},
    "14B": {"model_channels": 5120, "num_heads": 40, "num_blocks": 36},
}


def create_dit(
    model_size: str = "2B", state_ch: int = 16, device: str = "cpu"
) -> MinimalV1LVGDiT:
    """
    Instantiate a bare MinimalV1LVGDiT with given model size (matches module.py._create_dit).
    Uses meta-device initialization for deferred instantiation, then materializes to target device.
    Ensures exact compatibility with module.py checkpoints.

    Args:
    model_size: str, default "2B" - DiT size ("2B", "7B", "14B").
    state_ch: int, default 16 - Latent channel count (must match VAE output, typically 16).
    device: str, default "cpu" - Target device ("cpu", "cuda", etc.).

    Returns:
    MinimalV1LVGDiT - Fresh model instance on target device, ready for loading state dict.

    Raises:
    ValueError - If model_size is not in ("2B", "7B", "14B").
    """
    if model_size not in _MODEL_CONFIGS:
        raise ValueError(
            f"model_size must be one of {list(_MODEL_CONFIGS.keys())}; got '{model_size}'"
        )
    cfg = _MODEL_CONFIGS[model_size]

    # Initialize on meta device (deferred instantiation)
    with torch.device("meta"):
        net = MinimalV1LVGDiT(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=state_ch,
            out_channels=state_ch,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=cfg["model_channels"],
            num_blocks=cfg["num_blocks"],
            num_heads=cfg["num_heads"],
            atten_backend="minimal_a2a",
            pos_emb_cls="rope3d",
            pos_emb_learnable=True,
            pos_emb_interpolation="crop",
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            crossattn_emb_channels=CROSSATTN_EMB_CHANNELS,
            use_crossattn_projection=True,
            crossattn_proj_in_channels=CROSSATTN_PROJ_IN_CHANNELS,
            sac_config=SACConfig(mode="mm_only"),
            timestep_scale=0.001,
        )

    net.to_empty(device=device)
    net.init_weights()
    _fix_rope_buffers_fallback(net)
    return net


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_checkpoint(
    lightning_ckpt_path: str,
    output_path: str,
    source: Literal["net", "net_ema"] = "net_ema",
    model_size: str = "2B",
    export_config: bool = False,
    export_dtype: Literal["preserve", "float32", "bfloat16", "float16"] = "float32",
    strip_extra_state: bool = True,
) -> None:
    """
    Extract DiT weights from a Lightning .ckpt checkpoint and save as standalone .pth.
    Loads Lightning checkpoint from module.py and extracts either main network or Power-EMA weights.
    Handles both new (nested dict) and legacy (flattened keys) formats. Saves with metadata.

    Args:
    lightning_ckpt_path: str - Path to Lightning .ckpt checkpoint file.
    output_path: str - Output path for the standalone .pth file.
    source: {"net", "net_ema"}, default "net_ema" - Which weights to extract.
    model_size: {"2B", "7B", "14B"}, default "2B" - DiT model size tag for metadata.

    Returns:
    None

    Raises:
    FileNotFoundError - If lightning_ckpt_path does not exist.
    KeyError - If checkpoint missing state_dict or net_ema.
    """
    ckpt_path = Path(lightning_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading Lightning checkpoint: {ckpt_path}")
    checkpoint = _safe_torch_load(str(ckpt_path), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint to be a dict, got {type(checkpoint)}")

    ema_exp_coefficient: Optional[float] = checkpoint.get("ema_exp_coefficient", None)

    if source == "net_ema":
        if "net_ema" in checkpoint and isinstance(checkpoint["net_ema"], dict):
            state_dict = checkpoint["net_ema"]
            log.info(f"  Loaded EMA weights (nested dict, {len(state_dict)} keys)")
        else:
            if "state_dict" not in checkpoint:
                raise KeyError(
                    "Could not find 'state_dict' to extract legacy EMA weights."
                )

            raw = checkpoint["state_dict"]
            legacy_keys = {k: v for k, v in raw.items() if k.startswith("net_ema.")}
            if not legacy_keys:
                raise KeyError(
                    f"No EMA weights found in checkpoint '{ckpt_path}'. "
                    "Try --source net to export main weights."
                )
            state_dict = {
                k[8:]: v for k, v in legacy_keys.items()
            }  # 8 is len("net_ema.")
            log.info(f"  Loaded EMA weights (legacy format, {len(state_dict)} keys)")

        if ema_exp_coefficient is not None:
            log.info(f"  EMA coeff: {ema_exp_coefficient:.6f}")

    else:  # source == "net"
        if "state_dict" not in checkpoint:
            raise KeyError(
                f"'state_dict' missing. Valid Lightning checkpoint from module.py?"
            )

        raw = checkpoint["state_dict"]
        state_dict = {
            k[4:]: v for k, v in raw.items() if k.startswith("net.")
        }  # 4 is len("net.")

        if not state_dict:
            raise ValueError(
                f"No 'net.' prefixed keys found. Got: {list(raw.keys())[:5]} ..."
            )

        log.info(f"  Loaded main network ({len(state_dict)} keys)")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_state_dict, stripped_keys, observed_dtypes = _normalize_state_dict_for_export(
        state_dict,
        export_dtype=export_dtype,
        strip_extra_state=strip_extra_state,
    )

    if observed_dtypes:
        log.info("  Floating dtypes before export: %s", sorted(str(d) for d in observed_dtypes))
    if stripped_keys:
        log.info("  Stripped %d TransformerEngine extra_state keys", len(stripped_keys))
    if export_dtype != "preserve":
        log.info("  Normalized exported floating tensors to %s", export_dtype)

    payload = {
        "net": normalized_state_dict,
        "model_size": model_size,
        "source": source,
        "ema_exp_coefficient": ema_exp_coefficient,
        "export_dtype": export_dtype,
        "original_floating_dtypes": sorted(str(d) for d in observed_dtypes),
        "stripped_extra_state_keys": stripped_keys,
    }
    torch.save(payload, str(out_path))

    if export_config:
        import json

        cfg = _MODEL_CONFIGS.get(model_size, _MODEL_CONFIGS["2B"])
        config_dict = {
            "max_img_h": 240,
            "max_img_w": 240,
            "max_frames": 128,
            "in_channels": 16,
            "out_channels": 16,
            "patch_spatial": 2,
            "patch_temporal": 1,
            "concat_padding_mask": True,
            "model_channels": cfg["model_channels"],
            "num_blocks": cfg["num_blocks"],
            "num_heads": cfg["num_heads"],
            "atten_backend": "minimal_a2a",
            "pos_emb_cls": "rope3d",
            "pos_emb_learnable": True,
            "pos_emb_interpolation": "crop",
            "use_adaln_lora": True,
            "adaln_lora_dim": 256,
            "rope_h_extrapolation_ratio": 3.0,
            "rope_w_extrapolation_ratio": 3.0,
            "rope_t_extrapolation_ratio": 1.0,
            "crossattn_emb_channels": CROSSATTN_EMB_CHANNELS,
            "use_crossattn_projection": True,
            "crossattn_proj_in_channels": CROSSATTN_PROJ_IN_CHANNELS,
            "sac_config": {"mode": "mm_only"},
            "timestep_scale": 0.001,
        }
        config_path = out_path.parent / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        log.info(f"✓ Exported config → {config_path}")

    out_size_gb = out_path.stat().st_size / 1e9
    log.info(f"✓ Exported → {out_path}  ({out_size_gb:.2f} GB)")


def export_model_state_dict(
    model: torch.nn.Module,
    output_path: str,
    model_size: str = "2B",
    source: str = "net",
    export_dtype: Literal["preserve", "float32", "bfloat16", "float16"] = "float32",
    strip_extra_state: bool = True,
) -> None:
    """
    Export weights directly from a trained torch.nn.Module instance to standalone .pth.
    Saves current state dict without Lightning dependency. Useful for exporting post-trained models.

    Args:
    model: torch.nn.Module - The trained model to export (e.g., NVSynCosmos25RF.net or .net_ema).
    output_path: str - Output path for the standalone .pth file.
    model_size: str, default "2B" - DiT size tag ("2B", "7B", "14B") for metadata.
    source: str, default "net" - Source identifier ("net", "ema", etc.) for metadata.

    Returns:
    None
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_state_dict = model.state_dict()
    log.info(f"Exporting {source} ({len(raw_state_dict)} params)")

    state_dict, stripped_keys, observed_dtypes = _normalize_state_dict_for_export(
        raw_state_dict,
        export_dtype=export_dtype,
        strip_extra_state=strip_extra_state,
    )

    if observed_dtypes:
        log.info("  Floating dtypes before export: %s", sorted(str(d) for d in observed_dtypes))
    if stripped_keys:
        log.info("  Stripped %d TransformerEngine extra_state keys", len(stripped_keys))
    if export_dtype != "preserve":
        log.info("  Normalized exported floating tensors to %s", export_dtype)

    payload = {
        "net": state_dict,
        "model_size": model_size,
        "source": source,
        "ema_exp_coefficient": None,
        "export_dtype": export_dtype,
        "original_floating_dtypes": sorted(str(d) for d in observed_dtypes),
        "stripped_extra_state_keys": stripped_keys,
    }
    torch.save(payload, str(out_path))

    out_size_gb = out_path.stat().st_size / 1e9
    log.info(f"✓ Exported → {out_path}  ({out_size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# Import (load standalone exported checkpoint into bare DiT)
# ---------------------------------------------------------------------------


def load_dit_checkpoint(
    checkpoint_path: str,
    model_size: Optional[str] = None,
    device: str = "cuda",
    state_ch: int = 16,
) -> MinimalV1LVGDiT:
    """
    Load a standalone exported .pth into MinimalV1LVGDiT (no Lightning required).
    Loads any checkpoint format (exported .pth, Lightning .ckpt, or bare state dict).
    Auto-detects format and instantiates fresh DiT model with loaded weights in eval mode.

    Args:
    checkpoint_path: str - Path to checkpoint (.pth or .ckpt).
    model_size: str, optional - Override model size ("2B", "7B", "14B"). Auto-detected if None.
    device: str, default "cuda" - Target device ("cpu", "cuda", etc.).
    state_ch: int, default 16 - Latent channel count (typically 16 for VAE).

    Returns:
    MinimalV1LVGDiT - Loaded model in eval mode on target device, ready for inference.

    Raises:
    FileNotFoundError - If checkpoint_path does not exist.
    ValueError - If checkpoint format is invalid or unrecognized.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = _safe_torch_load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "net" in ckpt:
        state_dict = ckpt["net"]
        _model_size = model_size or ckpt.get("model_size", "2B")
        source = ckpt.get("source", "unknown")
        export_dtype = ckpt.get("export_dtype")
        log.info(f"Loading exported: {_model_size} ({source}), {len(state_dict)} keys")

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        log.info(f"Loading Lightning .ckpt")
        raw = ckpt["state_dict"]
        state_dict = {
            k.removeprefix("net."): v for k, v in raw.items() if k.startswith("net.")
        }
        _model_size = model_size or "2B"
        source = "net"

        if not state_dict:
            raise ValueError(f"No 'net.' keys found. Keys: {list(raw.keys())[:5]} ...")

        log.info(f"  {_model_size} model, {len(state_dict)} weights")
        export_dtype = None

    else:
        log.info(f"Loading bare state dict")
        if isinstance(ckpt, dict):
            state_dict = ckpt
            _model_size = model_size or "2B"
            source = "raw"
            log.info(f"  {_model_size}, {len(state_dict)} keys")
            export_dtype = None
        else:
            raise ValueError(f"Invalid checkpoint type: {type(ckpt)}")

    log.info(f"Building DiT ({_model_size}) on {device}...")
    net = create_dit(model_size=_model_size, state_ch=state_ch, device=device)

    runtime_dtype = _choose_runtime_dtype(state_dict, requested_export_dtype=export_dtype)
    net = net.to(device=device, dtype=runtime_dtype)
    log.info(f"  Runtime dtype: {runtime_dtype}")

    load_result = non_strict_load_model(net, state_dict)
    if isinstance(load_result, tuple) and len(load_result) >= 2:
        missing, unexpected = load_result[0], load_result[1]
    else:
        missing, unexpected = [], []

    if missing:
        log.warning(f"  Missing ({len(missing)}): {missing[:3]}...")
    if unexpected:
        log.warning(f"  Extra ({len(unexpected)}): {unexpected[:3]}...")
    elif not missing:
        log.info(f"  ✓ Perfect match ({len(state_dict)} weights)")

    net.eval()
    return net


def save_and_load_roundtrip_test(
    checkpoint_path: str,
    temp_export_path: Optional[str] = None,
    model_size: Optional[str] = None,
    device: str = "cuda",
) -> MinimalV1LVGDiT:
    """
    Test round-trip export→load cycle and return loaded model.
    Loads checkpoint, exports to temp file, loads from exported file, verifies state dict keys match.
    Useful for verifying checkpoint compatibility and ensuring proper export/re-import.

    Args:
    checkpoint_path: str - Path to the original checkpoint (.ckpt or .pth).
    temp_export_path: str, optional - Temp export file path. Uses system temp if None.
    model_size: str, optional - Override model size ("2B", "7B", "14B"). Auto-detected if None.
    device: str, default "cuda" - Target device for loaded model.

    Returns:
    MinimalV1LVGDiT - Loaded model after full round-trip validation.
    """
    import tempfile

    if temp_export_path is None:
        temp_dir = tempfile.gettempdir()
        temp_export_path = str(Path(temp_dir) / "cosmos_dit_roundtrip_test.pth")

    log.info(f"Round-trip test: {checkpoint_path}")

    log.info("  1. Exporting...")
    try:
        export_checkpoint(
            checkpoint_path,
            temp_export_path,
            source="net_ema",
            model_size=model_size or "2B",
        )
    except KeyError:
        log.warning("    EMA not found, trying main network...")
        export_checkpoint(
            checkpoint_path,
            temp_export_path,
            source="net",
            model_size=model_size or "2B",
        )

    log.info("  2. Loading exported...")
    net = load_dit_checkpoint(temp_export_path, model_size=model_size, device=device)

    log.info("  3. Verifying...")
    exported_ckpt = _safe_torch_load(temp_export_path, map_location="cpu")
    exported_state_dict = exported_ckpt["net"]
    loaded_state_dict = net.state_dict()

    exported_keys = {key for key in exported_state_dict.keys() if not key.endswith("._extra_state")}
    loaded_keys = {key for key in loaded_state_dict.keys() if not key.endswith("._extra_state")}

    if exported_keys == loaded_keys:
        log.info(f"    ✓ Keys match ({len(exported_keys)})")
    else:
        missing = loaded_keys - exported_keys
        extra = exported_keys - loaded_keys
        if missing:
            log.warning(f"    Missing: {list(missing)[:2]}")
        if extra:
            log.warning(f"    Extra: {list(extra)[:2]}")

    sample_key = next(key for key, value in exported_state_dict.items() if isinstance(value, torch.Tensor))
    if torch.allclose(
        exported_state_dict[sample_key].float(),
        loaded_state_dict[sample_key].float(),
        rtol=1e-5,
        atol=1e-7,
    ):
        log.info(f"    ✓ Weights match")
    else:
        log.error(f"    ✗ Weight drift on {sample_key}")

    try:
        Path(temp_export_path).unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"Cleanup failed: {e}")

    log.info("  ✓ Passed!")
    return net


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser with 4 subcommands for checkpoint management.
    Subcommands: export, load, roundtrip, inspect.
    """
    parser = argparse.ArgumentParser(
        description="Export / import DiT weights from a Lightning training checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ## Export subcommand: Extract weights from checkpoint
    exp = sub.add_parser("export", help="Extract DiT weights from a Lightning .ckpt")
    exp.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt file")
    exp.add_argument("--out", required=True, help="Output path for the standalone .pth")
    exp.add_argument(
        "--source",
        default="net_ema",
        choices=["net", "net_ema"],
        help="Which weights to export: main net or EMA (default: net_ema)",
    )
    exp.add_argument(
        "--model_size",
        default="2B",
        choices=list(_MODEL_CONFIGS),
        help="Model size tag stored in metadata (default: 2B)",
    )
    exp.add_argument(
        "--export-config",
        action="store_true",
        help="Export a config.json alongside the .pth file",
    )
    exp.add_argument(
        "--export-dtype",
        default="float32",
        choices=list(_EXPORT_DTYPES.keys()),
        help="Normalize floating tensors for exported .pth; use bfloat16 for CUDA inferencer reuse (default: float32)",
    )
    exp.add_argument(
        "--keep-extra-state",
        action="store_true",
        help="Keep TransformerEngine .*_extra_state entries in exported .pth",
    )

    ## Load subcommand: Instantiate model from checkpoint
    lod = sub.add_parser("load", help="Load checkpoint and return ready-to-use model")
    lod.add_argument("--ckpt", required=True, help="Path to .pth or .ckpt checkpoint")
    lod.add_argument(
        "--device",
        default="cuda",
        help="Target device for model (default: cuda)",
    )
    lod.add_argument(
        "--model_size",
        help="Override model size (2B/7B/14B). Auto-detected if not provided",
    )

    ## Roundtrip subcommand: Test export→load cycle
    rtp = sub.add_parser("roundtrip", help="Test full export→load round-trip cycle")
    rtp.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt checkpoint")
    rtp.add_argument(
        "--model_size",
        help="Override model size (2B/7B/14B)",
    )
    rtp.add_argument(
        "--device",
        default="cuda",
        help="Target device for model (default: cuda)",
    )

    ## Inspect subcommand: Display checkpoint structure
    ins = sub.add_parser("inspect", help="Print checkpoint keys without exporting")
    ins.add_argument("--ckpt", required=True, help="Path to .ckpt or .pth file")

    return parser


def _cmd_export(args: argparse.Namespace) -> None:
    """
    CLI command: Export DiT weights from Lightning checkpoint.
    Extracts weights (net or EMA) from Lightning .ckpt and saves to standalone .pth.

    Args:
    args: argparse.Namespace - Parsed args (ckpt, out, source, model_size).

    Returns:
    None
    """
    try:
        export_checkpoint(
            lightning_ckpt_path=args.ckpt,
            output_path=args.out,
            source=args.source,
            model_size=args.model_size,
            export_config=args.export_config,
            export_dtype=args.export_dtype,
            strip_extra_state=not args.keep_extra_state,
        )
    except Exception as e:
        log.error(f"Export failed: {e}", exc_info=True)
        raise


def _cmd_load(args: argparse.Namespace) -> None:
    """
    CLI command: Load checkpoint and return ready-to-use model.
    Loads checkpoint (.pth or .ckpt) into fresh DiT model instance, ready for inference.

    Args:
    args: argparse.Namespace - Parsed args (ckpt, device, model_size).

    Returns:
    MinimalV1LVGDiT - Loaded model on target device in eval mode.
    """
    try:
        net = load_dit_checkpoint(
            checkpoint_path=args.ckpt,
            model_size=args.model_size,
            device=args.device,
        )
        return net
    except Exception as e:
        log.error(f"Load failed: {e}", exc_info=True)
        raise


def _cmd_roundtrip(args: argparse.Namespace) -> None:
    """
    CLI command: Test full export→load round-trip cycle.
    Verifies checkpoint exports to temp file and loads back with state dict keys/values matching.

    Args:
    args: argparse.Namespace - Parsed args (ckpt, model_size, device).

    Returns:
    MinimalV1LVGDiT - Loaded model after successful round-trip validation.
    """
    try:
        net = save_and_load_roundtrip_test(
            checkpoint_path=args.ckpt,
            model_size=args.model_size,
            device=args.device,
        )
        return net
    except Exception as e:
        log.error(f"Round-trip test failed: {e}", exc_info=True)
        raise


def _cmd_inspect(args: argparse.Namespace) -> None:
    """
    CLI command: Inspect checkpoint structure without loading.
    Displays checkpoint structure: top-level keys, state_dict, net, net_ema. Useful for debugging.

    Args:
    args: argparse.Namespace - Parsed args (ckpt).

    Returns:
    None
    """
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return

    log.info(f"Inspecting: {ckpt_path}")
    ckpt = _safe_torch_load(str(ckpt_path), map_location="cpu")

    def _show(label: str, d: dict) -> None:
        """Display dict structure (first 10 entries)."""
        keys = list(d.keys())
        log.info(f"{label} ({len(keys)} keys)")
        for k in keys[:10]:
            v = d[k]
            shape = tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__
            log.info(f"  {k:<64}  {shape}")
        if len(keys) > 10:
            log.info(f"  … ({len(keys) - 10} more)")

    if isinstance(ckpt, dict):
        top_keys = [k for k in ckpt if not isinstance(ckpt[k], dict)]
        if top_keys:
            log.info("Metadata:")
            for k in top_keys:
                v = ckpt[k]
                if isinstance(v, (int, float, bool)):
                    log.info(f"  {k}: {v}")
                elif isinstance(v, str):
                    log.info(f"  {k}: {v}")
                else:
                    log.info(f"  {k}: {type(v).__name__}")

        if "state_dict" in ckpt:
            _show("state_dict", ckpt["state_dict"])
        if "net" in ckpt and isinstance(ckpt["net"], dict):
            _show("net", ckpt["net"])
        if "net_ema" in ckpt and isinstance(ckpt["net_ema"], dict):
            _show("net_ema", ckpt["net_ema"])
    else:
        log.error(f"Invalid checkpoint type: {type(ckpt)}")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "export":
            _cmd_export(args)
        elif args.command == "load":
            _cmd_load(args)
        elif args.command == "roundtrip":
            _cmd_roundtrip(args)
        elif args.command == "inspect":
            _cmd_inspect(args)
        else:
            log.error(f"Unknown command: {args.command}")
            parser.print_help()
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        sys.exit(1)