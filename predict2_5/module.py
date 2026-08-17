"""Cosmos-Predict 2.5 post-training module."""

import contextlib
import os

from predict2_5.utils import setup_early_logging
setup_early_logging()

from typing import Literal, Optional

import numpy as np

import torch
torch.set_float32_matmul_precision("high")

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lightning import LightningModule

from cosmos_predict2._src.imaginaire.utils.ema import FastEmaModelUpdater
from cosmos_predict2._src.imaginaire.utils.checkpointer import non_strict_load_model
from cosmos_predict2._src.predict2.utils.optim_instantiate import get_base_optimizer
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import download_checkpoint

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.configs.video2world.defaults.conditioner import (
    Video2WorldCondition,
)
from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow
from cosmos_predict2._src.predict2.models.fm_solvers_unipc import (
    FlowUniPCMultistepScheduler,
)

from predict2_5.constants import (
    NUM_LATENT_FRAMES,
    COSMOS_TOKENIZER_UUID,
    COSMOS_2B_PRETRAINED_UUID,
    CR1_EMBEDDING_DIM,
    CR1_MAX_LENGTH,
    CROSSATTN_EMB_CHANNELS,
    CROSSATTN_PROJ_IN_CHANNELS,
    NUM_FRAMES,
)

from predict2_5.utils import (
    get_local_rank,
    is_distributed,
    get_rank,
    get_world_size,
    get_logger,
    broadcast_model_states,
    broadcast_model_states_packed,
    sync_ema_ddp,
    arch_invariant_rand,
    fix_rope_buffers,
    move_tokenizer_to_device,
)
from predict2_5.text_encoder import CR1TextEncoder


# --- Logger --- #
logger = get_logger(__name__)


# Module-level initialization guards
_CHECKPOINT_RESOLVED = {}
_TOKENIZER_RESOLVED = {}


class CosmosXRay360(LightningModule):
    """Cosmos-Predict 2.5 Rectified Flow for video synthesis with PyTorch Lightning."""

    _setup_complete: bool = False

    def __init__(
        self,
        checkpoint_uuid: str = COSMOS_2B_PRETRAINED_UUID,
        tokenizer_uuid: str = COSMOS_TOKENIZER_UUID,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        tokenizer_chunk_duration: int = 93,
        tokenizer_temporal_window: int = 16,
        text_encoder_path: Optional[str] = None,
        learning_rate: float = 2 ** (-14.5),
        weight_decay: float = 0.001,
        warmup_steps: int = 2000,
        max_iters: int = 100000,
        loss_scale: float = 1.0,
        enable_ema: bool = True,
        ema_rate: float = 0.10,
        ema_offload_cpu: bool = False,
        ema_iteration_shift: int = 0,
        gradient_clip_val: float = 1.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 1.5,
        rf_shift: float = 5.0,
        model_size: str = "2B",
        state_t: int = NUM_LATENT_FRAMES,
        state_ch: int = 16,
        distributed_strategy: Literal["auto", "ddp", "fsdp"] = "auto",
        ema_sync_every_n_steps: int = 1,
        cfg_dropout_rate: float = 0.2,
        # Conditional frames configuration
        # NOTE: These are LATENT frame counts, not pixel frames!
        # With 93 pixel frames → 24 latent frames, so 1-2 latent = ~4-8% conditioning
        min_num_conditional_frames: int = 1,
        max_num_conditional_frames: int = 2,
        # Timestep for conditioning frames (official default: -1.0 = disabled)
        # When >= 0, sets a very low noise level for cond frames (e.g., 0.0 = clean)
        conditional_frame_timestep: float = -1.0,
        sac_mode: str = "predict2_2b_720_aggressive",
        # Physics & Geometric regularization
        gamma_side: float = 1.0,
        loss_atten_weight: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        local_rank = get_local_rank()
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.tensor_kwargs = {"device": device_str, "dtype": torch.float32}

        self._setup_rectified_flow()
        self._setup_tokenizer()
        self._setup_network()
        self._setup_ema()

        self.prompt_encoder = CR1TextEncoder(
            text_encoder_ckpt_path=text_encoder_path,
            device=device_str,
            cpu_offload=True,
        )

    @property
    def net_ema(self) -> Optional[MinimalV1LVGDiT]:
        """Return registered net_ema submodule."""
        return self._modules.get("net_ema", None)

    @net_ema.setter
    def net_ema(self, value: Optional[MinimalV1LVGDiT]) -> None:
        """Set net_ema as a registered submodule."""
        if value is None:
            self._modules.pop("net_ema", None)
        else:
            self.add_module("net_ema", value)

    def _setup_rectified_flow(self):
        """Initialize flow scheduler and sampler."""
        self.rectified_flow = RectifiedFlow(
            velocity_field=lambda *a, **k: None,
            train_time_distribution="logitnormal",
            train_time_weight_method="uniform",
            use_dynamic_shift=False,
            shift=self.hparams.rf_shift,
            device=self.device if hasattr(self, "device") else torch.device("cpu"),
            dtype=torch.float32,
        )
        self._num_train_timesteps = self.rectified_flow.num_train_timesteps

        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

    def _setup_tokenizer(self):
        """Initialize VAE tokenizer."""
        global _TOKENIZER_RESOLVED

        os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
        tokenizer_path = None

        if self.hparams.tokenizer_path and os.path.exists(self.hparams.tokenizer_path):
            tokenizer_path = self.hparams.tokenizer_path
        elif self.hparams.tokenizer_uuid:
            uuid = self.hparams.tokenizer_uuid
            try:
                if uuid not in _TOKENIZER_RESOLVED:
                    from cosmos_oss.checkpoints_predict2 import register_checkpoints
                    register_checkpoints()
                    tokenizer_path = download_checkpoint(uuid)
                    _TOKENIZER_RESOLVED[uuid] = tokenizer_path
                tokenizer_path = _TOKENIZER_RESOLVED[uuid]
            except Exception as exc:
                if get_local_rank() == 0:
                    logger.warning(f"Tokenizer checkpoint download failed ({exc}). Using unweighted tokenizer.")

        try:
            self.tokenizer = Wan2pt1VAEInterface(
                chunk_duration=self.hparams.tokenizer_chunk_duration,
                load_mean_std=False,
                vae_pth=tokenizer_path or "",
                temporal_window=self.hparams.tokenizer_temporal_window,
                keep_decoder_cache=False,
                keep_encoder_cache=False,
            )
        except Exception as exc:
            if get_local_rank() == 0:
                logger.warning(f"Failed to instantiate Wan2pt1VAEInterface ({exc}). Creating mock tokenizer.")
            self.tokenizer = None

        if self.tokenizer is not None:
            assert self.tokenizer.latent_ch == self.hparams.state_ch
            local_rank = get_local_rank()
            device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            move_tokenizer_to_device(self.tokenizer, device_str)

    def _get_model_config(self) -> dict:
        """Get DiT config for model size (2B/7B/14B)."""
        configs = {
            "2B": {"model_channels": 2048, "num_heads": 16, "num_blocks": 28},
            "7B": {"model_channels": 4096, "num_heads": 32, "num_blocks": 28},
            "14B": {"model_channels": 5120, "num_heads": 40, "num_blocks": 36},
        }
        return configs[self.hparams.model_size]

    def _create_dit(self, device: str = "meta") -> MinimalV1LVGDiT:
        """Create DiT network (use 'meta' for deferred init)."""
        config = self._get_model_config()

        with torch.device(device):
            net = MinimalV1LVGDiT(
                max_img_h=240,
                max_img_w=240,
                max_frames=128,
                in_channels=self.hparams.state_ch,
                out_channels=self.hparams.state_ch,
                patch_spatial=2,
                patch_temporal=1,
                concat_padding_mask=True,
                model_channels=config["model_channels"],
                num_blocks=config["num_blocks"],
                num_heads=config["num_heads"],
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
                sac_config=SACConfig(mode=getattr(self.hparams, "sac_mode", "predict2_2b_720_aggressive")),
                timestep_scale=0.001,
            )
        return net

    def _setup_network(self):
        """Initialize DiT and load pretrained weights."""
        self.net = self._create_dit(device="meta")

        local_rank = get_local_rank()
        init_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.net.to_empty(device=init_device)
        self.net.init_weights()
        fix_rope_buffers(self.net)

        os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")
        checkpoint_path = None

        if self.hparams.checkpoint_path and os.path.exists(self.hparams.checkpoint_path):
            checkpoint_path = self.hparams.checkpoint_path
        elif self.hparams.checkpoint_uuid:
            uuid = self.hparams.checkpoint_uuid
            try:
                if uuid not in _CHECKPOINT_RESOLVED:
                    from cosmos_oss.checkpoints_predict2 import register_checkpoints
                    register_checkpoints()
                    checkpoint_path = download_checkpoint(uuid)
                    _CHECKPOINT_RESOLVED[uuid] = checkpoint_path
                checkpoint_path = _CHECKPOINT_RESOLVED[uuid]
            except Exception as exc:
                if get_local_rank() == 0:
                    logger.warning(f"DiT checkpoint download failed ({exc}). Model initialized with random weights.")

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "net" in ckpt:
                state_dict = ckpt["net"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            if isinstance(state_dict, dict):
                state_dict = {
                    k.replace("._checkpoint_wrapped_module", "").removeprefix("net."): v
                    for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in ["prompt_encoder.", "tokenizer."])
                }
            _ = non_strict_load_model(self.net, state_dict)
        else:
            if get_local_rank() == 0:
                logger.warning("No checkpoint loaded! Model initialized with random weights.")

        self.net.train()
        self.net.requires_grad_(True)

    def _setup_ema(self):
        """Initialize EMA model (Power EMA from EDM2 paper)."""
        if self.hparams.enable_ema:
            # Use factory method to create identical architecture
            net_ema_module = self._create_dit(device="meta")

            local_rank = get_local_rank()
            target_device = "cpu" if (self.hparams.ema_offload_cpu and self.hparams.distributed_strategy != "fsdp") else (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            net_ema_module.to_empty(device=target_device)
            net_ema_module.init_weights()
            fix_rope_buffers(net_ema_module)

            # EMA must be in float32 for stable accumulation
            net_ema_module.to(dtype=torch.float32, device=target_device)
            net_ema_module.eval()
            net_ema_module.requires_grad_(False)

            self.ema_updater = FastEmaModelUpdater()

            # Power EMA coefficient (EDM2 paper)
            # Formula: β(k) = (1 - 1/(k+1))^(γ+1) where γ solves cubic equation
            s = self.hparams.ema_rate
            self.ema_exp_coefficient = np.roots(
                [1, 7, 16 - s**-2, 12 - s**-2]
            ).real.max()

            # Copy initial weights from net to net_ema
            with torch.no_grad():
                for p_ema, p_net in zip(
                    net_ema_module.parameters(), self.net.parameters()
                ):
                    p_ema.data.copy_(p_net.data.to(target_device))

            self.net_ema = net_ema_module

            if self.hparams.distributed_strategy == "fsdp" and get_local_rank() == 0:
                logger.info(
                    f"[Cosmos25] net_ema is registered as an FSDP-sharded submodule on GPU cuda:{local_rank}"
                )
        else:
            self.net_ema = None
            self.ema_updater = None
            self.ema_exp_coefficient = None

    def _ensure_tokenizer_device(self):
        """Move tokenizer to correct device (once per rank)."""
        if not hasattr(self, "_tokenizer_device_set"):
            self._tokenizer_device_set = False

        if not self._tokenizer_device_set:
            move_tokenizer_to_device(self.tokenizer, self.device)
            self._tokenizer_device_set = True

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video [B,C,T,H,W] to latent space."""
        if self.tokenizer is None:
            B, C, T, H, W = video.shape
            T_latent = 1 + (T - 1) // 4
            return torch.zeros(
                (B, self.hparams.state_ch, T_latent, H // 8, W // 8),
                device=video.device,
                dtype=torch.float32,
            )
        
        self._ensure_tokenizer_device()
        
        video_f32 = video.float() if video.dtype != torch.float32 else video
        latent = self.tokenizer.encode(video_f32)
        return latent.float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video space.
        Output range: [-1, 1] (validated during inference).
        """
        if self.tokenizer is None:
            B, C, T_latent, H_latent, W_latent = latent.shape
            T_pixel = (T_latent - 1) * 4 + 1
            return torch.zeros(
                (B, 3, T_pixel, H_latent * 8, W_latent * 8),
                device=latent.device,
                dtype=torch.float32,
            )
        
        self._ensure_tokenizer_device()

        latent_f32 = latent.float() if latent.dtype != torch.float32 else latent
        video = self.tokenizer.decode(latent_f32)
        video_f32 = video.float()

        # Validate range during inference
        if not self.training and not torch.is_grad_enabled():
            v_min, v_max = video_f32.min().item(), video_f32.max().item()
            assert v_min >= -1.5 and v_max <= 1.5, (
                f"VAE decode range error: [{v_min:.3f}, {v_max:.3f}], expected [-1, 1]"
            )

        return video_f32

    def encode_prompt(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """
        Encode text prompts: try loading from .pkl first, fallback to on-the-fly encoding.
        """
        self.prompt_encoder.device = str(device)
        return self.prompt_encoder.encode_prompts(prompts, device=device)

    def encode_prompt_on_the_fly(
        self, prompts: list[str], device: torch.device
    ) -> torch.Tensor:
        """
        Stub for on-the-fly text encoding. To be implemented by user.
        """
        # Placeholder: Return zeros
        # Users should instantiate their text encoder and replace this logic
        # e.g., return self.text_encoder.compute_text_embeddings_online(prompts)
        B = len(prompts)
        logger.warning(
            "On-the-fly text encoding is not implemented. Returning zero embeddings."
        )
        return torch.zeros(
            (B, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM), dtype=torch.float32, device=device
        )

    def _process_batch(self, batch: dict, stage: str = "train") -> dict:
        """Extract and normalize batch data for CT/video training and validation."""
        text_embeddings = batch.get("text_embeddings")
        video = batch.get("video")
        if video is None:
            video = batch.get("ct")
        pre_cached_latent = batch.get("pre_cached_latent")
        if pre_cached_latent is None:
            pre_cached_latent = batch.get("latent")

        prompts = batch.get("prompt")

        if pre_cached_latent is not None:
            B = pre_cached_latent.shape[0]
            device = pre_cached_latent.device
        elif video is not None:
            B = video.shape[0]
            device = video.device
        elif text_embeddings is not None:
            B = text_embeddings.shape[0]
            device = text_embeddings.device
        elif prompts is not None:
            B = len(prompts)
            device = self.device
        else:
            raise ValueError(
                "Batch must contain at least one of: pre_cached_latent, latent, video, ct, text_embeddings, or prompt"
            )

        if text_embeddings is None:
            if prompts is not None:
                text_embeddings = self.encode_prompt(prompts, device=device)
            else:
                text_embeddings = torch.zeros(
                    (B, CR1_MAX_LENGTH, CR1_EMBEDDING_DIM),
                    device=device,
                    dtype=torch.float32,
                )
        else:
            text_embeddings = text_embeddings.to(device=device, dtype=torch.float32)

        if video is not None:
            if video.dtype == torch.uint8:
                video = video.float() / 255.0
            else:
                video = video.to(device=device, dtype=torch.float32)
            if video.dim() == 5 and video.shape[1] == 1:
                video = video.repeat(1, 3, 1, 1, 1)

        if pre_cached_latent is not None:
            target_dtype = next(self.net.parameters()).dtype
            pre_cached_latent = pre_cached_latent.to(device=device, dtype=target_dtype)

        return {
            "video": video,
            "pre_cached_latent": pre_cached_latent,
            "text_embeddings": text_embeddings,
        }

    def _apply_cfg_dropout_per_sample(
        self,
        text_embeddings: torch.Tensor,
        dropout_rate: float = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-sample CFG text dropout (returns embeddings + mask)."""
        if dropout_rate is None:
            dropout_rate = self.hparams.cfg_dropout_rate

        B = text_embeddings.shape[0]
        device = text_embeddings.device

        if dropout_rate <= 0.0 or not self.training:
            return text_embeddings, torch.ones(B, dtype=torch.bool, device=device)

        # Per-sample Bernoulli mask: 1 = keep, 0 = drop
        # P(keep) = 1 - dropout_rate, applied INDEPENDENTLY per sample
        keep_mask_flat = torch.bernoulli(
            (1.0 - dropout_rate) * torch.ones(B, device=device)
        )  # [B]
        keep_mask = keep_mask_flat.view(B, 1, 1)  # [B, 1, 1] for broadcasting

        # Zero out dropped samples (official: use_empty_string=False → zero tensor)
        text_out = text_embeddings * keep_mask

        return text_out, keep_mask_flat.bool()

    def _get_condition(
        self,
        text_embeddings: torch.Tensor,
        latent: torch.Tensor,
        num_conditional_frames: int = None,
        use_video_condition: bool = True,
        dtype: torch.dtype = None,
        apply_cfg_dropout: bool = False,
        input_height: int = None,
        input_width: int = None,
    ) -> Video2WorldCondition:
        """
        Build Video2World conditioning with per-sample CFG dropout.
        num_conditional_frames: None=random, int=fixed (LATENT frame count).
        """
        B, C, T, H, W = latent.shape
        device = latent.device

        if dtype is None:
            dtype = next(self.net.parameters()).dtype

        # Per-sample CFG dropout
        conditional_frames_probs = None
        if apply_cfg_dropout and self.training:
            text_embeddings, _ = self._apply_cfg_dropout_per_sample(text_embeddings)

            if use_video_condition:
                dropout_rate = self.hparams.cfg_dropout_rate
                min_cf = self.hparams.min_num_conditional_frames
                max_cf = self.hparams.max_num_conditional_frames
                num_cond_options = max_cf - min_cf + 1
                keep_prob_per_option = (1.0 - dropout_rate) / num_cond_options

                conditional_frames_probs = {0: dropout_rate}
                for nf in range(min_cf, max_cf + 1):
                    conditional_frames_probs[nf] = keep_prob_per_option

        # Padding mask from actual dimensions
        H_pixel_full = H * 8
        W_pixel_full = W * 8

        if input_height is not None and input_width is not None:
            padding_mask = torch.ones(
                B, 1, H_pixel_full, W_pixel_full, device=device, dtype=dtype
            )
            padding_mask[:, :, :input_height, :input_width] = 0.0
        else:
            padding_mask = torch.zeros(
                B, 1, H_pixel_full, W_pixel_full, device=device, dtype=dtype
            )

        fps = torch.full((B,), 24.0, device=device, dtype=dtype)

        base_condition = Video2WorldCondition(
            crossattn_emb=text_embeddings.to(device=device, dtype=dtype),
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.VIDEO,
            use_video_condition=use_video_condition,
        )

        condition = base_condition.set_video_condition(
            gt_frames=latent.to(dtype=dtype),
            random_min_num_conditional_frames=self.hparams.min_num_conditional_frames,
            random_max_num_conditional_frames=self.hparams.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=conditional_frames_probs,
        )

        return condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Video2WorldCondition,
    ) -> torch.Tensor:
        """
        Predict velocity using FRAME_REPLACE conditioning.
        Replaces conditioning frames in input and GT velocity in output.
        """
        B, C, T, H, W = xt_B_C_T_H_W.shape
        condition_video_mask = None

        gt_frames = condition.gt_frames
        mask = condition.condition_video_input_mask_B_C_T_H_W

        if condition.is_video and gt_frames is not None and mask is not None:
            condition_state_in = gt_frames.type_as(xt_B_C_T_H_W)

            use_video_cond = condition.use_video_condition
            if isinstance(use_video_cond, torch.Tensor):
                assert bool((use_video_cond == use_video_cond[0]).all().item())
                use_video_cond = bool(use_video_cond[0].item())

            if not use_video_cond:
                condition_state_in = condition_state_in * 0

            condition_video_mask = mask.repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            xt_B_C_T_H_W = condition_state_in * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            conditional_frame_timestep = getattr(
                self.hparams, "conditional_frame_timestep", -1.0
            )
            if conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(
                    dim=[1, 3, 4], keepdim=True
                )
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1)
                    * conditional_frame_timestep
                )
                timesteps_B_T = (
                    timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1
                    + timesteps_B_T.view(B, 1, 1, 1, 1)
                    * (1 - condition_video_mask_B_1_T_1_1)
                )
                timesteps_B_T = timesteps_B_T.view(B, T)

        target_device = next(self.net.parameters()).device
        target_dtype = next(self.net.parameters()).dtype
        cond_dict = condition.to_dict()
        for k, v in cond_dict.items():
            if isinstance(v, torch.Tensor):
                cond_dict[k] = v.to(device=target_device)

        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(device=target_device, dtype=target_dtype),
            timesteps_B_T=timesteps_B_T.to(device=target_device, dtype=target_dtype),
            **cond_dict,
        ).float()

        # Replace velocity for conditioning frames with GT
        if condition.is_video and condition_video_mask is not None:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise.type_as(net_output_B_C_T_H_W) - gt_frames_x0
            net_output_B_C_T_H_W = (
                gt_frames_velocity * condition_video_mask
                + net_output_B_C_T_H_W * (1 - condition_video_mask)
            )

        return net_output_B_C_T_H_W

    def ema_beta(self, iteration: int) -> float:
        """Power EMA coefficient (EDM2 formula)."""
        iteration = iteration + self.hparams.ema_iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def _is_fsdp(self) -> bool:
        """Check if distributed training is using FSDP strategy or if network is FSDP wrapped."""
        if self.hparams.distributed_strategy == "fsdp":
            return True
        if hasattr(self, "trainer") and self.trainer is not None:
            if "fsdp" in str(type(self.trainer.strategy)).lower():
                return True
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(self.net, FSDP):
                return True
            for m in self.net.modules():
                if isinstance(m, FSDP):
                    return True
        except ImportError:
            pass
        
        return False

    def on_before_zero_grad(self, optimizer):
        """Update and sync EMA model."""
        if self.hparams.enable_ema and self.net_ema is not None:
            ema_beta = self.ema_beta(self.global_step)

            with torch.no_grad():
                if self._is_fsdp():
                    # FSDP-sharded net_ema: pure local elementwise update (zero communication)
                    self.ema_updater.update_average(self.net, self.net_ema, beta=ema_beta)
                else:
                    if self.hparams.ema_offload_cpu:
                        self.net_ema.to(self.device)
                        torch.cuda.synchronize(self.device)

                    self.ema_updater.update_average(self.net, self.net_ema, beta=ema_beta)

                    # Sync across ranks (with safety measures)
                    if self.hparams.distributed_strategy == "ddp" and get_world_size() > 1:
                        # Only sync if not offloading to CPU (to avoid NCCL hangs)
                        if not self.hparams.ema_offload_cpu:
                            sync_ema_ddp(
                                self.net_ema,
                                sync_every_n_steps=self.hparams.ema_sync_every_n_steps,
                                current_step=self.global_step,
                                logger=logger,
                            )
                        else:
                            # CPU offload + DDP sync is prone to hangs, skip sync
                            if get_rank() == 0 and self.global_step % 100 == 0:
                                logger.warning(
                                    "[EMA] CPU offload enabled, skipping DDP sync to prevent hangs"
                                )

                    # Offload to CPU after sync
                    if self.hparams.ema_offload_cpu:
                        torch.cuda.synchronize(self.device)
                        self.net_ema.to("cpu")

    @contextlib.contextmanager
    def ema_scope(self):
        """Swap to EMA model for validation (safe state save/restore)."""
        if self.hparams.enable_ema and self.net_ema is not None:
            if self.global_step >= self.hparams.ema_iteration_shift:
                original_net = self.net
                original_dtype = next(self.net.parameters()).dtype

                if not self._is_fsdp() and self.hparams.ema_offload_cpu:
                    self.net_ema.to(self.device)

                self.net_ema.to(dtype=original_dtype)
                self.net = self.net_ema
                try:
                    yield
                finally:
                    self.net = original_net
                    self.net_ema.to(dtype=torch.float32)

                    if not self._is_fsdp() and self.hparams.ema_offload_cpu:
                        self.net_ema.to("cpu")
            else:
                yield
        else:
            yield

    @contextlib.contextmanager
    def ema_scope_generation(self):
        """Swap to EMA (or convert model to bf16) for generation."""
        use_ema = (
            self.hparams.enable_ema
            and self.net_ema is not None
            and self.global_step >= self.hparams.ema_iteration_shift
        )

        if use_ema:
            original_net = self.net

            if not self._is_fsdp() and self.hparams.ema_offload_cpu:
                self.net_ema.to(self.device)

            self.net_ema.to(dtype=torch.bfloat16)
            self.net = self.net_ema
            try:
                yield
            finally:
                self.net = original_net
                self.net_ema.to(dtype=torch.float32)

                if not self._is_fsdp() and self.hparams.ema_offload_cpu:
                    self.net_ema.to("cpu")
        else:
            original_dtype = next(self.net.parameters()).dtype
            self.net.to(dtype=torch.bfloat16)
            try:
                yield
            finally:
                self.net.to(dtype=original_dtype)

    def configure_optimizers(self):
        """Setup optimizer and LR scheduler (FusedAdam + warmup + cosine)."""
        optimizer = get_base_optimizer(
            model=self.net,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            optim_type="fusedadam",
            betas=(0.9, 0.99),
            eps=1e-8,
            master_weights=False,
            capturable=True,
        )

        warmup_steps = min(self.hparams.warmup_steps, self.hparams.max_iters // 5)

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-6, end_factor=0.5, total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_iters - warmup_steps,
            eta_min=self.hparams.learning_rate * 0.2,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save EMA weights (Lightning doesn't save them by default)."""
        if self.hparams.enable_ema and self.net_ema is not None:
            if self._is_fsdp():
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig

                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                fsdp_modules = [m for m in self.net_ema.modules() if isinstance(m, FSDP)]
                if fsdp_modules:
                    with contextlib.ExitStack() as stack:
                        for mod in fsdp_modules:
                            stack.enter_context(FSDP.state_dict_type(mod, StateDictType.FULL_STATE_DICT, save_policy))
                        raw_ema_sd = self.net_ema.state_dict()
                else:
                    raw_ema_sd = self.net_ema.state_dict()
            else:
                raw_ema_sd = self.net_ema.state_dict()

            if get_local_rank() == 0:
                clean_ema_sd = {
                    k.removeprefix("_forward_module.").removeprefix("_orig_mod.").removeprefix("net_ema.").removeprefix("net."): v
                    for k, v in raw_ema_sd.items()
                }
                checkpoint["net_ema"] = clean_ema_sd
                checkpoint["ema_exp_coefficient"] = self.ema_exp_coefficient
                logger.info(
                    f"[Cosmos25] Saved EMA weights ({len(checkpoint['net_ema'])} keys)"
                )

    def _load_ema_state_dict(self, state_dict: dict) -> None:
        """Load state dict into net_ema, handling FSDP sharding if active."""
        if self._is_fsdp():
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            fsdp_modules = [m for m in self.net_ema.modules() if isinstance(m, FSDP)]
            if fsdp_modules:
                with contextlib.ExitStack() as stack:
                    for mod in fsdp_modules:
                        stack.enter_context(FSDP.state_dict_type(mod, StateDictType.FULL_STATE_DICT, save_policy))
                    self.net_ema.load_state_dict(state_dict, strict=False)
                return
        self.net_ema.load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load EMA weights from checkpoint (backward compatible)."""
        if self.hparams.enable_ema and self.net_ema is not None:
            # Try nested dict format first (new, preferred)
            if "net_ema" in checkpoint and isinstance(checkpoint["net_ema"], dict):
                if get_local_rank() == 0:
                    logger.info(
                        f"[Cosmos25] Found EMA weights in checkpoint ({len(checkpoint['net_ema'])} keys), loading..."
                    )

                try:
                    self._load_ema_state_dict(checkpoint["net_ema"])

                    # Restore EMA metadata if available
                    if "ema_exp_coefficient" in checkpoint:
                        self.ema_exp_coefficient = checkpoint["ema_exp_coefficient"]

                    if get_local_rank() == 0:
                        logger.info("[Cosmos25] ✓ Loaded EMA model")
                except Exception as e:
                    if get_local_rank() == 0:
                        logger.warning(f"[Cosmos25] Could not load EMA weights: {e}")

            # Fallback: try flattened keys format (old checkpoints)
            elif any(k.startswith("net_ema.") for k in checkpoint.keys()):
                if get_local_rank() == 0:
                    logger.info("[Cosmos25] Found EMA weights (legacy format), loading...")

                try:
                    ema_state_dict = {
                        k.replace("net_ema.", ""): v
                        for k, v in checkpoint.items()
                        if k.startswith("net_ema.")
                    }
                    self._load_ema_state_dict(ema_state_dict)

                    if get_local_rank() == 0:
                        logger.info("[Cosmos25] ✓ Loaded EMA model (legacy format)")
                except Exception as e:
                    if get_local_rank() == 0:
                        logger.warning(f"[Cosmos25] Could not load EMA weights: {e}")
            else:
                if get_local_rank() == 0:
                    logger.info(
                        "[Cosmos25] No EMA weights in checkpoint (old checkpoint format)"
                    )

    def on_train_start(self):
        """Setup tokenizer device, rectified flow device, and EMA dtype."""
        self._ensure_tokenizer_device()
        self.rectified_flow.device = self.device

        if self.hparams.enable_ema and self.net_ema is not None:
            self.net_ema.to(dtype=torch.float32)

        if get_local_rank() == 0:
            logger.info("Training started (step=0)")

    def _compute_physical_losses(
        self,
        vt_pred_B_C_T_H_W: torch.Tensor,
        vt_B_C_T_H_W: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        tensor_kwargs: dict,
        x1_gt: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute angular-offset weighted velocity matching loss and global attenuation mass loss.
        """
        B, C, T, H, W = vt_pred_B_C_T_H_W.shape

        # 1. Angular-Offset Weighted Loss (w(theta_k) = 1 + gamma_side * sin^2(theta_k))
        gamma_side = getattr(self.hparams, "gamma_side", 1.0)
        angles = torch.linspace(0.0, 2.0 * np.pi, T, device=vt_pred_B_C_T_H_W.device, dtype=torch.float32)
        w_theta = (1.0 + gamma_side * (torch.sin(angles) ** 2)).view(1, 1, T, 1, 1)

        sq_err = (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2
        weighted_sq_err = sq_err * w_theta

        per_instance_loss = torch.mean(
            weighted_sq_err,
            dim=list(range(1, vt_pred_B_C_T_H_W.dim())),
        )
        time_weights_B = self.rectified_flow.train_time_weight(timesteps, tensor_kwargs)
        loss_angle_rf = torch.mean(time_weights_B.view(-1) * per_instance_loss)

        # 2. Line-Integral Global Attenuation Mass Loss (L_atten)
        # Always anchor D_pa to ground-truth PA frame from x1_gt (x1_gt[:, :, 0:1]) if available
        sigmas_5d = sigmas.view(B, 1, 1, 1, 1)
        x0_pred = xt_B_C_T_H_W - sigmas_5d * vt_pred_B_C_T_H_W
        D_pred = torch.mean(x0_pred, dim=[1, 3, 4])  # [B, T]
        if x1_gt is not None:
            D_pa = torch.mean(x1_gt[:, :, 0:1], dim=[1, 3, 4])  # [B, 1] GT PA attenuation mass
        else:
            D_pa = D_pred[:, 0:1]
        loss_atten = torch.mean((D_pred - D_pa) ** 2)

        loss_atten_weight = getattr(self.hparams, "loss_atten_weight", 0.02)
        total_loss = loss_angle_rf + loss_atten_weight * loss_atten

        return total_loss, loss_angle_rf, loss_atten, per_instance_loss

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Training step with rectified flow loss."""
        result = self._process_batch(batch, stage="train")
        video = result["video"]
        pre_cached_latent = result.get("pre_cached_latent")
        text_embeddings = result["text_embeddings"]
        text_filtered = text_embeddings

        if pre_cached_latent is not None:
            x_1 = pre_cached_latent
        elif video is not None:
            # Normalize video frames to [-1, 1]
            video_norm = video * 2.0 - 1.0
            # Encode video to latent space
            x_1 = self.encode(video_norm)
        else:
            raise ValueError("Training batch must contain 'video', 'ct', or 'pre_cached_latent'/'latent'.")

        B = x_1.shape[0]
        tensor_kwargs = {"device": x_1.device, "dtype": torch.float32}

        # Sample noise
        epsilon_B_C_T_H_W = torch.randn(x_1.size(), **tensor_kwargs)

        # Sample training time
        t_B = self.rectified_flow.sample_train_time(B).to(**tensor_kwargs)
        t_B = t_B.view(B, 1)

        # Get timesteps and sigmas
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tensor_kwargs)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tensor_kwargs)

        timesteps = timesteps.view(B, 1)
        sigmas = sigmas.view(B, 1)

        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x_1.float(), sigmas
        )

        condition = self._get_condition(
            text_filtered, x_1, None, True, apply_cfg_dropout=True
        )
        cond_mask = condition.condition_video_input_mask_B_C_T_H_W
        mean_cond_frames = cond_mask[:, 0, :, 0, 0].sum(dim=1).mean().item()

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W,
            timesteps_B_T=timesteps,
            condition=condition,
        )

        total_loss, loss_angle_rf, loss_atten, per_instance_loss = self._compute_physical_losses(
            vt_pred_B_C_T_H_W, vt_B_C_T_H_W, xt_B_C_T_H_W, sigmas, timesteps, tensor_kwargs, x1_gt=x_1
        )

        # Log training metrics
        self.log(
            "train/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/loss_angle_rf",
            loss_angle_rf,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/loss_atten",
            loss_atten,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/mse",
            per_instance_loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/timestep_mean",
            timesteps.float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/num_video_samples",
            float(B),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "train/mean_cond_frames",
            mean_cond_frames,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )

        # Move outputs to CPU to save GPU memory
        output_batch = {
            "x_1": x_1.detach().cpu(),
            "v_pred": vt_pred_B_C_T_H_W.detach().cpu(),
            "sigma": sigmas.detach().cpu(),
            "loss": total_loss.detach().cpu(),
        }

        return {"loss": total_loss, "output_batch": output_batch}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Validation step using EMA model."""
        result = self._process_batch(batch, stage="val")
        video = result["video"]
        pre_cached_latent = result.get("pre_cached_latent")
        text_embeddings = result["text_embeddings"]
        text_filtered = text_embeddings

        if pre_cached_latent is not None:
            x_1 = pre_cached_latent
        elif video is not None:
            # Normalize video frames to [-1, 1]
            video_norm = video * 2.0 - 1.0
            # Encode video to latent space
            x_1 = self.encode(video_norm)
        else:
            raise ValueError("Validation batch must contain 'video', 'ct', or 'pre_cached_latent'/'latent'.")

        B = x_1.shape[0]
        tensor_kwargs = {"device": x_1.device, "dtype": torch.float32}

        # Sample noise
        epsilon_B_C_T_H_W = torch.randn(x_1.size(), **tensor_kwargs)

        # Sample training time
        t_B = self.rectified_flow.sample_train_time(B).to(**tensor_kwargs)
        t_B = t_B.view(B, 1)

        # Get timesteps and sigmas
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, tensor_kwargs)
        sigmas = self.rectified_flow.get_sigmas(timesteps, tensor_kwargs)

        timesteps = timesteps.view(B, 1)
        sigmas = sigmas.view(B, 1)

        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x_1.float(), sigmas
        )

        condition = self._get_condition(text_filtered, x_1, 1, True)

        with self.ema_scope():
            vt_pred_B_C_T_H_W = self.denoise(
                noise=epsilon_B_C_T_H_W,
                xt_B_C_T_H_W=xt_B_C_T_H_W,
                timesteps_B_T=timesteps,
                condition=condition,
            )

        total_loss, loss_angle_rf, loss_atten, per_instance_loss = self._compute_physical_losses(
            vt_pred_B_C_T_H_W, vt_B_C_T_H_W, xt_B_C_T_H_W, sigmas, timesteps, tensor_kwargs, x1_gt=x_1
        )

        # Log validation metrics (EMA-based)
        self.log(
            "val/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=False,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/loss_angle_rf",
            loss_angle_rf,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/loss_atten",
            loss_atten,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/mse",
            per_instance_loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        self.log(
            "val/timestep_mean",
            timesteps.float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )

        # Move outputs to CPU to save GPU memory
        output_batch = {
            "x_1": x_1.detach().cpu(),
            "v_pred": vt_pred_B_C_T_H_W.detach().cpu(),
            "sigma": sigmas.detach().cpu(),
            "loss": total_loss.detach().cpu(),
        }

        return {"loss": total_loss, "output_batch": output_batch}

    @torch.inference_mode()
    def generate(
        self,
        image: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        shift: float = None,
        num_conditional_frames: int = 1,
        verbose: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate video using CFG.
        conditioning_input: [B,C,H,W] image or [B,C,T,H,W] video in [0,1].
        Returns: [B,C,T,H,W] in [0,1].
        """
        if image is None:
            raise ValueError("conditioning_input cannot be None for generation")
        if text_embeddings is None:
            raise ValueError("text_embeddings cannot be None for generation")
        if num_conditional_frames not in [1, 2]:
            raise ValueError(
                f"num_conditional_frames must be 1 or 2, got {num_conditional_frames}"
            )

        num_steps = (
            num_steps if num_steps is not None else self.hparams.num_inference_steps
        )
        guidance_scale = (
            guidance_scale
            if guidance_scale is not None
            else self.hparams.guidance_scale
        )
        shift = shift if shift is not None else self.hparams.rf_shift
        seed = seed if seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()

        with self.ema_scope_generation():
            model = self.net
            was_training = model.training
            model.eval()
            gen_dtype = next(model.parameters()).dtype

            if verbose:
                logger.info(
                    f"[generate] dtype={gen_dtype}, seed={seed}, steps={num_steps}, cfg={guidance_scale}"
                )

            # Handle image vs video input
            if image.dim() == 4:
                B, C_in, H_in, W_in = image.shape
                device = image.device
                input_norm = image.to(dtype=gen_dtype) * 2.0 - 1.0
                input_video = input_norm.unsqueeze(2)
                repeat_source = input_video[:, :, -1:, :, :]
                repeat_pad = repeat_source.repeat(1, 1, NUM_FRAMES - 1, 1, 1)
                input_video = torch.cat([input_video, repeat_pad], dim=2)
            else:
                B = image.shape[0]
                device = image.device
                input_video = image.to(dtype=gen_dtype) * 2.0 - 1.0

                pixels_needed = 4 * (num_conditional_frames - 1) + 1
                if input_video.shape[2] < pixels_needed:
                    raise ValueError(
                        f"Video needs {pixels_needed} frames, got {input_video.shape[2]}"
                    )

                if input_video.shape[2] < NUM_FRAMES:
                    last_frame = input_video[:, :, -1:, :, :]
                    padding = last_frame.repeat(
                        1, 1, NUM_FRAMES - input_video.shape[2], 1, 1
                    )
                    input_video = torch.cat([input_video, padding], dim=2)

            latent_cond = self.encode(input_video.to(device)).to(dtype=gen_dtype)
            _, C, T, H, W = latent_cond.shape
            state_shape = (C, T, H, W)

            if verbose:
                logger.info(
                    f"[generate] Latent: {latent_cond.shape}, cond_frames={num_conditional_frames}"
                )

            # Build CFG conditions (both keep video conditioning, only text differs)
            condition = self._get_condition(
                text_embeddings,
                latent_cond,
                num_conditional_frames,
                True,
                dtype=gen_dtype,
            )
            uncondition = self._get_condition(
                torch.zeros_like(text_embeddings),
                latent_cond,
                num_conditional_frames,
                True,
                dtype=gen_dtype,
            )

            noise = arch_invariant_rand(
                shape=(B,) + state_shape,
                dtype=torch.float32,
                device=device,
                seed=seed,
            ).to(dtype=gen_dtype)

            seed_generator = torch.Generator(device=device)
            seed_generator.manual_seed(seed)

            self.sample_scheduler.set_timesteps(
                num_inference_steps=num_steps,
                device=device,
                shift=shift,
                use_kerras_sigma=False,
            )
            timesteps = self.sample_scheduler.timesteps

            initial_noise = noise
            latents = noise.clone()

            for i, t in enumerate(timesteps):
                timestep = t.view(1, 1).expand(B, 1)

                v_cond = self.denoise(initial_noise, latents, timestep, condition)
                v_uncond = self.denoise(initial_noise, latents, timestep, uncondition)
                velocity_pred = v_uncond + guidance_scale * (v_cond - v_uncond)

                if verbose:
                    logger.info(
                        f"  Step {i}/{num_steps}: t={t.item():.0f}, |v|={velocity_pred.norm().item():.2f}"
                    )

                latents = self.sample_scheduler.step(
                    model_output=velocity_pred,
                    timestep=t,
                    sample=latents,
                    return_dict=False,
                    generator=seed_generator,
                )[0]

                # Lock conditioning frames to GT
                cond_mask = condition.condition_video_input_mask_B_C_T_H_W
                if cond_mask is not None:
                    latents = latents * (1 - cond_mask) + latent_cond * cond_mask

            denoised_latent = latents.detach().clone() if return_intermediates else None

            video = self.decode(latents.float())
            video = video / 2.0 + 0.5
            video = video.clamp(0, 1)

            if was_training:
                model.train()

            if return_intermediates:
                return {
                    "video": video,
                    "noise": initial_noise.detach().cpu(),
                    "denoised_latent": denoised_latent.detach().cpu(),
                }
            
            return video
        