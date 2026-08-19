"""AR Reasoner / DM Generator tower classification for Cosmos 3.

Cosmos 3 uses a Mixture-of-Transformers (MoT) architecture with two distinct towers:
  - AR Reasoner tower ('understanding' / 'und'): Causal language and ViT vision layers.
  - DM Generator tower ('generation' / 'gen'): Bidirectional video/action diffusion layers.

In SFT training, the AR Reasoner is frozen and only the DM Generator parameters are updated.
This module classifies parameter names to selectively target generator weights.
"""

from __future__ import annotations

import re

# ==============================================================================
# Canonical AR/DM Split Configuration
# ==============================================================================

# Top-level generator modules: VAE latent projections and time embedder
GENERATOR_TOP_LEVEL: tuple[str, ...] = ("vae2llm", "llm2vae", "time_embedder")

# Top-level reasoner modules: text token embeddings and language head
REASONER_TOP_LEVEL: tuple[str, ...] = ("embed_tokens", "lm_head", "norm")

# Per-layer parameter markers for generator-side sub-modules
GENERATOR_LAYER_MARKERS: tuple[str, ...] = ("moe_gen", "k_norm_und_for_gen")

# Allowlist keys selected during generator-only optimizer training
KEYS_TO_SELECT: tuple[str, ...] = (*GENERATOR_TOP_LEVEL, *GENERATOR_LAYER_MARKERS)

# Native parameter names mapped to diffusers export format
NATIVE_TO_DIFFUSERS_ATTN_REMAP: dict[str, str] = {
    "q_proj_moe_gen": "add_q_proj",
    "k_proj_moe_gen": "add_k_proj",
    "v_proj_moe_gen": "add_v_proj",
    "o_proj_moe_gen": "to_add_out",
    "q_norm_moe_gen": "norm_added_q",
    "k_norm_moe_gen": "norm_added_k",
    "vae2llm": "proj_in",
    "llm2vae": "proj_out",
}


def is_generator_param(name: str) -> bool:
    """Checks whether a parameter name belongs to the DM Generator tower.

    Args:
        name: Dotted parameter name (e.g., 'layers.0.mlp_moe_gen.weight').

    Returns:
        True if the parameter belongs to the generator, False if reasoner.
    """
    top = name.split(".")[0]

    if top in GENERATOR_TOP_LEVEL:
        return True

    if top in REASONER_TOP_LEVEL:
        return False

    return any(marker in name for marker in GENERATOR_LAYER_MARKERS)


def freeze_reasoner_tower(model) -> dict[str, int]:
    """Freezes the AR Reasoner tower parameters in-place, keeping the DM Generator trainable.

    Args:
        model: PyTorch model instance.

    Returns:
        Dictionary with 'trainable' and 'frozen' parameter counts.
    """
    trainable = frozen = 0

    for name, param in model.named_parameters():
        if is_generator_param(name):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()

    if trainable == 0:
        raise RuntimeError(
            "freeze_reasoner_tower() left 0 trainable parameters. "
            "Please check parameter naming against cosmos_framework."
        )

    return {"trainable": trainable, "frozen": frozen}


def parse_toml_keys_to_select(toml_text: str) -> tuple[str, ...]:
    """Extracts 'keys_to_select' list from a recipe TOML configuration string.

    Args:
        toml_text: Raw TOML file content.

    Returns:
        Tuple of selected parameter key pattern strings.
    """
    match = re.search(r"keys_to_select\s*=\s*\[(.*?)\]", toml_text, re.DOTALL)
    if match is None:
        raise ValueError("No `keys_to_select = [...]` entry found in the given TOML text.")

    items = re.findall(r'"([^"]+)"', match.group(1))

    return tuple(items)
