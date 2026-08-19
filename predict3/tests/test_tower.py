"""Unit tests for predict3.tower AR/DM parameter classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predict3.tower import (
    KEYS_TO_SELECT,
    freeze_reasoner_tower,
    is_generator_param,
    parse_toml_keys_to_select,
)


class TestIsGeneratorParam(unittest.TestCase):
    """Tests generator vs. reasoner parameter classification."""

    def test_reasoner_top_level_and_attention_are_reasoner_side(self) -> None:
        for name in (
            "embed_tokens.weight",
            "lm_head.weight",
            "norm.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
        ):
            self.assertFalse(is_generator_param(name), name)

    def test_moe_gen_suffixed_attention_and_mlp_are_generator_side(self) -> None:
        for name in (
            "layers.0.self_attn.q_proj_moe_gen.weight",
            "layers.0.self_attn.k_proj_moe_gen.weight",
            "layers.0.self_attn.v_proj_moe_gen.weight",
            "layers.0.self_attn.o_proj_moe_gen.weight",
            "layers.0.self_attn.q_norm_moe_gen.weight",
            "layers.0.self_attn.k_norm_moe_gen.weight",
            "layers.0.mlp_moe_gen.gate_proj.weight",
            "layers.0.input_layernorm_moe_gen.weight",
            "layers.0.post_attention_layernorm_moe_gen.weight",
        ):
            self.assertTrue(is_generator_param(name), name)

    def test_k_norm_und_for_gen_is_generator_side(self) -> None:
        self.assertTrue(is_generator_param("layers.0.self_attn.k_norm_und_for_gen.weight"))

    def test_top_level_vfm_projections_are_generator_side(self) -> None:
        for name in ("vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "time_embedder.mlp.0.weight"):
            self.assertTrue(is_generator_param(name), name)


class TestFreezeReasonerTower(unittest.TestCase):
    """Tests freezing reasoner parameters on a toy model."""

    def _toy_model(self) -> nn.Module:
        model = nn.Module()
        model.embed_tokens = nn.Embedding(4, 4)
        model.mlp_moe_gen = nn.Linear(4, 4)
        model.vae2llm = nn.Linear(4, 4)
        return model

    def test_freezes_reasoner_trains_generator(self) -> None:
        model = self._toy_model()
        summary = freeze_reasoner_tower(model)

        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertTrue(model.mlp_moe_gen.weight.requires_grad)
        self.assertTrue(model.vae2llm.weight.requires_grad)

        self.assertGreater(summary["trainable"], 0)
        self.assertGreater(summary["frozen"], 0)

    def test_raises_if_naming_convention_matches_nothing(self) -> None:
        model = nn.Module()
        model.embed_tokens = nn.Embedding(4, 4)
        with self.assertRaises(RuntimeError):
            freeze_reasoner_tower(model)


class TestParseTomlKeysToSelect(unittest.TestCase):
    """Tests TOML keys_to_select extraction."""

    def test_parses_a_minimal_toml_snippet(self) -> None:
        text = '[optimizer]\nkeys_to_select = [\n    "moe_gen",\n    "time_embedder",\n]\n'
        self.assertEqual(parse_toml_keys_to_select(text), ("moe_gen", "time_embedder"))

    def test_raises_when_missing(self) -> None:
        with self.assertRaises(ValueError):
            parse_toml_keys_to_select("[optimizer]\nlr = 1.0e-4\n")


class TestRecipeStaysInSyncWithTower(unittest.TestCase):
    """Tests that xray360_edge.toml keys_to_select matches predict3.tower.KEYS_TO_SELECT."""

    def test_recipe_toml_matches_keys_to_select(self) -> None:
        toml_path = REPO_ROOT / "predict3" / "recipes" / "xray360_edge.toml"
        self.assertTrue(toml_path.exists(), f"Recipe TOML not found at {toml_path}")

        toml_text = toml_path.read_text()
        parsed_keys = parse_toml_keys_to_select(toml_text)

        self.assertEqual(set(parsed_keys), set(KEYS_TO_SELECT))



class TestImageConditioningIsWired(unittest.TestCase):
    """Regression guard for image-conditioned generation (native I2V)."""

    def test_launch_script_forces_pure_i2v_conditioning(self) -> None:
        script_path = REPO_ROOT / "scripts" / "launch_cosmos3_worker.sh"
        text = script_path.read_text()
        self.assertIn("conditioning_config={0:0.0,1:1.0,2:0.0}", text)

    def test_inferencer_predict_passes_the_frontal_image_to_the_pipeline(self) -> None:
        inferencer_path = REPO_ROOT / "predict3" / "inferencer.py"
        source = inferencer_path.read_text()
        predict_body = source[source.index("def predict("):]
        self.assertRegex(
            predict_body,
            r"self\.pipe\(\s*\n(?:.*\n)*?\s*image=pil_image,",
            f"{inferencer_path}: InferencerV3.predict() no longer passes image=pil_image to the pipeline",
        )



if __name__ == "__main__":
    unittest.main()

