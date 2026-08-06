"""Regression tests for baseline loss functions, gradient propagation, and physics mappings."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def test_naf_fit_density_field_uses_beer_lambert():
    """Verifies that NAF fit_density_field's perspective ray marching (models.utils.
    perspective_ray_march) explicitly invokes apply_beer_lambert_correction."""
    from models.naf import fit_density_field, DensityNetwork, FreqEncoder

    device = "cpu"
    bound = 0.5
    encoder = FreqEncoder(
        input_dim=3,
        max_freq_log2=9,
        N_freqs=10,
        log_sampling=True,
        include_input=True,
        periodic_fns=(torch.sin, torch.cos),
    )
    model = DensityNetwork(
        encoder=encoder,
        bound=bound,
        num_layers=4,
        hidden_dim=64,
        skips=[2],
        out_dim=1,
        last_activation="sigmoid",
    ).to(device)

    target_proj = torch.rand(1, 1, 64, 64)

    # perspective_ray_march (called once per fit_density_field iteration) lives in
    # models.utils and applies Beer-Lambert there, not inline in models.naf anymore.
    with patch("models.utils.apply_beer_lambert_correction", wraps=torch.exp) as mock_beer:
        loss = fit_density_field(model, target_proj, bound, device, iterations=2, lr=1e-3)
        assert mock_beer.called, "apply_beer_lambert_correction was not called in fit_density_field!"
        assert mock_beer.call_count == 2
        assert isinstance(loss, float)


def test_pixelnerf_coarse_mlp_receives_gradient():
    """Verifies that mlp_coarse parameters receive non-zero gradients when simple_output=False."""
    from models.pixelnerf import build_model_and_renderer, encode_source_view, render_view, PIXELNERF_MODEL_AVAILABLE

    if not PIXELNERF_MODEL_AVAILABLE:
        pytest.skip("PixelNeRF dependencies not available")

    render_res = 64
    lateral_azimuth_deg = 90.0
    device = "cpu"
    
    model, render_wrapper = build_model_and_renderer(device, simple_output=False)
    
    pa_tensor = torch.rand(1, 3, 64, 64)
    lat_tensor = torch.rand(1, 3, render_res, render_res)

    encode_source_view(model, pa_tensor, device)
    rgb_coarse, rgb_fine = render_view(
        render_wrapper, lateral_azimuth_deg, render_res, device, return_coarse=True
    )
    pred_coarse = rgb_coarse.permute(2, 0, 1).unsqueeze(0)
    pred_fine = rgb_fine.permute(2, 0, 1).unsqueeze(0)

    loss = F.mse_loss(pred_coarse, lat_tensor) + F.mse_loss(pred_fine, lat_tensor)
    loss.backward()

    # Verify coarse MLP received gradient
    coarse_has_grad = False
    for param in model.mlp_coarse.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            coarse_has_grad = True
            break

    assert coarse_has_grad, "mlp_coarse parameters received zero gradient!"


def test_pixelnerf_simple_output_warning_guard(caplog):
    """Verifies warning guard is triggered if return_coarse=True is called on simple_output=True renderer."""
    import logging
    from models.pixelnerf import build_model_and_renderer, encode_source_view, render_view, PIXELNERF_MODEL_AVAILABLE

    if not PIXELNERF_MODEL_AVAILABLE:
        pytest.skip("PixelNeRF dependencies not available")

    device = "cpu"
    model, render_wrapper_simple = build_model_and_renderer(device, simple_output=True)

    pa_tensor = torch.rand(1, 3, 64, 64)
    encode_source_view(model, pa_tensor, device)

    with caplog.at_level(logging.WARNING):
        _ = render_view(render_wrapper_simple, 0.0, 32, device, return_coarse=True)

    assert "render_view called with return_coarse=True but renderer was built with simple_output=True" in caplog.text


def test_mednerf_fit_latent_and_weights_normalizes_target_range():
    """Verifies fit_latent_and_weights normalizes input target in [0, 1] to [-1, 1] before computing loss."""
    from models.mednerf import fit_latent_and_weights, MEDNERF_AVAILABLE

    if not MEDNERF_AVAILABLE:
        pytest.skip("MedNeRF dependencies not available")

    # Mock generator that tracks rays and forward execution
    class DummyGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self._parameters = {"param": self.param}
            self._named_parameters = {"param": self.param}
            self.focal = 100.0

        def val_ray_sampler(self, h, w, focal, pose):
            return torch.zeros(h, w, 3), None

        def forward(self, z, rays=None):
            return self.param * torch.zeros(32 * 32, 3)

    gen = DummyGenerator()
    target_zero = torch.zeros(1, 1, 32, 32)  # In range [0, 1]

    captured_targets = []
    original_mse = F.mse_loss

    def mock_mse(input_t, target_t, **kwargs):
        captured_targets.append(target_t.detach().clone())
        return original_mse(input_t, target_t, **kwargs)

    with patch("models.mednerf.F.mse_loss", side_effect=mock_mse), \
         patch("models.mednerf.lpips.PerceptualLoss", return_value=lambda x, y: torch.tensor(0.0)):
        _z, _loss = fit_latent_and_weights(
            gen, target_zero, z_dim=1, img_size=32, radius=1.0, theta_mean=80.0, device="cpu", iterations=1
        )

    assert len(captured_targets) > 0, "MSE loss was not called during fit_latent_and_weights!"
    # Target was 0.0 in [0, 1]; after target * 2.0 - 1.0, it should be -1.0
    assert torch.allclose(captured_targets[0], torch.tensor(-1.0)), (
        f"Expected target normalized to -1.0, got min={captured_targets[0].min()}, max={captured_targets[0].max()}"
    )


def test_train_scripts_ast_check_undefined_names():
    """AST-parses all train/*.py files with scope tracking to statically verify no undefined names exist in function bodies."""
    import ast
    import builtins

    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "Exception"}
    train_dir = os.path.join(BASE_DIR, "train")
    train_scripts = ["dx2ct.py", "mednerf.py", "naf.py", "pixelnerf.py", "svdrr.py", "xraysyn.py"]

    class ScopeVisitor(ast.NodeVisitor):
        def __init__(self, global_names):
            self.scope_stack = [set(global_names)]
            self.undefined = []

        def visit_Import(self, node):
            for alias in node.names:
                self.scope_stack[-1].add(alias.asname or alias.name.split('.')[0])
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                self.scope_stack[-1].add(alias.asname or alias.name)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            self.scope_stack[-1].add(node.name)
            local_scope = set(self.scope_stack[-1])
            for arg in node.args.args + node.args.kwonlyargs:
                local_scope.add(arg.arg)
            if node.args.vararg:
                local_scope.add(node.args.vararg.arg)
            if node.args.kwarg:
                local_scope.add(node.args.kwarg.arg)
            
            for child in ast.walk(node):
                if isinstance(child, (ast.Assign, ast.AnnAssign, ast.For, ast.With, ast.ExceptHandler)):
                    for sub in ast.walk(child):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Param)):
                            local_scope.add(sub.id)
                elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    for gen in child.generators:
                        for sub in ast.walk(gen.target):
                            if isinstance(sub, ast.Name):
                                local_scope.add(sub.id)

            self.scope_stack.append(local_scope)
            for stmt in node.body:
                self.visit(stmt)
            self.scope_stack.pop()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                current_scope = self.scope_stack[-1]
                if node.id not in current_scope:
                    self.undefined.append((node.id, getattr(node, "lineno", 0)))
            self.generic_visit(node)

    for script_name in train_scripts:
        script_path = os.path.join(train_dir, script_name)
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=script_path)

        globals_set = set(builtin_names)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    globals_set.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    globals_set.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                globals_set.add(node.name)
            else:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        globals_set.add(sub.id)

        visitor = ScopeVisitor(globals_set)
        visitor.visit(tree)
        assert not visitor.undefined, f"Undefined names found in {script_name}: {visitor.undefined}"


