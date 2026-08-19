"""Unit tests for predict3 physical loss functions."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from predict3.losses import (
    angular_offset_weights,
    angular_weighted_velocity_loss,
    attenuation_mass_loss,
    total_physical_loss,
)


class TestAngularOffsetWeights(unittest.TestCase):
    """Tests per-frame angular weight calculations."""

    def test_peaks_near_90_and_270_degrees(self) -> None:
        num_frames = 24
        w = angular_offset_weights(num_frames, gamma_side=1.0)

        quarter = num_frames // 4
        three_quarter = 3 * num_frames // 4

        self.assertGreater(w[quarter].item(), w[0].item())
        self.assertGreater(w[three_quarter].item(), w[0].item())

        angle_at_quarter = quarter * 2.0 * math.pi / (num_frames - 1)
        expected = 1.0 + math.sin(angle_at_quarter) ** 2

        self.assertAlmostEqual(w[quarter].item(), expected, places=4)
        self.assertLessEqual(w.max().item(), 2.0 + 1e-4)

    def test_zero_offset_at_0_degrees(self) -> None:
        num_frames = 24
        w = angular_offset_weights(num_frames, gamma_side=1.0)

        self.assertAlmostEqual(w[0].item(), 1.0, places=4)
        self.assertAlmostEqual(w[-1].item(), 1.0, places=4)

    def test_gamma_side_zero_is_uniform(self) -> None:
        w = angular_offset_weights(24, gamma_side=0.0)
        self.assertTrue(torch.allclose(w, torch.ones(24)))


class TestAngularWeightedVelocityLoss(unittest.TestCase):
    """Tests angular-weighted velocity matching loss."""

    def test_zero_when_prediction_matches_target(self) -> None:
        v = torch.randn(2, 16, 24, 4, 4)
        loss, per_instance = angular_weighted_velocity_loss(v, v.clone())

        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        self.assertTrue(torch.allclose(per_instance, torch.zeros(2), atol=1e-5))

    def test_positive_when_prediction_differs(self) -> None:
        v_pred = torch.zeros(2, 16, 24, 4, 4)
        v_target = torch.ones(2, 16, 24, 4, 4)
        loss, _ = angular_weighted_velocity_loss(v_pred, v_target)

        self.assertGreater(loss.item(), 0.0)

    def test_gradient_flows(self) -> None:
        v_pred = torch.randn(1, 4, 8, 2, 2, requires_grad=True)
        v_target = torch.randn(1, 4, 8, 2, 2)

        loss, _ = angular_weighted_velocity_loss(v_pred, v_target)
        loss.backward()

        self.assertIsNotNone(v_pred.grad)
        self.assertTrue(torch.isfinite(v_pred.grad).all())

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            angular_weighted_velocity_loss(torch.randn(1, 4, 8, 2, 2), torch.randn(1, 4, 4, 2, 2))


class TestAttenuationMassLoss(unittest.TestCase):
    """Tests global attenuation mass conservation loss."""

    def test_zero_for_constant_mass_orbit(self) -> None:
        batch, channels, frames, h, w = 2, 16, 24, 4, 4
        sigmas = torch.rand(batch)
        v_pred = torch.randn(batch, channels, frames, h, w)

        x_t = v_pred * sigmas.view(batch, 1, 1, 1, 1)
        x1_gt = torch.zeros(batch, channels, frames, h, w)

        loss = attenuation_mass_loss(v_pred, x_t, sigmas, x1_gt=x1_gt)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_falls_back_to_predicted_anchor_without_gt(self) -> None:
        batch, channels, frames, h, w = 1, 4, 8, 2, 2
        sigmas = torch.zeros(batch)

        x_t = torch.full((batch, channels, frames, h, w), 3.0)
        v_pred = torch.zeros_like(x_t)

        loss = attenuation_mass_loss(v_pred, x_t, sigmas, x1_gt=None)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestTotalPhysicalLoss(unittest.TestCase):
    """Tests combined physical loss function."""

    def test_combines_both_terms(self) -> None:
        batch, channels, frames, h, w = 2, 16, 24, 4, 4
        v_pred = torch.randn(batch, channels, frames, h, w, requires_grad=True)
        v_target = torch.randn(batch, channels, frames, h, w)
        x_t = torch.randn(batch, channels, frames, h, w)
        sigmas = torch.rand(batch)
        x1_gt = torch.randn(batch, channels, frames, h, w)

        result = total_physical_loss(
            v_pred, v_target, x_t, sigmas, x1_gt=x1_gt, gamma_side=1.0, loss_atten_weight=0.02
        )

        self.assertSetEqual(set(result.keys()), {"total", "angle_rf", "atten", "per_instance"})

        expected_total = result["angle_rf"] + 0.02 * result["atten"]
        self.assertAlmostEqual(result["total"].item(), expected_total.item(), places=5)

        result["total"].backward()
        self.assertIsNotNone(v_pred.grad)

    def test_atten_weight_zero_matches_plain_velocity_loss(self) -> None:
        v_pred = torch.randn(1, 4, 8, 2, 2)
        v_target = torch.randn(1, 4, 8, 2, 2)
        x_t = torch.randn(1, 4, 8, 2, 2)
        sigmas = torch.rand(1)

        result = total_physical_loss(v_pred, v_target, x_t, sigmas, gamma_side=0.0, loss_atten_weight=0.0)
        plain_mse = torch.mean((v_pred - v_target) ** 2)

        self.assertAlmostEqual(result["total"].item(), plain_mse.item(), places=5)


if __name__ == "__main__":
    unittest.main()

