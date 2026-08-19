"""Physical loss regularizers for 360-degree DRR rotation synthesis."""

from __future__ import annotations

import torch


def angular_offset_weights(
    num_frames: int,
    gamma_side: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes per-frame weights that emphasize lateral and oblique views.

    Weight formula: w(theta) = 1 + gamma_side * sin^2(theta), peaking at 90/270 degrees
    where view ambiguity is highest.

    Args:
        num_frames: Number of rotation frames.
        gamma_side: Weighting intensity factor (0 disables weighting).
        device: PyTorch device.
        dtype: PyTorch data type.

    Returns:
        Tensor of shape [num_frames] with per-frame weights.
    """
    angles = torch.linspace(0.0, 2.0 * torch.pi, num_frames, device=device, dtype=dtype)

    return 1.0 + gamma_side * torch.sin(angles) ** 2


def angular_weighted_velocity_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    gamma_side: float = 1.0,
    time_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes angular-offset weighted velocity matching loss (L_angle_rf).

    Args:
        v_pred: Predicted velocity tensor [B, C, T, H, W].
        v_target: Ground-truth velocity tensor [B, C, T, H, W].
        gamma_side: Angular weight strength.
        time_weights: Optional per-sample timestep weights [B].

    Returns:
        Tuple of (scalar_loss, per_instance_loss).
    """
    if v_pred.shape != v_target.shape:
        raise ValueError(f"v_pred shape {tuple(v_pred.shape)} != v_target shape {tuple(v_target.shape)}")

    batch, _, num_frames = v_pred.shape[:3]

    w_theta = angular_offset_weights(
        num_frames, gamma_side=gamma_side, device=v_pred.device, dtype=torch.float32
    ).view(1, 1, num_frames, 1, 1)

    sq_err = (v_pred - v_target) ** 2
    weighted_sq_err = sq_err * w_theta
    per_instance_loss = torch.mean(weighted_sq_err, dim=list(range(1, v_pred.dim())))

    if time_weights is None:
        time_weights = torch.ones(batch, device=v_pred.device, dtype=per_instance_loss.dtype)

    loss = torch.mean(time_weights.view(-1) * per_instance_loss)

    return loss, per_instance_loss


def attenuation_mass_loss(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    x1_gt: torch.Tensor | None = None,
    anchor_frame_index: int = 0,
) -> torch.Tensor:
    """Computes global attenuation mass conservation loss (L_atten).

    Penalizes deviations in total latent mass across rotation views relative to
    the 0-degree anchor view to enforce Beer-Lambert physical consistency.

    Args:
        v_pred: Predicted velocity [B, C, T, H, W].
        x_t: Noisy sample at current timestep [B, C, T, H, W].
        sigmas: Timestep sigmas [B] or [B, 1].
        x1_gt: Optional ground-truth clean latent [B, C, T, H, W].
        anchor_frame_index: Index of the 0-degree anchor frame.

    Returns:
        Scalar loss tensor.
    """
    if v_pred.shape != x_t.shape:
        raise ValueError(f"v_pred shape {tuple(v_pred.shape)} != x_t shape {tuple(x_t.shape)}")

    batch = v_pred.shape[0]

    sigmas_5d = sigmas.reshape(batch, 1, 1, 1, 1).to(v_pred.dtype)
    x0_pred = x_t - sigmas_5d * v_pred

    mass_per_view = torch.mean(x0_pred, dim=[1, 3, 4])  # [B, T]

    if x1_gt is not None:
        anchor_mass = torch.mean(
            x1_gt[:, :, anchor_frame_index : anchor_frame_index + 1], dim=[1, 3, 4]
        )  # [B, 1]
    else:
        anchor_mass = mass_per_view[:, anchor_frame_index : anchor_frame_index + 1]

    return torch.mean((mass_per_view - anchor_mass) ** 2)


def total_physical_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    x1_gt: torch.Tensor | None = None,
    gamma_side: float = 1.0,
    loss_atten_weight: float = 0.02,
    time_weights: torch.Tensor | None = None,
    anchor_frame_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Combines angular-weighted velocity loss and attenuation mass loss.

    Returns:
        Dictionary containing 'total', 'angle_rf', 'atten', and 'per_instance' losses.
    """
    loss_angle_rf, per_instance_loss = angular_weighted_velocity_loss(
        v_pred, v_target, gamma_side=gamma_side, time_weights=time_weights
    )

    loss_atten = attenuation_mass_loss(
        v_pred, x_t, sigmas, x1_gt=x1_gt, anchor_frame_index=anchor_frame_index
    )

    total = loss_angle_rf + loss_atten_weight * loss_atten

    return {
        "total": total,
        "angle_rf": loss_angle_rf,
        "atten": loss_atten,
        "per_instance": per_instance_loss,
    }

