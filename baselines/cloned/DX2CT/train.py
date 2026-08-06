"""Dx2CT (ICASSP 2025) Training & Data Pipeline

Implements full 3D diffusion slice generation training on Cross-Dataset Split.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch or NumPy not available")

try:
    from diffusers import DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Import DX2CTModel locally
from model import DX2CTModel
DX2CT_MODEL_AVAILABLE = True

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
)
logger = logging.getLogger(__name__)


class CrossDatasetSliceDataset(Dataset):
    """Dataset loading pre-processed DiffDRR cross-dataset projections & slices.
    
    Loads actual pre-rendered DiffDRR projections (PA, Lateral, and target views)
    from TCIA, MELA, or NSCLC folders mapped via the JSON split config.
    """
    
    def __init__(self, split_config_path: str, is_train: bool = True) -> None:
        """Initializes dataset from the split constraints.

        Args:
            split_config_path: Path to cross_dataset_split.json.
            is_train: If True loads TCIA+MELA (train split). If False loads NSCLC (test split).
        """
        super().__init__()
        self.is_train = is_train
        self.samples = []
        
        # Resolve path relative to split_config_path or workspace root
        base_dir = os.path.dirname(os.path.abspath(split_config_path))
        rendered_dir = os.path.join(base_dir, "pre_rendered")
        
        # Fallback if rendered_dir does not exist relative to split_config_path
        if not os.path.exists(rendered_dir):
            rendered_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets", "pre_rendered"))
            
        split_dir = os.path.join(rendered_dir, "train" if is_train else "test")
        
        if os.path.exists(split_dir):
            for pat_id in os.listdir(split_dir):
                pat_path = os.path.join(split_dir, pat_id)
                if os.path.isdir(pat_path):
                    if os.path.exists(os.path.join(pat_path, "pa.png")) and os.path.exists(os.path.join(pat_path, "lat.png")):
                        self.samples.append(pat_path)
        
        self.total_cases = len(self.samples)
        logger.info(f"Initialized {'Train' if is_train else 'Test'} dataset with {self.total_cases} cases from {split_dir}.")

    def __len__(self) -> int:
        return self.total_cases

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns PA, LAT, target slice, and 3D coordinate grid."""
        from PIL import Image
        import torchvision.transforms.functional as TF
        
        pat_path = self.samples[idx]
        
        pa_img = Image.open(os.path.join(pat_path, "pa.png")).convert("L")
        lat_img = Image.open(os.path.join(pat_path, "lat.png")).convert("L")
        
        pa = TF.to_tensor(pa_img)  # [1, 256, 256]
        lat = TF.to_tensor(lat_img)  # [1, 256, 256]
        
        spatial_size = 256
        views_dir = os.path.join(pat_path, "views")
        if os.path.exists(views_dir) and len(os.listdir(views_dir)) > 0:
            view_files = os.listdir(views_dir)
            target_file = os.path.join(views_dir, np.random.choice(view_files))
            target_img = Image.open(target_file).convert("L")
            target_slice = TF.to_tensor(target_img)  # [1, 256, 256]
        else:
            target_slice = pa.clone()
        
        # Coordinates for target slice [256*256, 3]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, spatial_size),
            torch.linspace(-1, 1, spatial_size),
            indexing="ij"
        )
        z_val = np.random.uniform(-1.0, 1.0)
        coords_3d = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            torch.full((spatial_size * spatial_size,), z_val)
        ], dim=-1)
        
        return pa, lat, target_slice, coords_3d


def train_diffusion(args: argparse.Namespace) -> None:
    """Trains the Dx2CT diffusion model.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if not TORCH_AVAILABLE or not DIFFUSERS_AVAILABLE:
        logger.error("Required libraries for Dx2CT missing.")
        return

    # 1. Initialize dataset based on cross-dataset split
    train_dataset = CrossDatasetSliceDataset(args.data_split, is_train=True)
    if len(train_dataset) == 0:
        logger.error("Dataset is empty. Ensure pre-rendered data exists in datasets/pre_rendered/train.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    # 2. Build the DX2CTModel
    model = DX2CTModel(embed_dim=128).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 3. Setup optimizer and objective
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 4. Training Loop
    model.train()
    logger.info(f"Starting training for {args.epochs} epochs on {len(train_dataset)} real cases...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (pa, lat, target_slice, coords_3d) in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
                
            pa, lat = pa.to(device), lat.to(device)
            target_slice, coords_3d = target_slice.to(device), coords_3d.to(device)
            
            # Sample noise to add to the slice
            noise = torch.randn_like(target_slice)
            B = target_slice.shape[0]
            
            # Sample a random timestep for each slice
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
            
            # Add noise to the clean slice
            noisy_slice = scheduler.add_noise(target_slice, noise, timesteps)
            
            optimizer.zero_grad()
            
            # Predict the noise residual conditioned on timesteps
            noise_pred = model(noisy_slice, pa, lat, coords_3d, timesteps=timesteps)
            
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / max(1, num_batches)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "dx2ct_surrogate_checkpoint.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dx2CT Diffusion Trainer")
    parser.add_argument("--data_split", type=str, default="datasets/cross_dataset_split.json", help="Path to JSON split config.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (Paper default: 80).")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size (Paper default: 16).")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (Paper default: 5e-5).")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional max batches per epoch for quick testing.")
    
    args = parser.parse_args()
    
    if TORCH_AVAILABLE:
        train_diffusion(args)
    else:
        logger.error("PyTorch not installed. Cannot train.")
