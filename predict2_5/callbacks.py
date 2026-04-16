"""Callbacks for Cosmos-Predict 2.5 post-training."""

import copy
from typing import Optional

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

import torchvision

import lightning as L
from lightning.pytorch.callbacks import Callback

from predict2_5.utils import get_local_rank, get_logger


# --- Logger --- #
logger = get_logger(__name__)


class TensorBoardCallback(Callback):
    """
    Log training and validation results to TensorBoard, including ground truth and predictions.
    """
    
    def __init__(
        self, 
        num_frames_to_log: int = 8, 
        log_every_n_epochs: int = 1, 
        generate_every_n_epochs: int = 5
    ):
        """
        Initialize TensorBoard logging callback.
        
        Args
        -----
            num_frames_to_log: Number of video frames to log in TensorBoard.
            log_every_n_epochs: Log samples every N epochs.
            generate_every_n_epochs: Generate predictions every N epochs.
        """
        super().__init__()

        self.num_frames_to_log = num_frames_to_log
        self.log_every_n_epochs = log_every_n_epochs
        self.generate_every_n_epochs = generate_every_n_epochs 

        # Placeholders for train and validation batch data and outputs to be logged at epoch end
        self.train_batch_data = None
        self.train_outputs = None
        self.val_batch_data = None
        self.val_outputs = None

    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: dict, 
        batch: dict, 
        batch_idx: int
    ):
        """
        Store first training batch data and outputs for logging.

        Args
        -----
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            outputs: Model outputs from the training step.
            batch: Batch data dictionary to store.
            batch_idx: Index of the current batch in the epoch.
        """
        if batch_idx == 0:
            self._store_batch_data('train', batch, outputs)

    def on_validation_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: dict, 
        batch: dict, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ):
        """
        Store first validation batch data and outputs for logging.

        Args
        -----
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            outputs: Model outputs from the validation step.
            batch: Batch data dictionary to store.
            batch_idx: Index of the current batch in the epoch.
            dataloader_idx: Index of the validation dataloader (when multiple loaders).
                
        Notes
        -----
            Lightning processes validation dataloaders sequentially, so this identifies which dataloader (0, 1, 2, ...) the current batch originated from.
            This is mandatory for validation/test but absent in training because Lightning merges multiple training dataloaders into the batch itself.
        """
        if batch_idx == 0 and dataloader_idx == 0:
            self._store_batch_data('val', batch, outputs)

    def _store_batch_data(
        self, 
        stage: str, 
        batch: dict, 
        outputs: Optional[dict] = None
    ):
        """
        Store a copy of the first batch data and outputs for logging at epoch end.
        
        Args
        -----
            stage: "train" or "val" to indicate which phase the batch belongs to.
            batch: Batch data dictionary to store.
            outputs: Batch outputs to store.
        """
        try:
            if stage == "val" and self.val_batch_data is not None:
                # Already stored validation batch, skip to avoid overwriting
                return
            
            if batch.get("video") is None:
                # No video data in batch, skip storing
                return
            
            # Store a copy of the batch on CPU for logging
            cpu_batch_data = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else copy.deepcopy(v) 
                for k, v in batch.items()
            }
            
            # Store a copy of the outputs on CPU for logging
            cpu_outputs = None

            if outputs is not None and isinstance(outputs, dict):
                cpu_outputs = {}

                # Move all tensor components to CPU, keep non-tensor data as is
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        cpu_outputs[k] = v.detach().cpu()
                    elif isinstance(v, dict):
                        cpu_outputs[k] = {
                            kk: vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv
                            for kk, vv in v.items()
                        }
                    else:
                        cpu_outputs[k] = v
            
            if stage == 'train':
                self.train_batch_data = cpu_batch_data
                self.train_outputs = cpu_outputs
            else:
                if self.val_batch_data is None:
                    self.val_batch_data = cpu_batch_data
                    self.val_outputs = cpu_outputs
                
        except Exception as e:
            if get_local_rank() == 0:
                logger.exception("Failed to store %s batch data: %s", stage, e)

    def on_train_epoch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ):
        """
        Log training samples at epoch end.

        Args
        -----
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
        """
        if trainer.current_epoch % self.log_every_n_epochs == 0 and trainer.is_global_zero:
            # Only log on rank 0 to avoid duplicate logs in DDP strategy
            self._log_samples(
                trainer=trainer, 
                pl_module=pl_module, 
                data=self.train_batch_data, 
                outputs=self.train_outputs, 
                stage="train"
            )
        
        # Clean the training batch data and outputs for next epoch
        self.train_batch_data = None
        self.train_outputs = None

    def on_validation_epoch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ):
        """
        Log validation samples at epoch end.

        Args
        -----
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
        """
        if trainer.current_epoch % self.log_every_n_epochs == 0 and trainer.is_global_zero:
            # Only log on rank 0 to avoid duplicate logs in DDP strategy
            self._log_samples(
                trainer=trainer, 
                pl_module=pl_module, 
                data=self.val_batch_data, 
                outputs=self.val_outputs, 
                stage="val"
            )

        # Clean the validation batch data and outputs for next epoch
        self.val_batch_data = None
        self.val_outputs = None

    def _log_samples(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        data: Optional[dict], 
        outputs: Optional[dict],
        stage: str = "train"
    ):
        """
        Log all samples in the batch to TensorBoard, including ground truth and predictions.

        Args
        -----
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            data: Batch data dictionary to log.
            outputs: Batch outputs to log.
            stage: "train" or "val".
        """
        if data is None:
            # No batch data stored, skip logging
            return
        
        if not trainer.is_global_zero:
            # Only log on rank 0 to avoid duplicate logs in DDP strategy
            return
            
        if not hasattr(trainer, "logger") or not hasattr(trainer.logger, "experiment"):
            # No logger available, skip logging
            return
        
        # Get TensorBoard Summary Writer from the trainer's logger
        writer = trainer.logger.experiment

        # Get current epoch for logging step
        epoch = trainer.current_epoch
        
        has_video = data.get('video') is not None
        if not has_video:
            # No video data to log, skip logging for this batch
            return

        try:
            # Get the ground truth video
            # Multi-view X-Ray images are stored as a single video
            gt_video = data['video'][0].float()  # (C, T, H, W)

            # Permute to (T, C, H, W) to become a list of frames for later processing
            gt_frames = gt_video.permute(1, 0, 2, 3)  # (T, C, H, W)
            
            # Select frames to log
            num_frames = gt_frames.shape[0]
            k = min(self.num_frames_to_log, num_frames) 
            indices = torch.linspace(0, num_frames - 1, k).long() 
            gt_selected = gt_frames[indices]
            
            # Convert to grayscale if RGB
            if gt_selected.shape[1] == 3:
                gt_gray = gt_selected.mean(dim=1, keepdim=True)
            else:
                gt_gray = gt_selected
            
            # Log GT images
            gt_grid = torchvision.utils.make_grid(gt_gray, nrow=4, normalize=False)
            writer.add_image(f"{stage}/GT", gt_grid, global_step=epoch)
            
            # Placeholder for generated frames
            gen_frames = None

            # Determine if we should run generation based on the current epoch and the specified frequency
            should_generate = (
                self.generate_every_n_epochs > 0 and 
                epoch % self.generate_every_n_epochs == 0
            )
            
            # CRITICAL: Generation MUST only run on rank-0 to avoid DDP collective mismatch.
            # Other ranks never enter generation code path - this is the safest approach.
            # Using inference_mode() + non_blocking transfers eliminates remaining sync risks.
            if trainer.is_global_zero and should_generate:
                gen_frames = self._generate_video(pl_module, data)
            
            if gen_frames is not None:
                gen_selected = gen_frames[indices]
                
                # Convert to grayscale if RGB
                if gen_selected.shape[1] == 3:
                    gen_gray = gen_selected.mean(dim=1, keepdim=True)
                else:
                    gen_gray = gen_selected
                
                # Compute difference map
                diff_map = torch.abs(gt_gray - gen_gray)
                
                # Make grids
                gen_grid = torchvision.utils.make_grid(gen_gray, nrow=4, normalize=False)
                diff_grid = torchvision.utils.make_grid(diff_map, nrow=4, normalize=False)
                
                # Log predictions and difference
                writer.add_image(f"{stage}/Prediction", gen_grid, global_step=epoch)
                writer.add_image(f"{stage}/Difference_Map", diff_grid, global_step=epoch)
                
        except Exception as e:
            if get_local_rank() == 0:
                logger.warning("Error logging %s samples: %s", stage, e)
    
    def _generate_video(
        self, 
        pl_module: L.LightningModule, 
        data: dict
    ) -> Optional[torch.Tensor]:
        """
        Generate video using EMA model.

        Args
        -----
            pl_module: Lightning Module being trained, which contains the EMA model for generation.
            data: Batch data dictionary containing necessary inputs for generation (e.g., video, text embeddings).
        """
        try:
            # Get the current device of the Lightning module
            device = pl_module.device
            
            # Clear CUDA cache before generation to avoid cuSOLVER errors
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Set preferred linalg library to avoid cuSOLVER initialization issues
            # This helps when GPU memory is fragmented during training
            try:
                torch.backends.cuda.preferred_linalg_library("default")
            except Exception:
                pass  # Older PyTorch versions may not support this
            
            # Get ground truth video from data dict, move to GPU if available
            gt_video = data['video'].to(device, non_blocking=True)

            # Get text embeddings from data dict, move to GPU if available
            text_embeddings = data.get('text_embeddings')
            
            if text_embeddings is None:
                return None
            
            text_embeddings = text_embeddings.to(device, non_blocking=True)
            
            with torch.inference_mode():
                # Get first frame for conditioning (Image2World mode)
                first_frame = gt_video[:, :, 0, :, :]  # (B, C, H, W)
                
                # Generate using EMA model with deterministic seed
                pl_module.eval()
                generated_video = pl_module.generate(
                    image=first_frame,
                    text_embeddings=text_embeddings,
                    seed=42,  # Fixed seed for reproducibility
                    guidance_scale=1.0,
                    num_conditional_frames=1,  # 1 latent frame from first pixel frame
                    verbose=False,
                )  # (B, C, T, H, W)
            
            # Return first sample, permuted to (T, C, H, W)
            gen_sample = generated_video[0].detach().cpu().float()
            
            # Explicitly free GPU memory
            del generated_video, gt_video, text_embeddings, first_frame
            torch.cuda.empty_cache()
            
            return gen_sample.permute(1, 0, 2, 3)
            
        except Exception as e:
            if get_local_rank() == 0:
                logger.warning("Could not generate video: %s", e)

            # Clean up on error
            torch.cuda.empty_cache()

            return None


class EMAMonitorCallback(Callback):
    """
    Monitor EMA (Exponential Moving Average) model health during training.
    Tracks EMA parameter updates, divergence from the main model, and logs warnings if EMA appears frozen.
    """
    
    def __init__(self, log_every_n_steps: int = 500, warmup_steps: int = 100):
        """
        Initialize EMA monitor callback.
        
        Args:
            log_every_n_steps: Log EMA statistics every N training steps.
            warmup_steps: Skip monitoring for first N steps during warmup.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.last_ema_params = None
        self.zero_diff_count = 0
        self.dtype_checked = False

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pass

    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: dict, 
        batch: dict, 
        batch_idx: int
    ):
        """
        Monitor EMA status at the end of each training batch.

        Args:
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            outputs: Model outputs from the training step.
            batch: Batch data dictionary.
            batch_idx: Index of the current batch in the epoch.
        """
        if not hasattr(pl_module, 'net_ema') or pl_module.net_ema is None:
            return
        
        # Validate EMA dtype once (must be float32)
        if trainer.global_step == 1 and not self.dtype_checked:
            # Check EMA parameter dtype to ensure it's float32 for stable accumulation
            ema_dtype = next(pl_module.net_ema.parameters()).dtype

            if ema_dtype != torch.float32:
                raise RuntimeError(f"EMA dtype is {ema_dtype}, MUST be float32 for stable accumulation")
            
            self.dtype_checked = True
        
        if trainer.global_step < self.warmup_steps:
            return

        if trainer.global_step % self.log_every_n_steps != 0:
            return
            
        try:
            # Log EMA beta
            if hasattr(pl_module, '_get_ema_beta'):
                shift = pl_module.hparams.ema_iteration_shift
                effective_step = trainer.global_step - shift + 1
                if effective_step >= 1:
                    beta = pl_module._get_ema_beta(effective_step)
                    pl_module.log("ema/beta", beta, prog_bar=False)
            
            # Check EMA param changes
            current_params = next(pl_module.net_ema.parameters()).detach().cpu().clone()
            
            if self.last_ema_params is not None:
                diff = (current_params - self.last_ema_params).abs().mean().item()
                pl_module.log("ema/param_diff", diff, prog_bar=False)
                
                # Frozen EMA detection - log warning only
                if diff == 0.0:
                    self.zero_diff_count += 1
                    if self.zero_diff_count >= 5:
                        pl_module.log("ema/frozen_warning", 1.0, prog_bar=False)
                else:
                    self.zero_diff_count = 0
            
            # Model vs EMA divergence
            model_params = next(pl_module.net.parameters()).detach().cpu().float()
            divergence = (model_params - current_params.float()).abs().mean().item()
            pl_module.log("ema/model_divergence", divergence, prog_bar=False)
            
            self.last_ema_params = current_params
            
        except Exception:
            pass  # Silently skip monitoring errors


class GradClipCallback(Callback):
    """
    Clip gradients to prevent exploding gradients and NaN/Inf issues during training.
    Clips gradients to a specified norm and optionally replaces NaN/Inf values with zeros before clipping
    """
    
    def __init__(self, clip_norm: float = 1.0, force_finite: bool = True, log_every_n_steps: int = 50):
        """
        Initialize gradient clipping callback.
        
        Args:
            clip_norm: Maximum gradient norm allowed (L2 norm).
            force_finite: Replace NaN/Inf with zeros before clipping.
            log_every_n_steps: Log gradient norm statistics every N steps.
        """
        super().__init__()

        self.clip_norm = clip_norm
        self.force_finite = force_finite
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        optimizer
    ):
        """
        Clip gradients before optimizer step.

        Args:
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            optimizer: Optimizer being used for training.
        """
        # Sanitize NaN/Inf gradients
        if self.force_finite:
            for param in pl_module.net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
        
        # Clip gradients
        total_norm = clip_grad_norm_(
            pl_module.net.parameters(),
            max_norm=self.clip_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )
        
        # Log periodically
        if trainer.global_step % self.log_every_n_steps == 0:
            pl_module.log("grad/norm", total_norm, prog_bar=False)


class DDPSamplerEpochCallback(Callback):
    """
    Update DistributedSampler epoch at the start of each training epoch.
    Ensures proper shuffling of data across epochs in DDP training.
    """
    
    def on_train_epoch_start(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ):
        """
        Update sampler epoch for proper shuffling.

        Args:
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
        """
        # Get the training dataloader(s) from the trainer
        loader = trainer.train_dataloader
        
        if loader is None:
            return
        
        # Handle different dataloader structures
        if hasattr(loader, 'loaders'):
            # Multiple dataloaders
            for _loader in loader.loaders.values():
                self._set_sampler_epoch(_loader, trainer.current_epoch, trainer.global_rank)
        else:
            self._set_sampler_epoch(loader, trainer.current_epoch, trainer.global_rank)
    
    def _set_sampler_epoch(
        self, 
        loader, 
        epoch: int, 
        rank: int
    ):
        """
        Set epoch on sampler if available.

        Args:
            loader: DataLoader to check for sampler.
            epoch: Current epoch number to set.
            rank: Global rank of the current process for logging.
        """
        # Check for sampler (DistributedSampler)
        if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

            if rank == 0:
                logger.debug("Updated DistributedSampler epoch to %s", epoch)
        
        # Check for batch_sampler (MixedBatchSampler)
        elif hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, 'set_epoch'):
            # Set epoch for batch_sampler if it has set_epoch method (e.g., MixedBatchSampler)
            loader.batch_sampler.set_epoch(epoch)

            if rank == 0:
                logger.debug("Updated BatchSampler epoch to %s", epoch)


class GradientMonitorCallback(Callback):
    """
    Monitor gradient norms including cross-attention module health.
    Tracks overall model gradients and specifically monitors cross-attention
    gradients to detect training issues early.
    """
    
    def __init__(self, log_every_n_steps: int = 50):
        """
        Initialize gradient monitor callback.
        
        Args:
            log_every_n_steps: Log gradient statistics every N steps.
        """
        super().__init__()

        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        optimizer: torch.optim.Optimizer
    ):
        """
        Log gradient statistics

        Args:
            trainer: Lightning Trainer object.
            pl_module: Lightning Module being trained.
            optimizer: Optimizer being used for training.
        """
        if trainer.global_step % self.log_every_n_steps != 0:
            return
            
        try:
            total_norm_sq = 0.0
            crossattn_norm_sq = 0.0
            
            for name, p in pl_module.net.named_parameters():
                if p.grad is not None:
                    grad_norm_sq = p.grad.data.norm(2).item() ** 2
                    total_norm_sq += grad_norm_sq
                    if "crossattn" in name.lower() or "cross_attn" in name.lower():
                        crossattn_norm_sq += grad_norm_sq
            
            pl_module.log("grad/norm", total_norm_sq ** 0.5, prog_bar=False)
            pl_module.log("grad/crossattn_norm", crossattn_norm_sq ** 0.5, prog_bar=False)
            
        except Exception:
            pass


