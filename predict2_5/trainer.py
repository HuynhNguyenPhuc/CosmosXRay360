"""Trainer for NVSyn Cosmos-Predict 2.5 post-training."""

import warnings

warnings.filterwarnings("ignore")

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import argparse
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path for direct/torchrun execution (`torchrun ... predict2_5/trainer.py`
# puts this file's own directory on sys.path[0], not the repo root, so the `predict2_5` package
# itself wouldn't otherwise be importable).
_BASE_DIR = Path(__file__).resolve().parents[1]
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

from predict2_5.module import CosmosXRay360
from predict2_5.callbacks import (
    TensorBoardCallback,
    EMAMonitorCallback,
    GradientMonitorCallback,
    GradClipCallback,
)

from predict2_5.datamodule import PreRenderedDataModule
from predict2_5.utils import get_logger, get_local_rank


# --- Logger --- #
logger = get_logger(__name__)


class TrainingConfig:
    """
    Configuration for post-training Cosmos-Predict 2.5.
    """

    # Model
    model_size: str = "2B"
    sac_mode: str = "predict2_2b_720_aggressive"
    checkpoint_path: str = None
    tokenizer_path: str = None
    text_encoder_path: str = None

    # Training
    learning_rate: float = 2 ** (-14.5)
    weight_decay: float = 0.001
    warmup_steps: int = 2000
    max_iters: int = 100000
    loss_scale: float = 1.0
    gradient_clip_val: float = 1.0

    # EMA
    enable_ema: bool = True
    ema_rate: float = 0.10
    ema_offload_cpu: bool = False
    ema_iteration_shift: int = 0

    # Rectified Flow
    rf_shift: float = 5.0
    num_inference_steps: int = 35
    guidance_scale: float = 1.5

    # Physics & Geometric regularization
    gamma_side: float = 1.0
    loss_atten_weight: float = 0.02

    # Hardware
    precision: str = "bf16-mixed"
    num_gpus: int = 4
    batch_size: int = 1
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    strategy: str = "auto"
    ema_sync_every_n_steps: int = 10

    # Data
    dataset_path: str = "datasets/pre_rendered"
    use_latent_cache: bool = False
    latent_cache_dir: str = "datasets/pre_rendered_latents"
    cache_dir: str = "./cache"
    load_text_embeddings: bool = True
    enable_disk_cache: bool = True
    cache_strategy: str = "disk"
    build_cache_only: bool = False
    img_shape: int = 256

    # Output
    output_dir: str = "outputs"
    experiment_name: str = "nvsyn_cosmos25_rf"
    resume_from_checkpoint: str = None
    val_check_interval: float = 1.0
    log_every_n_steps: int = 100
    seed: int = 42

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def get_callbacks(config: TrainingConfig):
    """
    Create list of Lightning callbacks for training.

    Args:
        config: Training configuration object.

    Returns:
        List of configured Lightning callback instances.
    """
    logger.info(
        f"Configuring callbacks for training (EMA: {config.enable_ema}, Gradient Clip: {config.gradient_clip_val})"
    )

    # List of callbacks
    callbacks = [
        RichProgressBar(),
        # Rolling checkpoint: keep last 3 epochs
        ModelCheckpoint(
            dirpath=os.path.join(
                config.output_dir, config.experiment_name, "checkpoints"
            ),
            filename="epoch={epoch:04d}",
            every_n_epochs=1,
            save_top_k=3,
            monitor="epoch",
            mode="max",
            save_last=True,
        ),
        # Best checkpoint: lowest validation loss
        ModelCheckpoint(
            dirpath=os.path.join(
                config.output_dir, config.experiment_name, "checkpoints"
            ),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
        # Learning Rate Monitor
        LearningRateMonitor(logging_interval="step"),
        # Tensorboard callback for logging generated samples and metrics
        TensorBoardCallback(
            num_frames_to_log=8, log_every_n_epochs=1, generate_every_n_epochs=1
        ),
    ]

    if config.enable_ema:
        # EMA monitoring callback to log EMA stats
        callbacks.append(EMAMonitorCallback(log_every_n_steps=500, warmup_steps=100))

    # Gradient clipping monitoring callback
    callbacks.append(
        GradClipCallback(
            clip_norm=config.gradient_clip_val,
            force_finite=True,
            log_every_n_steps=config.log_every_n_steps,
        )
    )

    # Gradient norm monitoring callback
    callbacks.append(
        GradientMonitorCallback(log_every_n_steps=config.log_every_n_steps)
    )

    return callbacks


def get_logger(config: TrainingConfig):
    """
    Create TensorBoard logger for experiment tracking.

    Args:
        config: Training configuration object.

    Returns:
        TensorBoardLogger instance with timestamped version.
    """
    # Use timestamp for version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Creating TensorBoard logger: {config.experiment_name}/{timestamp}")

    return TensorBoardLogger(
        save_dir=config.output_dir,
        name=config.experiment_name,
        version=timestamp,
    )


def get_strategy(config: TrainingConfig):
    """
    Get distributed training strategy based on configuration.

    Args:
        config: Training configuration object.

    Returns:
        Configured Lightning strategy instance (DDPStrategy, FSDPStrategy, or "auto").
    """
    strategy_name = config.strategy.lower()
    logger.info(f"Configuring strategy: {strategy_name} (GPUs: {config.num_gpus})")

    if config.num_gpus == 1:
        return "auto"

    if strategy_name == "auto":
        strategy_name = "ddp"

    if strategy_name == "ddp":
        # Disable static_graph when using gradient accumulation
        use_static_graph = config.accumulate_grad_batches <= 1

        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=use_static_graph,
            broadcast_buffers=False,
        )
    elif strategy_name == "fsdp":
        from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
        from cosmos_predict2._src.predict2.networks.minimal_v4_dit import Block
        return FSDPStrategy(
            auto_wrap_policy={MinimalV1LVGDiT, Block},
            use_orig_params=True,
            cpu_offload=False,
            activation_checkpointing_policy=None,
            sharding_strategy="FULL_SHARD",
            state_dict_type="full",
        )
    else:
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. Valid: 'auto', 'ddp', 'fsdp'"
        )


def get_datamodule(config: TrainingConfig) -> PreRenderedDataModule:
    """
    Create datamodule instance for pre-rendered 93-view 360° rotation projections or pre-encoded latents.

    Args:
        config: Training configuration object.

    Returns:
        PreRenderedDataModule instance.
    """
    logger.info(
        f"Creating PreRenderedDataModule with dataset path {config.dataset_path}, use_latent_cache={config.use_latent_cache}, batch size {config.batch_size}"
    )

    return PreRenderedDataModule(
        dataset_path=config.dataset_path,
        use_latent_cache=config.use_latent_cache,
        latent_cache_dir=config.latent_cache_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        seed=config.seed,
    )


def get_model(config: TrainingConfig) -> CosmosXRay360:
    """
    Create Lightning Module instance for Cosmos-Predict 2.5 post-training.
    """
    logger.info(
        f"Creating model: {config.model_size} with checkpoint {config.checkpoint_path} and tokenizer {config.tokenizer_path}"
    )

    return CosmosXRay360(
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        text_encoder_path=config.text_encoder_path,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_iters=config.max_iters,
        loss_scale=config.loss_scale,
        enable_ema=config.enable_ema,
        ema_rate=config.ema_rate,
        ema_offload_cpu=config.ema_offload_cpu,
        ema_iteration_shift=config.ema_iteration_shift,
        gradient_clip_val=config.gradient_clip_val,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        rf_shift=config.rf_shift,
        model_size=config.model_size,
        sac_mode=config.sac_mode,
        distributed_strategy=config.strategy,
        ema_sync_every_n_steps=config.ema_sync_every_n_steps,
        gamma_side=config.gamma_side,
        loss_atten_weight=config.loss_atten_weight,
    )


def train(config: TrainingConfig):
    """
    Main training loop for post-training Cosmos-Predict 2.5.

    Args:
        config: Training configuration object with all settings.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    logger.info(f"Starting training with config: {config}")

    # Set random seed for reproducibility
    seed_everything(config.seed, workers=True)

    # Create datamodule
    datamodule = get_datamodule(config)

    # If only building cache, run setup and exit
    if config.build_cache_only:
        datamodule.setup()
        return

    # Get the Lightning Module for training
    model = get_model(config)

    # Get TensorBoard logger for experiment tracking.
    tb_logger = get_logger(config)

    # Get all callbacks
    callbacks = get_callbacks(config)

    # Get the distributed training strategy
    strategy = get_strategy(config)

    # Create the Lightning Trainer with all configurations
    trainer = Trainer(
        accelerator="gpu",
        devices=config.num_gpus,
        strategy=strategy,
        precision=config.precision,
        max_steps=config.max_iters,
        gradient_clip_val=None,
        accumulate_grad_batches=config.accumulate_grad_batches,
        val_check_interval=config.val_check_interval,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=False,
        benchmark=True,
    )

    # Start training, optionally resuming from checkpoint
    trainer.fit(model, datamodule, ckpt_path=config.resume_from_checkpoint)


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        Namespace object with all training configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description="Post-training Cosmos-Predict 2.5 with Chest CT dataset."
    )

    # Checkpoint resumption
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., 'outputs/experiment_name/checkpoints/epoch=0005.ckpt' or 'outputs/experiment_name/checkpoints/best.ckpt')",
    )

    # Model configuration
    parser.add_argument(
        "--model_size", type=str, default="2B", choices=["2B", "7B", "14B"]
    )
    parser.add_argument(
        "--sac_mode", type=str, default="predict2_2b_720_aggressive", help="Selective Activation Checkpointing (SAC) mode"
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--text_encoder_path", type=str, default=None)

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2 ** (-14.5))
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_iters", type=int, default=100000)

    # Physics & Geometric regularization
    parser.add_argument(
        "--gamma_side",
        type=float,
        default=1.0,
        help="Angular-offset loss weight w(theta_k) = 1 + gamma_side * sin^2(theta_k); 0.0 disables angular weighting (ablation Variant A/baseline).",
    )
    parser.add_argument(
        "--loss_atten_weight",
        type=float,
        default=0.02,
        help="Weight lambda_atten for the global attenuation mass loss L_atten; 0.0 disables it (ablation Variants A/B/C).",
    )

    # EMA configuration
    parser.add_argument("--enable_ema", action="store_true", default=True)
    parser.add_argument(
        "--ema_rate",
        type=float,
        default=0.10,
        help="EMA decay rate (default 0.10 for faster adaptation during post-training)",
    )
    parser.add_argument(
        "--ema_offload_cpu",
        action="store_true",
        default=False,
        help="Offload EMA parameters to CPU to save GPU memory (default False for faster updates during post-training)",
    )

    # Hardware and optimization
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", "--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory in DataLoader")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="Keep DataLoader workers active across epochs")

    # Distributed strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "fsdp"],
        help="Distributed training strategy. 'auto' will choose DDP for multi-GPU and single-node. 'ddp' uses DistributedDataParallel. 'fsdp' uses FullyShardedDataParallel.",
    )
    parser.add_argument(
        "--ema_sync_every_n_steps",
        type=int,
        default=10,
        help="[DEPRECATED — no-op now that net_ema is FSDP-sharded; kept for CLI backward-compat with existing launch scripts]",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=100,
        help="Logging frequency in training steps for TensorBoard metrics",
    )

    # Data configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/pre_rendered",
        help="Path to pre-rendered dataset directory containing 'train' and 'test' subdirectories.",
    )
    parser.add_argument(
        "--use_latent_cache",
        action="store_true",
        default=False,
        help="Enable loading pre-encoded Wan2.1 VAE latent tensors z_0 from disk (bypasses VAE encoder).",
    )
    parser.add_argument(
        "--latent_cache_dir",
        type=str,
        default="datasets/pre_rendered_latents",
        help="Directory containing pre-encoded VAE latents (default: datasets/pre_rendered_latents).",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--img_shape", type=int, default=256)
    parser.add_argument(
        "--enable_disk_cache",
        action="store_true",
        default=True,
        help="Enable disk caching of preprocessed data to reduce memory usage (default True for large datasets during post-training)",
    )
    parser.add_argument(
        "--cache_strategy", type=str, default="disk", choices=["memory", "disk"]
    )
    parser.add_argument(
        "--build_cache_only",
        action="store_true",
        help="Only build the cache and exit without training (useful for large datasets to prepare cache before training)",
    )

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--experiment_name", type=str, default="nvsyn_cosmos25_rf")

    # Random seed
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # Get training configuration from command-line arguments
    args = parse_args()

    # Create default training configuration
    config = TrainingConfig()

    # Override default config with command-line arguments
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)

    logger.info("=" * 80)
    logger.info(f"Training Configuration: {config.experiment_name}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Model: {config.model_size} | Strategy: {config.strategy}")
    logger.info(f"GPUs: {config.num_gpus} | Precision: {config.precision}")
    logger.info("=" * 80)

    # Start the training process
    train(config)
