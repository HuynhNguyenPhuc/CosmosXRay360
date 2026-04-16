"""Logging utilities."""

import sys
import logging

from predict2_5.utils.distributed import get_local_rank


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger configured for distributed training.

    Suppress logs from non-master ranks to avoid clutter in DDP.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance configured for DDP.
    """
    # Create a logger with the specified name
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Get the local rank of the current process
        rank = get_local_rank()

        if rank != 0:
            # If non-master rank, set logger to WARNING to suppress output
            logger.setLevel(logging.WARNING)
        else:
            # If master rank, set logger to INFO for detailed output
            logger.setLevel(logging.INFO)

        if rank == 0:
            # Add a StreamHandler to output logs to stderr for the master rank
            handler = logging.StreamHandler(sys.stderr)

            # Set a simple formatter that only outputs the message for cleaner logs
            formatter = logging.Formatter("%(message)s")

            # Configure the handler with the formatter and add it to the logger
            handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(handler)

        # Prevent log messages from propagating to the root logger to avoid duplicate logs
        logger.propagate = False

    return logger


def setup_early_logging() -> None:
    """Supress early logs from non-master ranks before the main logger is set up."""

    import warnings

    # Ignore some specific warnings that are from third-party libraries and not relevant to our logging setup
    warnings.filterwarnings("ignore", message=".*_extra_state.*")
    warnings.filterwarnings("ignore", message=".*FP8.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformer_engine.*")

    # Suppress logs from some libraries that are commonly noisy in distributed settings
    for logger_name in ["torch.distributed", "torch.nn.parallel", "PIL", "matplotlib"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if get_local_rank() != 0:
        # Suppress all logs from non-master ranks to avoid clutter in DDP
        logging.getLogger().setLevel(logging.WARNING)

        # Also suppress logs from PyTorch Lightning which can be verbose in distributed settings
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

        try:
            from loguru import logger as loguru_logger

            # Disable loguru logs from non-master ranks if loguru is being used in the project
            loguru_logger.disable("")

        except ImportError:
            pass
