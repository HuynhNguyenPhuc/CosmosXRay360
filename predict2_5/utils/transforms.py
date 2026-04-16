import os
import gc
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

import torch

from monai.transforms import MapTransform, Transform

from decord import VideoReader

from predict2_5.utils import get_logger


# --- Logger --- #
logger = get_logger(__name__)


# ============================================================
# Intensity Transforms
# ============================================================

class ClipMinIntensityDict(MapTransform):
    """
    Minimum intensity clipping for CT volumes to handle out-of-window values.

    Args:
        keys: Dictionary keys containing tensors to clip (e.g., ['ct_volume']).
        min_val: Minimum intensity value to clip to (default: -512 Hounsfield, typical for lung window).
    """

    def __init__(self, keys: List[str], min_val: float = -512):
        super().__init__(keys)
        self.min_val = min_val

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clip tensor values to a minimum threshold.

        Args:
            data: Dictionary containing tensors at specified keys to be clipped.

        Returns:
            Dictionary with tensors clipped to minimum value.
        """
        d = dict(data)
        for key in self.keys:
            d[key] = torch.clamp(d[key], min=self.min_val)
        return d


def correct_window(
    T_old: torch.Tensor,
    a_min: float = -1024,
    a_max: float = 3071,
    b_min: float = -512,
    b_max: float = 3071,
) -> torch.Tensor:
    """
    Rescale CT volume intensities from original Hounsfield window [a_min, a_max] to target window [b_min, b_max].

    This function maps the original Hounsfield units (e.g., -1024 to 3071) to a new range (e.g., -512 to 3071) and normalizes to [0, 1].
    It handles out-of-window values by clipping to the target range. 
    This is important for consistent preprocessing of CT volumes.

    Args:
        T_old: Input tensor with original Hounsfield units.
        a_min: Minimum of original Hounsfield window (default: -1024).
        a_max: Maximum of original Hounsfield window (default: 3071).
        b_min: Minimum of target Hounsfield window (default: -512).
        b_max: Maximum of target Hounsfield window (default: 3071).

    Returns:
        Tensor rescaled to [0, 1] based on target window.
    """
    # Calculate width of original and target windows
    range_old = a_max - a_min
    range_new = b_max - b_min

    # Map from original Hounsfield window to standard range [0, 1]
    T_raw = (T_old * range_old) + a_min
    T_new = (T_raw - b_min) / range_new

    # Clip to [0, 1] to handle out-of-window values
    return T_new.clamp(0, 1)


def rescaled(
    x: torch.Tensor, 
    val: float = 64, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Rescale tensor by a fixed value with numerical stability.

    Args:
        x: Input tensor to be rescaled.
        val: Value to rescale.
        eps: Epsilon for numerical stability to prevent division by zero.

    Returns:
        Rescaled tensor.
    """
    return (x + eps) / (val + eps)


def minimized(
    x: torch.Tensor, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Minimize tensor values to [0, 1] by dividing by the maximum value.

    Args:
        x: Input tensor to be minimized.
        eps: Epsilon for numerical stability to prevent division by zero.

    Returns:
        Minimized tensor in [0, 1] range.
    """
    return (x + eps) / (x.max() + eps)


def normalized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalize tensor to [0, 1] range.

    Args:
        x: Input tensor to be normalized.
        eps: Epsilon for numerical stability to prevent division by zero.

    Returns:
        Normalized tensor in [0, 1] range.
    """
    return (x - x.min() + eps) / (x.max() - x.min() + eps)


def standardized(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize tensor to zero mean and unit variance.

    Args:
        x: Input tensor to be standardized.
        eps: Epsilon for numerical stability to prevent division by zero.

    Returns:
        Standardized tensor: (x - mean) / (std + eps).
    """
    return (x - x.mean()) / (x.std() + eps)


# ============================================================
# I/O Transforms
# ============================================================

class LoadMP4d(Transform):
    """
    Load MP4 video file and convert to normalized float32 tensor in (C, T, H, W) format.

    Converts video frames from uint8 [0, 255] to float32 [0, 1].
    Transpose from (T, H, W, C) to (C, T, H, W) for further processing.

    Args:
        keys: Dictionary keys containing MP4 file paths to load.
        allow_missing_keys: If True, skip missing keys without error (default: False). If False, raise KeyError for missing keys.
    """

    def __init__(self, keys: List[str], allow_missing_keys: bool = False):
        super().__init__()

        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load MP4 video, normalize to [0, 1], and transpose to (C, T, H, W).

        Args:
            data: Dictionary containing MP4 file paths at specified keys.

        Returns:
            Dictionary with added video tensors in (C, T, H, W) format, or None on error.
        """
        # Create a copy of the input dictionary
        d = dict(data)

        for key in self.keys:
            # Check if key exists in dictionary
            if key not in d:
                if self.allow_missing_keys:
                    continue

                raise KeyError(f"Key '{key}' not found in data dictionary")

            # Get file path and verify it exists
            filepath = d[key]

            try:
                # Load video using decord VideoReader (efficient GPU-accelerated decoding)
                vr = VideoReader(filepath)

                # Read all frames into a numpy array (shape: T, H, W, C) with uint8 [0, 255]
                video_array = vr.get_batch(range(len(vr))).asnumpy()

                # Normalize uint8 [0, 255] to float32 [0, 1]
                video_array = video_array.astype(np.float32) / 255.0

                # Transpose from (T, H, W, C) -> (C, T, H, W) for further processing
                video_array = video_array.transpose(3, 0, 1, 2)

                # Store the processed video tensor in the output dictionary
                d[key] = video_array

            except Exception as e:
                logger.error("Failed to load MP4 video %s: %s", filepath, e)
                return None

        return d


class TranscodeMP4ToNPY(Transform):
    """
    Transcode MP4 video to .npy memmap format for efficient training data loading.

    Converts MP4 videos to memory-mapped .npy files containing uint8 frames in (T, H, W, C) format.
    This allows efficient frame-level access during training without loading entire videos into memory.

    Args:
        keys: Dictionary keys containing MP4 file paths to transcode.
        cache_dir: Directory to store transcoded .npy files (default: "./cache_videos").
        chunk_size: Number of frames to process in memory at once (default: 128). 
                    Larger chunks = faster but higher RAM; smaller = slower but lower RAM.
        allow_missing_keys: If True, skip missing keys without error (default: False). If False, raise KeyError for missing keys.
    """

    def __init__(
        self,
        keys: List[str],
        cache_dir: str = "./cache_videos",
        chunk_size: int = 128,
        allow_missing_keys: bool = False,
    ):
        super().__init__()

        self.keys = keys

        # Set up cache directory for transcoded .npy files
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = int(chunk_size)
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transcode MP4 video to .npy memmap format, processing in chunks to manage memory.

        Args:
            data: Dictionary containing MP4 file paths at specified keys.

        Returns:
            Dictionary with added memmap paths at keys like '<key>_npy', or None on error.
        """
        # Create a copy of the input dictionary
        d = dict(data)

        for key in self.keys:
            # Check if key exists in dictionary
            if key not in d:
                if self.allow_missing_keys:
                    continue

                raise KeyError(f"Key '{key}' not found in dictionary")

            mp4_path = Path(d[key])

            # Verify MP4 file exists
            if not mp4_path.exists():
                logger.error("File not found: %s", mp4_path)
                return None

            # Derive output memmap path from input filename
            stem = mp4_path.stem
            out_path = self.cache_dir / f"{stem}.npy"

            # Skip if already transcoded to avoid redundant work
            if out_path.exists():
                d[f"{key}_npy"] = str(out_path)
                logger.debug("Already cached: %s", out_path.name)
                continue

            try:
                # 1. Open video and validate frame count
                vr = VideoReader(str(mp4_path))

                # Get total number of frames in the video
                n_frames = len(vr)

                if n_frames == 0:
                    logger.error("No frames in %s", mp4_path)
                    return None

                # 2. Get frame dimensions from first frame
                first_frame = vr.get_batch([0]).asnumpy()[0]

                # Extract height, width, and channels from the first frame
                H, W, C = first_frame.shape

                # 3. Create temporary memmap file to write frames in chunks
                tmp_fd, tmp_file = tempfile.mkstemp(
                    suffix=".npy", 
                    dir=str(self.cache_dir)
                )

                # Close the file descriptor immediately since numpy will handle file writing
                os.close(tmp_fd)  

                # 4. Create memmap array with shape (n_frames, H, W, C) and dtype uint8 for efficient writing
                memmap = np.lib.format.open_memmap(
                    tmp_file, mode="w+", dtype=np.uint8, shape=(n_frames, H, W, C)
                )

                # 5. Write frames in chunks to keep RAM usage bounded
                for start in range(0, n_frames, self.chunk_size):
                    # Determine end index for the current chunk
                    end = min(n_frames, start + self.chunk_size)

                    # Get frame indices for the current chunk
                    frame_indices = list(range(start, end))

                    # Load chunk into RAM and write to memmap
                    frames_chunk = vr.get_batch(frame_indices).asnumpy()
                    memmap[start:end, ...] = frames_chunk

                    # Explicitly free temporary arrays to reduce peak memory
                    del frames_chunk
                    gc.collect()

                # 6. Flush memmap to disk and move from temporary location to final path
                memmap.flush()
                del memmap

                gc.collect()  # Force garbage collection to ensure file handle is closed before moving

                # After flushing and deleting memmap, we can safely move the temporary file to the final location
                os.replace(tmp_file, str(out_path))

                # 7. Update dictionary with memmap path
                d[f"{key}_npy"] = str(out_path)
                logger.info("Completed: %s -> %s (%d frames)", mp4_path.name, out_path.name, n_frames)

            except Exception as e:
                logger.error("Error processing %s: %s", mp4_path, e)
                return None

        return d


class LoadNPYd(Transform):
    """
    Load .npy memmap containing video frames for efficient access during training.

    Loads memmap from .npy file paths specified in the dictionary. 
    This allows efficient frame-level access during DataLoader iteration without loading the entire video into main process memory.

    Args:
        keys: Dictionary keys containing memmap paths to load (e.g., ['video']).
        memmap_key_suffix: Suffix to append to keys to find memmap paths (default: '_npy').
                          E.g., keys=['video'] -> looks for 'video_npy' in the dictionary.
        allow_missing_keys: If True, skip missing keys without error (default: False). If False, raise KeyError for missing keys.
    """

    def __init__(
        self,
        keys: List[str],
        memmap_key_suffix: str = "_npy",
        allow_missing_keys: bool = False,
    ):
        super().__init__()

        self.keys = keys
        self.suffix = memmap_key_suffix
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load .npy memmap for each specified key and store in the dictionary.

        For each key in self.keys, looks for a corresponding memmap path in data with the suffix (e.g., 'video_npy').
        Loads the memmap as a read-only numpy array and stores it back in data[key] for use during training.

        This transform runs in DataLoader workers, so the memmap is loaded in parallel across workers, allowing efficient access to video frames without loading entire videos into main memory.

        Args:
            data: Dictionary containing memmap paths at keys like '<key>_npy'.

        Returns:
            Dictionary with memmap arrays loaded at keys specified in self.keys, or None on error.
        """
        # Create a copy of the input dictionary
        d = dict(data)

        for key in self.keys:
            # Construct memmap key (e.g., 'video' -> 'video_npy')
            npy_key = f"{key}{self.suffix}"

            # Check if memmap path is available
            if npy_key not in d:
                if self.allow_missing_keys:
                    continue

                # Helpful error message if user forgot cache-building step
                if key in d and Path(d[key]).suffix == ".mp4":
                    logger.error("Memmap key '%s' not found", npy_key)
                    logger.error("Did you run build_cache_only=True first?")

                raise KeyError(f"Memmap key '{npy_key}' not found")

            # Get memmap path and verify it exists
            memmap_path = d[npy_key]

            # Verify .npy file exists
            if not Path(memmap_path).exists():
                logger.error("NPY file missing: %s", memmap_path)
                return None

            try:
                # Load as read-only memmap (memory-mapped access)
                # mmap_mode='r' is read-only, which allows efficient access to large files without loading into memory

                memmap_array = np.load(memmap_path, mmap_mode="r")
                d[key] = memmap_array

            except Exception as e:
                logger.error("Error loading memmap %s: %s", memmap_path, e)
                return None

        return d


class NPYToFloatd(Transform):
    """
    Convert uint8 video frames from .npy memmap to normalized float tensors in (C, T, H, W) format.

    Transforms video frames from uint8 [0, 255] to float32 [0, 1] and transposes from (T, H, W, C) to (C, T, H, W) for PyTorch processing.
    This transform runs in DataLoader workers after LoadNPYd, allowing efficient conversion of video frames to the format needed for training.

    Args:
        keys: Dictionary keys containing .npy memmap arrays to convert (e.g., ['video']).
        to_dtype: Target dtype for conversion (default: np.float32).
    """

    def __init__(self, keys: List[str], to_dtype: Any = np.float32):
        super().__init__()

        self.keys = keys
        self.to_dtype = to_dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert uint8 video frames from memmap to normalized float tensors in (C, T, H, W) format.

        For each key in self.keys, takes the uint8 memmap array (shape: T, H, W, C), normalizes to [0, 1], 
        converts to the specified dtype, and transposes to (C, T, H, W) for PyTorch processing.

        Args:
            data: Dictionary with uint8 arrays to convert (shape: T, H, W, C).

        Returns:
            Dictionary with float32 arrays in (C, T, H, W) format [0, 1] range.
        """
        d = dict(data)

        for key in self.keys:
            # Get array from dictionary (may be None if key optional)
            arr = d.get(key, None)

            if arr is None:
                continue

            # Transpose from (T, H, W, C) to (C, T, H, W) for further processing
            try:
                arr = arr.transpose(3, 0, 1, 2)
            except Exception:
                arr = np.transpose(arr, (3, 0, 1, 2))

            # Normalize from uint8 [0, 255] to float32 [0, 1] and convert dtype
            arr = arr.astype(self.to_dtype) / 255.0

            d[key] = arr

        return d


class LoadPromptd(Transform):
    """
    Load prompt from text file paths specified in the dictionary.

    For each key in self.keys, looks for a file path in data[key]. 
    If the file exists, reads the text content and stores it in data['prompt'].

    Args:
        keys: Dictionary keys containing text file paths to load (e.g., ['txt_path']).
        allow_missing_keys: If True, skip missing keys without error and set 'prompt' to empty string (default: True). If False, raise KeyError for missing keys.
    """

    def __init__(
        self,
        keys: List[str] = ["txt_path"],
        allow_missing_keys: bool = True
    ):
        super().__init__()

        self.keys = keys
        self.allow_missing_keys = allow_missing_keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load prompt text from file paths in the dictionary.

        For each key in self.keys, looks for a file path in data[key]. 
        If the file exists, reads the text content and stores it in data['prompt'].

        Args:
            data: Dictionary containing text file paths at specified keys.

        Returns:
            Dictionary with 'prompt' key containing loaded text, or empty string if file missing and allowed.
        """
        # Create a copy of the input dictionary
        d = dict(data)

        for key in self.keys:
            # Check if file path key exists in dictionary
            if key not in d:
                if not self.allow_missing_keys:
                    raise KeyError(f"File path key '{key}' not found in dictionary")
                
                # Default to empty prompt if file path missing and allowed
                d["prompt"] = ""

                continue

            # Extract file path
            txt_path = d[key]

            # Prompt placeholder
            prompt = ""

            # Load text from file if it exists
            if txt_path and Path(txt_path).exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()  # Strip whitespace before/after

                except Exception as e:
                    logger.warning("Could not read %s: %s", txt_path, e)
                    prompt = ""  # Use empty string on read error

            # Store text in output dictionary
            d["prompt"] = prompt

        return d
