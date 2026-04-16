"""HuggingFace download helpers for CosmosXRay2XRay."""

import os
import json
import shutil
import contextlib
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

from predict2_5.constants import CR1_HF_REVISION
from predict2_5.utils import get_logger


# --- Logger --- #
logger = get_logger(__name__)


# ============================================================
# HuggingFace Upload
# ============================================================

def upload_artifacts(
    repo_id: str,
    local_paths: list[str],
    path_in_repo_prefix: str = "",
    private: bool = True,
    create_repo_if_missing: bool = True,
) -> list[str]:
    """
    Upload local files to HuggingFace Hub repository.
    
    Args:
        repo_id (str): The HuggingFace repository ID (e.g., "username/repo_name").
        local_paths (list[str]): List of local file paths to upload.
        path_in_repo_prefix (str, optional): Prefix path in the repository. Defaults to "".
        private (bool, optional): Whether the repository is private. Defaults to True.
        create_repo_if_missing (bool, optional): Whether to create the repository if it doesn't exist. Defaults to True.
    Returns:
        list[str]: List of uploaded file paths in the repository.
    """
    from huggingface_hub import HfApi

    # Resolve authentication token
    token = resolve_hf_token()
    if not token:
        raise RuntimeError("No HuggingFace token found. Set HF_TOKEN or run huggingface-cli login.")

    # Initialize API client
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    if create_repo_if_missing:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    uploaded_paths: list[str] = []
    for local_path in local_paths:
        src = Path(local_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Upload source not found or not file: {src}")

        dst_name = src.name if not path_in_repo_prefix else f"{path_in_repo_prefix.rstrip('/')}/{src.name}"
        logger.info(f"Uploading: {src} -> {dst_name}")

        # Use API to upload file (handles large files and retries)
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst_name,
            repo_id=repo_id,
            repo_type="model",
        )

        uploaded_paths.append(dst_name)

    logger.info(f"Uploaded {len(uploaded_paths)} artifact(s) to {repo_id}")

    return uploaded_paths


# ============================================================
# HuggingFace Download
# ============================================================

def resolve_hf_token() -> str:
    """
    Resolve HuggingFace authentication token.

    Order of precedence:
    1. Environment variables: HF_TOKEN, HUGGING_FACE_HUB_TOKEN
    2. Config files: ~/.cache/huggingface/token, ~/.huggingface/token
    3. HuggingFace library API (get_token)
    4. If none found, return empty string (anonymous access)
    """
    # Check environment variables
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    if token:
        return token

    # Check config files
    for token_path in [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]:
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                return token

    # Try library API
    try:
        from huggingface_hub.utils import get_token

        token = get_token() or ""
        if token:
            return token
        
    except Exception:
        pass

    return ""


def _get_lock_manager():
    """
    Get a file lock manager for DDP-safe downloads.
    
    Returns:
    - A context manager class for file locking (e.g., FileLock from filelock library).
    - If filelock is not available, returns a no-op context manager.
    """
    try:
        from filelock import FileLock

        return FileLock
    
    except ImportError:
        return lambda _p: contextlib.nullcontext()


def curl_download(
    url: str, 
    dest: Path, 
    token: str, 
    attempt: int
) -> None:
    """
    Execute a single curl download with resume capability.
    
    Args:
        url (str): The URL to download.
        dest (Path): The destination file path.
        token (str): The HuggingFace authentication token.
        attempt (int): The current download attempt number.
    """
    cmd = [
        "curl",
        "-fL",                    # fail on HTTP error, follow redirects
        "-C", "-",                # resume partial downloads
        "--speed-limit", "102400",  # min speed: 100 KB/s
        "--speed-time", "30",       # for 30 consecutive seconds
        "-o", str(dest),
        url,
    ]
    
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])

    # Execute the curl command
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.warning(
            f"Curl attempt {attempt}/5 failed (exit code={result.returncode}); "
            f"retrying …"
        )


def hf_download(
    repo_id: str,
    filename: str,
    revision: str = "main",
    override_dest: str = None,
) -> str:
    """
    Download a HuggingFace file via curl with resume capability.

    Args:
        repo_id (str): The HuggingFace repository ID (e.g., "nvidia/Cosmos-Reason1-7B").
        filename (str): The filename within the repository to download.
        revision (str): The git revision (branch, tag, or commit) to download from.
        override_dest (str): Optional path to save the downloaded file. If None, uses HF cache.
    """
    # Fast path: already in local HF cache
    if override_dest is None:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_files_only=True,
            )
        
        except Exception:
            pass  # Fall through to curl download

    # Determine destination path
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"

    if override_dest is not None:
        # Use override destination (e.g., for manual cache)
        dest = Path(override_dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use manual cache directory under HF_HUB_CACHE to avoid conflicts with standard HF snapshot cache
        cache_dir = (
            Path(HF_HUB_CACHE) / "manual" / repo_id.replace("/", "--")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / Path(filename).name

    # Skip if already downloaded
    if dest.exists() and dest.stat().st_size > 1_000_000:
        logger.info(f"Reusing: {dest}")
        return str(dest)

    # Resolve authentication token
    token = resolve_hf_token()

    # Acquire a file lock to ensure only one process downloads at a time
    lock_manager = _get_lock_manager()
    lock_path = dest.parent / f".lock_{dest.name}"
    
    with lock_manager(str(lock_path)):
        # Re-check after acquiring lock (another process may have downloaded)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            logger.info(f"Already downloaded: {dest}")
            return str(dest)

        logger.info(f"Downloading: {repo_id}/{filename}")

        for attempt in range(1, 6):
            # Perform the download with curl
            curl_download(url, dest, token, attempt)

            # Check if download was successful
            if dest.exists() and dest.stat().st_size > 1_000_000:
                size_gb = dest.stat().st_size / 1e9
                logger.info(f"Download complete: {dest} ({size_gb:.2f} GB)")
                return str(dest)

    raise RuntimeError(
        f"Failed to download {repo_id}/{filename!r} "
        f"after 5 curl attempts"
    )


# ============================================================
# Text Encoder Snapshot (nvidia/Cosmos-Reason1-7B)
# ============================================================

_TE_REPO_ID = "nvidia/Cosmos-Reason1-7B"
_TE_REVISION = CR1_HF_REVISION
_TE_SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "chat_template.json",
]
_TE_SHARD_MIN_SIZE = 1_500_000_000  # 1.5 GB


def _get_manual_snapshot_dir() -> Path:
    """Get the manual snapshot cache directory."""
    cache_dir = (
        Path(HF_HUB_CACHE)
        / "manual"
        / _TE_REPO_ID.replace("/", "--")
        / _TE_REVISION[:8]
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def _check_hf_snapshot_cache() -> str:
    """Check if text encoder is complete in standard HuggingFace snapshot cache."""
    try:
        from huggingface_hub import snapshot_download
        
        # Try to locate the snapshot in the standard HuggingFace cache
        cached = snapshot_download(
            repo_id=_TE_REPO_ID,
            revision=_TE_REVISION,
            local_files_only=True,
        )
        
        # Verify all shards are complete (≥1.5 GB)
        cached_path = Path(cached)
        index_p = cached_path / "model.safetensors.index.json"
        
        if not index_p.exists():
            return None
        
        with open(index_p) as f:
            index = json.load(f)
        
        shards = sorted(set(index["weight_map"].values()))
        all_complete = all(
            (cached_path / s).exists()
            and (cached_path / s).stat().st_size >= _TE_SHARD_MIN_SIZE
            for s in shards
        )
        
        return cached if all_complete else None
    
    except Exception:
        return None


def _download_text_encoder_with_lock(
    snap_dir: Path, 
    done_file: Path
) -> str:
    """
    Download text encoder with DDP-safe locking.

    Args:
        snap_dir (Path): The directory to store the snapshot.
        done_file (Path): The sentinel file to indicate completion.

    Returns:
        str: The local directory path where the text encoder snapshot is stored.
    """
    # Acquire a lock to ensure only one process downloads at a time
    lock_manager = _get_lock_manager()
    lock_path = snap_dir / ".snapshot.lock"
    
    with lock_manager(str(lock_path)):
        # Double-check after acquiring lock
        if done_file.exists():
            logger.info(f"Complete (re-check): {snap_dir}")
            return str(snap_dir)
        
        # Download small files (config, tokenizer)
        _download_small_files(snap_dir)
        
        # Parse shard list from index
        index_path = snap_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        
        # Download large shards via curl
        _download_large_shards(snap_dir, shard_files)
        
        # Hard-link into HF snapshot dir for checkpoint_db
        _hardlink_to_hf_snapshot(snap_dir, shard_files)
        
        # Write completion sentinel
        done_file.write_text("done\n")
        logger.info(f"Download complete: {snap_dir}")

        return str(snap_dir)


def _download_small_files(snap_dir: Path) -> None:
    """
    Download small files (config, tokenizer) via HuggingFace API.

    Args:
        snap_dir (Path): The directory to store the snapshot.
    """
    for fname in _TE_SMALL_FILES:
        dest_f = snap_dir / fname

        if dest_f.exists():
            continue

        logger.info(f"Fetching: {fname}")

        # Use HuggingFace API for small files (fast and reliable)
        src = hf_hub_download(
            repo_id=_TE_REPO_ID,
            filename=fname,
            revision=_TE_REVISION,
        )

        # Copy to destination (API may return a cached path)
        shutil.copy2(src, dest_f)


def _download_large_shards(
    snap_dir: Path, 
    shard_files: list
) -> None:
    """
    Download large shard files via curl with resume capability.

    Args:
        snap_dir (Path): The directory to store the snapshot.
        shard_files (list): The list of shard files to download.
    """
    for shard in shard_files:
        dest_f = snap_dir / shard
        
        # Skip if already complete (1.5 GB threshold for large shards)
        if dest_f.exists() and dest_f.stat().st_size >= _TE_SHARD_MIN_SIZE:
            size_gb = dest_f.stat().st_size / 1e9
            logger.info(f"Shard complete: {shard} ({size_gb:.2f} GB)")
            continue
        
        # Download shard via curl with retries
        hf_download(
            _TE_REPO_ID,
            shard,
            revision=_TE_REVISION,
            override_dest=str(dest_f),
        )


def _hardlink_to_hf_snapshot(
    snap_dir: Path, 
    shard_files: list
) -> None:
    """
    Hard-link downloaded shards into the standard HuggingFace snapshot cache for checkpoint_db compatibility.
    
    Args:
        snap_dir (Path): The directory where the snapshot was downloaded.
        shard_files (list): The list of shard files to link.
    """
    try:
        hf_snap = (
            Path(HF_HUB_CACHE)
            / f"models--{_TE_REPO_ID.replace('/', '--')}"
            / "snapshots"
            / _TE_REVISION
        )
        
        if not hf_snap.exists():
            logger.info("Snapshot dir not found; checkpoint_db may re-download")
            return
        
        for shard in shard_files:
            hf_shard = hf_snap / shard
            src = snap_dir / shard
            
            # Skip if already complete
            if hf_shard.exists() and hf_shard.stat().st_size >= _TE_SHARD_MIN_SIZE:
                continue
            
            try:
                if hf_shard.exists():
                    # Remove incomplete shard before linking
                    hf_shard.unlink()

                # Create hard link (fast and space-efficient)
                hf_shard.hardlink_to(src)
                logger.info(f"Linked: {shard}")

            except OSError:
                # Fallback to copy if hardlink fails
                shutil.copy2(str(src), str(hf_shard))
                logger.info(f"Copied: {shard}")

    except Exception as e:
        logger.warning(f"Could not populate snapshot dir: {e}")


def download_text_encoder_snapshot() -> str:
    """
    Download and cache the text encoder snapshot with DDP safety.

    Returns:
        str: The local directory path where the text encoder snapshot is stored.
    """
    # Determine manual snapshot directory
    snap_dir = _get_manual_snapshot_dir()
    
    # Check standard HF snapshot cache
    cached = _check_hf_snapshot_cache()
    if cached:
        logger.info(f"Found in cache: {cached}")
        return cached
    
    # Check manual cache sentinel
    done_file = snap_dir / ".download_complete"
    if done_file.exists():
        logger.info(f"Manual snapshot already complete: {snap_dir}")
        return str(snap_dir)
    
    # Slow path: download with DDP concurrency control
    return _download_text_encoder_with_lock(snap_dir, done_file)
