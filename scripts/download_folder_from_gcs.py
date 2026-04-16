#!/usr/bin/env python3
"""
Download directory from Google Cloud Storage (GCS) to local path.

Usage:
    python scripts/download_from_gcs.py --bucket graphicsminer-data-science-bucket --gcs-folder data/ChestCT/ --local-dir data/ --num-workers 8

Environment:
    Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
"""

import os
import time
import argparse
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

from predict2_5.utils import get_logger


# --- Configuration --- #
DEFAULT_NUM_WORKERS = 8
DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1  # seconds

# --- Logger --- #
logger = get_logger(__name__)


class DownloadStats:
    """Track download statistics."""
    
    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.total_bytes = 0
        self.start_time = None
        self.failed_files = []
    
    def elapsed(self):
        """Elapsed time in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def bandwidth_mbps(self):
        """Average bandwidth in MB/s."""
        elapsed = self.elapsed()
        if elapsed > 0:
            return (self.total_bytes / (1024**2)) / elapsed
        return 0
    
    def print_summary(self):
        """Print download summary."""
        logger.info("%s", "=" * 60)
        logger.info("Download Summary")
        logger.info("%s", "=" * 60)
        logger.info("Total files processed: %s", self.total_files)
        logger.info("Successful: %s", self.successful)
        logger.info("Failed: %s", self.failed)
        logger.info("Skipped: %s", self.skipped)
        logger.info("Total data: %.2f GB", self.total_bytes / (1024**3))
        logger.info("Time elapsed: %.1f seconds", self.elapsed())
        logger.info("Average bandwidth: %.2f MB/s", self.bandwidth_mbps())
        
        if self.failed_files:
            logger.error("Failed files: %s", len(self.failed_files))

            for fname, error in self.failed_files:
                logger.error("- %s: %s", fname, error)

        logger.info("%s", "=" * 60)


def download_single_file(
    bucket: storage.Bucket,
    blob_name: str,
    local_path: str,
    stats: DownloadStats,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY,
):
    """
    Download a single file from GCS with retry logic.

    Args:
        bucket: GCS bucket object.
        blob_name: Name of the blob in GCS.
        local_path: Path to save the downloaded file locally.
        stats: DownloadStats object to update statistics.
        chunk_size: Chunk size for multipart downloads in bytes (default: 32 MB).
        max_retries: Maximum retry attempts for failed downloads (default: 3).
        retry_delay: Initial delay between retries in seconds (default: 1).

    Returns:
        True if download succeeded, False otherwise.
    """
    try:
        # Create local directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with retry logic
        for attempt in range(max_retries):
            try:
                blob = bucket.blob(blob_name)
                blob.chunk_size = chunk_size
                
                # Check if file exists
                if not blob.exists():
                    stats.skipped += 1
                    return True
                
                # Download the blob to local path
                blob.download_to_filename(local_path)
                
                # Update stats
                file_size = os.path.getsize(local_path)
                stats.total_bytes += file_size
                stats.successful += 1

                return True
                
            except GoogleAPIError as e:
                if attempt < max_retries - 1:
                    # Exponential backoff before retrying
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise
        
    except Exception as e:
        stats.failed += 1
        stats.failed_files.append((blob_name, str(e)))

        return False


def list_blobs(bucket: storage.Bucket, prefix: str) -> list[storage.Blob]:
    """
    List all blobs in bucket with given prefix.

    Args:
        bucket: GCS bucket object.
        prefix: Prefix to filter blobs (e.g., folder path).

    Returns:
        List of Blob objects matching the prefix.    
    """
    blobs = []

    for blob in bucket.list_blobs(prefix=prefix):
        # Skip directories (blobs ending with /)
        if not blob.name.endswith("/"):
            blobs.append(blob)

    return blobs


def download_from_gcs_parallel(
    bucket_name: str,
    gcs_folder: str,
    local_dir: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """
    Download entire folder from GCS using parallel downloads.

    Args:
        bucket_name: Name of the GCS bucket (e.g., 'graphicsminer-data-science-bucket').
        gcs_folder: Folder path in GCS to download (e.g., 'data/ChestCT/').
        local_dir: Local directory to save downloaded files (e.g., 'data/').
        num_workers: Number of parallel download threads (default: 8).
        chunk_size: Chunk size for multipart downloads in bytes (default: 32 MB).

    Returns:
        True if all downloads succeeded, False if any failed.
    """
    logger.info("%s", "=" * 60)
    logger.info("Google Cloud Storage Download (Parallel)")
    logger.info("%s", "=" * 60)
    logger.info("Source bucket: %s", bucket_name)
    logger.info("Source folder: %s", gcs_folder)
    logger.info("Destination directory: %s", local_dir)
    logger.info("Parallel workers: %s", num_workers)
    logger.info("Chunk size: %.0f MB", chunk_size / (1024**2))
    logger.info("%s", "=" * 60)
    
    try:
        # Initialize GCS client
        storage_client = storage.Client()

        # Get bucket object
        bucket = storage_client.bucket(bucket_name)

    except Exception as e:
        logger.error("Error when accessing GCS bucket, check authentication and permissions: %s", e)
        
        return False

    # List blobs to download
    blobs = list_blobs(bucket, prefix=gcs_folder)
    if not blobs:
        logger.warning("No files found in '%s' to download", gcs_folder)
        return True

    # Initialize download statistics
    stats = DownloadStats()
    stats.total_files = len(blobs)
    stats.start_time = time.time()
    
    logger.info("Found %s files to download", len(blobs))

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        # Submit all download tasks
        for blob in blobs:
            # Compute local path by removing GCS folder prefix
            relative_path = blob.name[len(gcs_folder):].lstrip("/")
            local_path = os.path.join(local_dir, relative_path)
            
            # Submit download task to executor
            future = executor.submit(
                download_single_file,
                bucket=bucket,
                blob_name=blob.name,
                local_path=local_path,
                stats=stats,
                chunk_size=chunk_size,
            )
            futures[future] = (blob.name, local_path)
        
        # Process completed downloads with progress bar
        with tqdm(as_completed(futures), total=len(futures), desc="Downloading from GCS") as pbar:
            for future in pbar:
                try:
                    # Wait for the download to complete and check result
                    future.result()

                except Exception as e:
                    blob_name, local_path = futures[future]

                    stats.failed += 1
                    stats.failed_files.append((blob_name, str(e)))
                
                pbar.update(1)
    
    stats.print_summary()
    return stats.failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from Google Cloud Storage.")
    
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name.",
    )
    parser.add_argument(
        "--gcs-folder",
        type=str,
        required=True,
        help="Source folder in GCS to download.",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory to save downloaded files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel download threads (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for downloads in bytes (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts for failed downloads (default: {DEFAULT_MAX_RETRIES}).",
    )
    
    args = parser.parse_args()
    
    # Create local directory if needed
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Run download
    success = download_from_gcs_parallel(
        bucket_name=args.bucket,
        gcs_folder=args.gcs_folder,
        local_dir=args.local_dir,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )
    
    # Exit with status code (0 for success, 1 for failure)
    exit(0 if success else 1)
