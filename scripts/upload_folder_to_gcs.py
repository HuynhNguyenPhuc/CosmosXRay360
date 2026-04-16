"""
Upload local directory to Google Cloud Storage (GCS).

Usage:
    python scripts/upload_to_gcs.py \\
        --source-dir data/ \\
        --bucket graphicsminer-data-science-bucket \\
        --gcs-folder data/ChestCT/ \\
        --num-workers 8 \\
        --compress

Environment:
    Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
"""

import os
import time
import gzip
import shutil
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


class UploadStats:
    """Track upload statistics."""
    
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
        """Print upload summary."""
        logger.info("%s", "=" * 60)
        logger.info("Upload Summary")
        logger.info("%s", "=" * 60)
        logger.info("Total files processed: %s", self.total_files)
        logger.info("Successful: %s", self.successful)
        logger.info("Failed: %s", self.failed)
        logger.info("Skipped: %s", self.skipped)
        logger.info("Total data: %.2f GB", self.total_bytes / (1024**3))
        logger.info("Time elapsed: %.1f seconds", self.elapsed())
        logger.info("Average bandwidth: %.2f MB/s", self.bandwidth_mbps())
        
        if self.failed_files:
            logger.error("Failed files (%s):", len(self.failed_files))

            for fname, error in self.failed_files:
                logger.error("- %s: %s", fname, error)

        logger.info("%s", "=" * 60)


def compress_file(local_path, temp_path):
    """Compress a file using gzip."""
    try:
        with open(local_path, 'rb') as f_in:
            with gzip.open(temp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return temp_path
    
    except Exception as e:
        logger.error(f"Compression failed for {local_path}: {e}")

        return local_path


def upload_single_file(
    bucket: storage.Bucket,
    local_path: str,
    gcs_path: str,
    stats: UploadStats,
    compress: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY
):
    """
    Upload a single file to GCS.

    Args:
        bucket: GCS bucket object.
        local_path: Path to local file to upload.
        gcs_path: Destination path in GCS bucket.
        stats: UploadStats object to update statistics.
        compress: Whether to compress the file before uploading (default: False).
        chunk_size: Chunk size for multipart uploads in bytes (default: 32 MB).
        max_retries: Maximum retry attempts for failed uploads (default: 3).
        retry_delay: Initial delay between retries in seconds (default: 1).

    Returns:
        True if upload succeeded, False otherwise.
    """
    temp_compressed = None
    upload_path = local_path
    
    try:
        # Compress if enabled
        if compress:
            temp_compressed = f"{local_path}.gz"
            upload_path = compress_file(local_path, temp_compressed)
            gcs_path = f"{gcs_path}.gz"
        
        # Update total bytes for statistics
        file_size = os.path.getsize(upload_path)
        stats.total_bytes += file_size
        
        # Upload with retry logic
        for attempt in range(max_retries):
            try:
                # Set chunk size for large files to enable resumable uploads
                blob = bucket.blob(gcs_path)
                blob.chunk_size = chunk_size
                blob.upload_from_filename(upload_path)
                
                # Update stats on successful upload
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
        # Update stats on failure upload
        stats.failed += 1
        stats.failed_files.append((local_path, str(e)))

        return False
    
    finally:
        # Clean up temporary compressed file
        if temp_compressed and os.path.exists(temp_compressed):
            try:
                os.remove(temp_compressed)
            except:
                pass


def collect_files(local_dir: str):
    """
    Collect all file paths from the local directory and its subdirectories.
    
    Args:
        local_dir: Path to the local directory.

    Returns:
        List of file paths.
    """
    filepaths = []

    for root, _, files in os.walk(local_dir):
        for filename in files:
            filepaths.append(os.path.join(root, filename))

    return filepaths


def upload_to_gcs_parallel(
    local_dir: str,
    bucket_name: str,
    gcs_folder: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
    compress: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """
    Upload local directory to GCS in parallel using multiple threads.

    Args:
        local_dir: Path to local directory to upload.
        bucket_name: GCS bucket name.
        gcs_folder: Destination folder in GCS bucket.
        num_workers: Number of parallel upload threads (default: 8).
        compress: Whether to compress files before uploading (default: False).
        chunk_size: Chunk size for multipart uploads in bytes (default: 32 MB).
    """
    logger.info("%s", "=" * 60)
    logger.info("Google Cloud Storage Upload (Parallel)")
    logger.info("%s", "=" * 60)
    logger.info("Source directory: %s", local_dir)
    logger.info("Destination bucket: %s", bucket_name)
    logger.info("Destination folder: %s", gcs_folder)
    logger.info("Parallel workers: %s", num_workers)
    logger.info("Compression: %s", "enabled" if compress else "disabled")
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

    # Collect files
    filepaths = collect_files(local_dir)
    if not filepaths:
        logger.warning("No files found in '%s' to upload", local_dir)
        return True

    # Initialize upload statistics
    stats = UploadStats()
    stats.total_files = len(filepaths)
    stats.start_time = time.time()
    
    logger.info("Found %s files to upload", len(filepaths))

    # Upload with thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        # Submit all upload tasks
        for local_path in filepaths:
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_folder, relative_path).replace("\\", "/")
            
            # Submit upload task to thread pool
            future = executor.submit(
                upload_single_file, 
                bucket=bucket, 
                local_path=local_path, 
                gcs_path=gcs_path, 
                stats=stats,
                compress=compress,
                chunk_size=chunk_size,
            )

            # Store future for tracking
            futures[future] = (local_path, gcs_path)
        
        # Process completed uploads with progress bar
        with tqdm(as_completed(futures), total=len(futures), desc="Uploading to GCS") as pbar:
            for future in pbar:
                try:
                    # Get the result when the thread completes
                    future.result()

                except Exception as e:
                    local_path, gcs_path = futures[future]

                    stats.failed += 1
                    stats.failed_files.append((local_path, str(e)))
                
                pbar.update(1)
    
    # Print final upload summary
    stats.print_summary()

    return stats.failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload local directory to Google Cloud Storage.")

    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to local directory to upload.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="GCS bucket name to upload to.",
    )
    parser.add_argument(
        "--gcs-folder",
        type=str,
        required=True,
        help="Destination folder in GCS bucket.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel upload threads (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size for multipart uploads in bytes (default: {DEFAULT_CHUNK_SIZE} bytes = 32 MB).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress files using gzip before uploading (default: False).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retry attempts for failed uploads (default: {DEFAULT_MAX_RETRIES}).",
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.isdir(args.source_dir):
        logger.error("Source directory '%s' does not exist", args.source_dir)
        exit(1)
    
    # Run the upload process
    success = upload_to_gcs_parallel(
        local_dir=args.source_dir,
        bucket_name=args.bucket,
        gcs_folder=args.gcs_folder,
        num_workers=args.num_workers,
        compress=args.compress,
        chunk_size=args.chunk_size,
    )
    
    # Exit with code 0 if successful, 1 if there were failures
    exit(0 if success else 1)
