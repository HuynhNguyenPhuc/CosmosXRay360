#!/usr/bin/env python3
"""
Fast concurrent download of dataset files from HuggingFace repository.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

MAX_RATE_LIMIT_RETRIES = 6

def download_file(repo_id, filename, repo_type, local_dir):
    # Check if already downloaded
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return filename, "skipped"

    # Multiple training VMs share one HF account/token, so a 429 here usually
    # means the *account's* 5-minute quota is briefly exhausted, not that this
    # particular file is unavailable - retry with backoff instead of giving up.
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            return filename, "downloaded"
        except HfHubHTTPError as e:
            status = e.response.status_code if e.response is not None else None
            is_last_attempt = attempt == MAX_RATE_LIMIT_RETRIES - 1
            if status == 429 and not is_last_attempt:
                wait_s = min(60, 2**attempt) + random.uniform(0, 1)
                time.sleep(wait_s)
                continue
            return filename, f"failed: {e}"
        except Exception as e:
            return filename, f"failed: {e}"
    return filename, "failed: exhausted rate-limit retries"

def main():
    project_root = Path(__file__).resolve().parents[1]
    default_datasets_dir = str(project_root / "datasets")

    parser = argparse.ArgumentParser(description="Download dataset files from Hugging Face")
    parser.add_argument("--repo_id", type=str, default="hvcl-gm/chest-medical-image-dataset", help="HF Repo ID")
    parser.add_argument("--repo_type", type=str, default="dataset", help="HF Repo type")
    parser.add_argument("--local_dir", type=str, default=default_datasets_dir, help="Target local directory")
    parser.add_argument("--percentage", type=float, default=1.0, help="Fraction of dataset to download (1.0 = full dataset)")
    parser.add_argument("--max_workers", type=int, default=16, help="Parallel download threads")
    args = parser.parse_args()

    repo_id = args.repo_id
    repo_type = args.repo_type
    local_dir = args.local_dir
    percentage = args.percentage

    print(f"Fetching file list from Hugging Face repository: {repo_id}...")
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    
    nii_files = [f for f in files if f.endswith('.nii.gz')]
    
    all_mela = sorted([f for f in nii_files if "MELA" in f])
    all_nsclc = sorted([f for f in nii_files if "NSCLC" in f])
    all_tcia = sorted([f for f in nii_files if "TCIA" in f])
    
    if percentage >= 1.0:
        mela_files = all_mela
        nsclc_files = all_nsclc
        tcia_files = all_tcia
    else:
        mela_files = all_mela[:int(len(all_mela) * percentage)]
        nsclc_files = all_nsclc[:int(len(all_nsclc) * percentage)]
        tcia_files = all_tcia[:int(len(all_tcia) * percentage)]
    
    selected_files = mela_files + nsclc_files + tcia_files
    pct_str = "100% FULL" if percentage >= 1.0 else f"{int(percentage*100)}%"
    print(f"Selected {len(selected_files)} files to download ({pct_str} dataset).")
    print(f" - MELA: {len(mela_files)} / {len(all_mela)}")
    print(f" - NSCLC: {len(nsclc_files)} / {len(all_nsclc)}")
    print(f" - TCIA: {len(tcia_files)} / {len(all_tcia)}")
    os.makedirs(local_dir, exist_ok=True)

    max_workers = args.max_workers
    print(f"Starting concurrent download with {max_workers} threads...")
    
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, repo_id, fname, repo_type, local_dir): fname 
            for fname in selected_files
        }
        
        for future in as_completed(futures):
            filename, status = future.result()
            if status == "downloaded":
                downloaded_count += 1
                print(f"✓ Downloaded [{downloaded_count + skipped_count + failed_count}/{len(selected_files)}]: {filename}")
            elif status == "skipped":
                skipped_count += 1
            else:
                failed_count += 1
                print(f"✗ Failed: {filename} - {status}")

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Already cached / Skipped: {skipped_count}")
    print(f"Newly downloaded:       {downloaded_count}")
    print(f"Failed:                 {failed_count}")
    print(f"Total target files:     {len(selected_files)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
