"""
Upload model checkpoints and config file to HuggingFace Hub.

Usage:
	python scripts/upload_to_hf.py \
		--repo-id phuchuynh0904/CosmosXRay360 \
		--checkpoint-dir checkpoints \
		--public
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path


def _resolve_local_paths(checkpoint_dir: Path) -> list[Path]:
	"""
	Resolve the local paths for the required checkpoint files.

	Args:
		checkpoint_dir (Path): Directory containing the checkpoint files.

	Returns:
		list[Path]: List of resolved paths for net.pth, net_ema.pth, and config.json.
	"""
	# Define the required file names
	required_names = ["net.pth", "net_ema.pth", "config.json"]

	# Check for missing files
	missing = [name for name in required_names if not (checkpoint_dir / name).is_file()]
	if missing:
		missing_text = ", ".join(missing)
		raise FileNotFoundError(f"Missing required file(s) in {checkpoint_dir}: {missing_text}")

	return [checkpoint_dir / name for name in required_names]


def main() -> None:
	parser = argparse.ArgumentParser(description="Upload model checkpoints and config file to HuggingFace Hub.")
	
	parser.add_argument(
		"--repo-id",
		required=True,
		help='HuggingFace repo id, for example "username/CosmosXRay360".',
	)
	parser.add_argument(
		"--checkpoint-dir",
		default="checkpoints",
		help="Directory containing net.pth, net_ema.pth, and config.json.",
	)
	parser.add_argument(
		"--remote-prefix",
		default="",
		help="Optional prefix for the remote paths in the HuggingFace repo (e.g., 'checkpoints/').",
	)

	# Add mutually exclusive group for visibility options
	visibility_group = parser.add_mutually_exclusive_group()
	visibility_group.add_argument(
		"--private",
		action="store_true",
		help="Create repo as private if it needs to be created.",
	)
	visibility_group.add_argument(
		"--public",
		action="store_true",
		help="Create repo as public if it needs to be created.",
	)

	parser.add_argument(
		"--skip-create",
		action="store_true",
		help="Skip creating the repo if it does not exist. If this flag is set and the repo does not exist, the upload will fail.",
	)

	# Parse the arguments
	args = parser.parse_args()

	# Repository visibility
	repo_private = not args.public

	project_root = Path(__file__).resolve().parents[1]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))

	from predict2_5.hf import upload_artifacts

	# Resolve local paths for the required checkpoint files
	checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
	local_paths = _resolve_local_paths(checkpoint_dir)

	# Upload the artifacts to HuggingFace Hub
	uploaded = upload_artifacts(
		repo_id=args.repo_id,
		local_paths=[str(p) for p in local_paths],
		remote_prefix=args.remote_prefix,
		visibility="private" if repo_private else "public",
		ensure_repo=(not args.skip_create)
	)

	print("Upload completed:")
	for path in uploaded:
		print(f"- {path}")


if __name__ == "__main__":
	main()
