"""
Upload trained Hedwig weights + tokenizer + Model Card to Hugging Face.

Why: GitHub blocks files >100 MB, so large cloud brains (97.6M = 196 MB, 300M = ~600 MB)
live safely on Hugging Face instead.

First-time login (one-time, uses a Hugging Face Write token):
    .venv/bin/hf auth login

Upload (creates a PRIVATE repo by default; pass --public when ready to open-source):
    .venv/bin/python upload_hf.py <your-hf-username>/hedwig-97m
    .venv/bin/python upload_hf.py <your-hf-username>/hedwig-97m --public
    .venv/bin/python upload_hf.py <your-hf-username>/hedwig-300m --folder cloud_brains_300m
"""
import argparse, os, sys
from huggingface_hub import HfApi

def main():
    p = argparse.ArgumentParser(description="Upload a Hedwig brain folder to Hugging Face.")
    p.add_argument("repo_id", help="Target Hugging Face repo, e.g. ansarbuilds/hedwig-97m")
    p.add_argument("--folder", default="cloud_brains_97m", help="Local folder to upload (default: cloud_brains_97m)")
    p.add_argument("--public", action="store_true", help="Make the repo public (default: private)")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Folder not found: {args.folder}")
    files = sorted(f for f in os.listdir(args.folder) if not f.startswith("."))
    if not files:
        sys.exit(f"Folder is empty: {args.folder}")

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        sys.exit("Not logged in to Hugging Face yet.\nRun:  .venv/bin/hf auth login\n(Create a Write token at https://huggingface.co/settings/tokens)")

    vis = "PUBLIC" if args.public else "PRIVATE"
    print(f"🦉 Logged in as {who} | Uploading {len(files)} files from {args.folder}/ -> {args.repo_id} ({vis})...", flush=True)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.folder} weights, tokenizer, and model card",
    )
    print(f"✅ Uploaded! View at: https://huggingface.co/{args.repo_id}", flush=True)

if __name__ == "__main__":
    main()
