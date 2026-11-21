"""
Publish Hedwig's in-browser demo as a free static Hugging Face Space.

Static Spaces are free: Hugging Face only hands out files, and Hedwig itself runs in the
visitor's browser (web/hedwig.js). Uploads everything in web/ except the test file.

Steps:
    .venv/bin/python export_web.py           # 14M brain -> web/brains/14m/
    .venv/bin/python export_web.py 97m       # 97.6M brain -> web/brains/97m/
    node web/test_web.mjs 14m                # check JavaScript matches PyTorch
    .venv/bin/python deploy_space.py ansarbuilds/hedwig-demo
"""
import argparse, os, sys
from huggingface_hub import HfApi

def main():
    p = argparse.ArgumentParser(description="Upload web/ to a static Hugging Face Space.")
    p.add_argument("space_id", help="e.g. ansarbuilds/hedwig-demo")
    args = p.parse_args()
    if not os.path.exists("web/brains/14m/model.bin"):
        sys.exit("Run .venv/bin/python export_web.py first.")
    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="static", exist_ok=True)
    print(f"🦉 Uploading web/ to https://huggingface.co/spaces/{args.space_id} ...", flush=True)
    api.upload_folder(folder_path="web", repo_id=args.space_id, repo_type="space",
                      ignore_patterns=["test_web.mjs", "brains/*/check.json", ".*"],
                      commit_message="Deploy Hedwig in-browser demo")
    print(f"✅ Live in about a minute: https://huggingface.co/spaces/{args.space_id}")

if __name__ == "__main__":
    main()
