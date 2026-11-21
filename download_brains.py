"""
Download Hedwig's cloud-trained brains from Hugging Face so anyone who clones
this GitHub repo can run the 97.6M or 286.7M brains offline in app.py.

By default downloads the compact 4-bit DPO brain + tokenizer (~55 MB for 97m,
~162 MB for 300m). Pass --full to also download the 16-bit bfloat16 weights.

Run:
    .venv/bin/python download_brains.py 97m          # 97.6M 4-bit (55 MB) + tokenizer
    .venv/bin/python download_brains.py 300m         # 286.7M 4-bit (162 MB) + tokenizer
    .venv/bin/python download_brains.py 300m --full  # also grabs 16-bit weights (~575 MB each)
"""
import argparse, os, sys
from huggingface_hub import hf_hub_download

TARGETS = {
    "97m": {
        "repo": "ansarbuilds/hedwig-97m",
        "dir":  "cloud_brains_97m",
        "lite": ["tokenizer.json", "brain_dpo_int4.pt"],
        "full": ["tokenizer.json", "brain_dpo_int4.pt", "brain_chat_int4.pt", "brain_dpo.pt", "brain_chat.pt"],
    },
    "300m": {
        "repo": "ansarbuilds/hedwig-300m",
        "dir":  "cloud_brains_300m",
        "lite": ["tokenizer.json", "brain_dpo_int4.pt"],
        "full": ["tokenizer.json", "brain_dpo_int4.pt", "brain_chat_int4.pt", "brain_dpo.pt", "brain_chat.pt"],
    },
}

def main():
    p = argparse.ArgumentParser(description="Download Hedwig 97.6M or 286.7M weights from Hugging Face.")
    p.add_argument("size", choices=["97m", "300m"], help="Which cloud brain to download (97m or 300m)")
    p.add_argument("--full", action="store_true", help="Also download 16-bit weights (larger)")
    args = p.parse_args()

    cfg = TARGETS[args.size]
    os.makedirs(cfg["dir"], exist_ok=True)
    files = cfg["full"] if args.full else cfg["lite"]

    print(f"🦉 Downloading Hedwig {args.size} from https://huggingface.co/{cfg['repo']} -> {cfg['dir']}/", flush=True)
    for fname in files:
        dest = os.path.join(cfg["dir"], fname)
        if os.path.exists(dest):
            print(f"  ✓ already on disk: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)
            continue
        print(f"  ↓ fetching {fname} ...", flush=True)
        hf_hub_download(repo_id=cfg["repo"], filename=fname, local_dir=cfg["dir"])
        print(f"  ✓ saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)

    print(f"\n✅ Ready! Start the offline chat UI with:\n   .venv/bin/python app.py", flush=True)

if __name__ == "__main__":
    main()
