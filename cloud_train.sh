#!/usr/bin/env bash
# ==============================================================================
# Hedwig 1-Command Cloud GPU Pipeline (for Phase 3 & 4 next month / Dec 2026)
#
# Runs the entire ChatGPT-style storytelling pipeline end-to-end on a rented GPU:
#   1. Sets up Python .venv & checks GPU (nvidia-smi)
#   2. Downloads TinyStories + 10 Project Gutenberg novels & builds uint16 .bin shards
#   3. Stage 1: Pretrains Hedwig (tinygpt.py -> brain.pt)
#   4. Stage 2: Supervised Fine-Tuning for chat manners (sft.py -> brain_chat.pt)
#   5. Stage 3: Anchored DPO preference tuning (dpo.py -> brain_dpo.pt)
#   6. Phase 5: Quantizes to 8-bit & 4-bit (quantize.py -> brain_chat_int8/int4.pt)
#
# Usage on rented GPU:
#   bash cloud_train.sh           # fast 13.8M smoke test (~10 min on RTX 4090 / H100)
#   bash cloud_train.sh --124m    # scale up to ~115M GPT-2 Small size on 500 MB stories
# ==============================================================================
set -euo pipefail

echo "🦉 Hedwig Cloud GPU Pipeline starting..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ "${1:-}" = "--124m" ]; then
  echo "🚀 Scaling config to ~115M knobs (n_embd=768, n_head=12, n_layer=12) and 500 MB TinyStories..."
  .venv/bin/python -c '
for path, old, new in [
    ("prepare_data.py", "tinystories_train_mb = 50", "tinystories_train_mb = 500"),
    ("tinygpt.py", "n_embd     = 384", "n_embd     = 768"),
    ("tinygpt.py", "n_head     = 6",   "n_head     = 12"),
    ("tinygpt.py", "n_layer    = 6",   "n_layer    = 12"),
    ("tinygpt.py", "steps      = 3000","steps      = 6000"),
]:
    open(path, "w").write(open(path).read().replace(old, new, 1))
'
fi

rm -f ckpt.pt ckpt_sft.pt ckpt_dpo.pt data/sft_cache.pt data/dpo_cache.pt
.venv/bin/python -u prepare_data.py
.venv/bin/python -u tinygpt.py train
.venv/bin/python -u sft.py train
.venv/bin/python -u dpo.py train
.venv/bin/python -u quantize.py

echo "✅ All done! Copy your trained brains back to your Mac BEFORE stopping the pod:"
echo "   scp -P <port> root@<host>:~/hedwig/brain*.pt ./"
