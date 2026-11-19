# AGENTS.md: context for AI coding assistants (Cursor, Antigravity, Claude Code)

## Project
Hedwig is a learning project: build a ChatGPT-style language model from scratch, in the spirit of Karpathy's nanoGPT / nanochat. The owner is learning, so **explain changes in plain language** and keep code readable over clever.

Long-term goal: a focused **storytelling** model that runs offline on a 16 GB MacBook Air M4 and an iPhone.

## Current state
- `tinygpt.py`: Modern BPE-based GPT (4,096 vocab) with RoPE + RMSNorm + SwiGLU + bf16 autocast + `F.scaled_dot_product_attention`, ~13.8M params (`block_size=256, n_embd=384, n_head=6, n_layer=6`), reading `uint16` shards (`data/train.bin`, `data/val.bin`). `.venv/bin/python tinygpt.py train` / `.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "`
- `prepare_data.py`: Downloads TinyStories (`52.4 MB` train + `10.5 MB` val) + 10 public-domain Project Gutenberg novels (`~3.0 MB`), trains 4,096-vocab BPE, and writes `data/train.bin` (`14.1M` tokens) and `data/val.bin` (`2.7M` tokens).
- `tokenizer.py` & `tokenizer.json`: fast word-chunked from-scratch BPE tokenizer (4,096 vocab: 256 bytes + 3,840 merges, `3.87` chars/token)
- `brain.pt`: 13.8M Storytelling weights (`27.8 MB` in `bfloat16`, best val loss `2.080` across 3,000 steps on TinyStories + Gutenberg)
- `data/input.txt`: Tiny Shakespeare
- `logs/DEVLOG.md`: automatic daily commit log via `.github/workflows/devlog.yml`
- `docs/LEARNINGS.md`: concepts + measured results; `docs/ROADMAP.md`: next steps; `docs/CLOUD_GPU_100.md`: $100 cloud plan
- `how-hedwig-learns.html`: visual explainer (static page)

## Environment
- Local: macOS, Apple M4, 16 GB RAM, Python 3.12 venv in `.venv` (not committed). PyTorch uses `mps`.
- Cloud: NVIDIA GPUs (`cuda`). The device is auto-selected in `tinygpt.py`.
- Setup: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Conventions
- Pure PyTorch; avoid heavy frameworks (no HF Trainer / Lightning) so every step stays visible.
- Change one thing at a time and record the loss before/after in `docs/LEARNINGS.md`.
- Keep configs as plain variables at the top of the script.
- Any long run must save checkpoints and be resumable.
- Don't commit files >100 MB (GitHub limit). Large weights go to Hugging Face. Datasets go in `data/` and are gitignored except the small Shakespeare file.
- Training data must be legally usable: public domain (Project Gutenberg), permissive datasets (TinyStories, FineWeb-Edu), or permissively licensed open teacher models (Qwen/Mistral Apache-2.0, DeepSeek MIT). Never train on outputs of OpenAI/Anthropic/Google models.

## Next task (see ROADMAP Phase 3)
1. Phase 3 ($15–30 cloud GPU run, see `docs/CLOUD_GPU_100.md`): rent 1× GPU for a smoke test, or run Karpathy's `nanochat` speedrun end to end (pretrain → SFT → chat).
