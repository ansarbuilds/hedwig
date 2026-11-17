# AGENTS.md: context for AI coding assistants (Cursor, Antigravity, Claude Code)

## Project
Hedwig is a learning project: build a ChatGPT-style language model from scratch, in the spirit of Karpathy's nanoGPT / nanochat. The owner is learning, so **explain changes in plain language** and keep code readable over clever.

Long-term goal: a focused **storytelling** model that runs offline on a 16 GB MacBook Air M4 and an iPhone.

## Current state
- `tinygpt.py`: BPE-based GPT (512 vocab), 5.0M params, ~137 lines, with checkpoint save/resume every 250 steps (`ckpt.pt`) and token cache (`data/tokens_512.pt`). `.venv/bin/python tinygpt.py train` / `.venv/bin/python tinygpt.py talk "ROMEO:"`
- `tokenizer.py` & `tokenizer.json`: from-scratch BPE tokenizer (512 vocab: 256 bytes + 256 merges), wired into `tinygpt.py`
- `brain.pt`: BPE-trained weights (best val `3.007` at step 750, final `3.429` at step 3000 on Tiny Shakespeare)
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

## Next task (see ROADMAP Phase 1)
1. Save best-val `brain.pt` (to avoid saving overfit step-3000 weights) and modernize the architecture one change at a time (RoPE → RMSNorm → SwiGLU → bf16 autocast → `F.scaled_dot_product_attention`), then move to Phase 2 (TinyStories + Gutenberg) so the dataset is large enough not to overfit.
