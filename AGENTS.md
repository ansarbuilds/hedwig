# AGENTS.md: context for AI coding assistants (Cursor, Antigravity, Claude Code)

## Project
Hedwig is a learning project: build a ChatGPT-style language model from scratch, in the spirit of Karpathy's nanoGPT / nanochat. The owner is learning, so **explain changes in plain language** and keep code readable over clever.

Long-term goal: a focused **storytelling** model that runs offline on a 16 GB MacBook Air M4 and an iPhone.

## Current state
- `tinygpt.py`: Stage 1 modern BPE-based GPT (4,096 vocab) with RoPE + RMSNorm + SwiGLU + bf16 autocast + `F.scaled_dot_product_attention`, ~13.8M params. After `more_english.py` (200 MB TinyStories, same dictionary), val loss **`2.010`** on the original test pages (was `2.080`). `.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "`
- `more_english.py`: extra English reading from `brain.pt` on `data/train_more.bin` (54M pieces), then refreshes SFT and DPO. Chat score **`1.723`** (was `1.754`). `.venv/bin/python more_english.py`
- `sft.py`: Stage 2 Supervised Fine-Tuning (`13,852` instruction → story pairs from `TinyStories-Instruct` with `-100` prompt loss masking) + interactive chat. `.venv/bin/python sft.py train` / `.venv/bin/python sft.py talk "Write a story..."` / `.venv/bin/python sft.py chat`
- `dpo.py`: Stage 3 Direct Preference Optimization (`13,852` chosen 👍 vs rejected 👎 pairs with fluency anchor; `0% → 91.7%` win-rate in 100 steps). `.venv/bin/python dpo.py train` / `.venv/bin/python dpo.py talk "Write a short story about dragon and castle."`
- `moe.py`: Mixture-of-Experts Hedwig (`26.2M` total / `11.6M` active params, 8 experts top-2, upcycled from `brain_dpo.pt`, loss `2.461 → 1.966`).
- `distill.py`: Teacher → Student soft-label knowledge distillation (`13.8M` Teacher → `3.4M` Baby Hedwig `brain_baby.pt` at `6.8 MB`, loss `7.602 → 2.521`).
- `quantize.py`: From-scratch 8-bit and packed 4-bit. Stage 2 chat: `14.0 MB` loss `1.7547`, `7.9 MB` loss `1.7718`. Stage 3 taste: `brain_dpo_int8.pt` loss `1.7639`, `brain_dpo_int4.pt` loss `1.7827` (16-bit taste loss `1.7637`).
- `app.py`: Zero-dependency offline Mac Chat UI (`http://127.0.0.1:8000`) with 9 brains (including Stage 3 at 16-bit, 8-bit, and 4-bit), KV-cached word streaming, and clickable **🔬 Brain X-Ray** (Top-5 token probabilities + Top-2 active MoE experts).
- `prepare_data.py`: Downloads TinyStories (`52.4 MB` train + `10.5 MB` val) + 10 public-domain Project Gutenberg novels (`~3.0 MB`), trains 4,096-vocab BPE, and writes `data/train.bin` (`14.1M` tokens) and `data/val.bin` (`2.7M` tokens).
- `tokenizer.py` & `tokenizer.json`: fast word-chunked from-scratch BPE tokenizer (4,096 vocab: 256 bytes + 3,840 merges, `3.87` chars/token)
- `brain.pt`, `brain_chat.pt`, `brain_dpo.pt`, `brain_chat_int8.pt`, `brain_chat_int4.pt`: 13.8M Stage 1 (`27.8 MB`), Stage 2 SFT (`27.8 MB`), Stage 3 DPO (`27.8 MB`), 8-bit (`14.0 MB`), and 4-bit (`7.9 MB`) weights
- `data/input.txt`: Tiny Shakespeare
- `logs/DEVLOG.md`: automatic daily commit log via `.github/workflows/devlog.yml`
- `docs/LEARNINGS.md`: concepts + measured results; `docs/ROADMAP.md`: next steps; `docs/CLOUD_GPU_100.md`: $100 cloud plan
- `how-hedwig-learns.html`: interactive visual explainer updated with all Phase 1–5 & Stage 1–3 measured curves
- `cloud_train.sh`: 1-command cloud GPU script. `bash cloud_train.sh --dry-run` rehearses 2 steps in `dry_run/`. On a GPU: `bash cloud_train.sh` (13.8M smoke) or `bash cloud_train.sh --124m` (91,286,272 knobs, 20,000 steps). Restarts resume; `--fresh` wipes saves. Private clone steps are in `docs/CLOUD_GPU_100.md`.
- `export_iphone.py`: builds a Core ML brain (`iphone/Hedwig.mlpackage`, 27.7 MB, 64 fixed seats) that matched PyTorch on 30/30 greedy pieces, plus `ios/HedwigApp.swift` (real sampling loop) and `ios/HedwigTokenizer.swift` (same piece numbers as Python). `.venv/bin/python export_iphone.py`
- `lang_hedwig.py`: French Hedwig, separate from the English brain. 2.56M knobs on public-domain *Mousquetaires* + *Vingt mille lieues*, val `7.776 → 3.541` in 600 steps. `.venv/bin/python lang_hedwig.py train` / `talk "Il était une fois un petit hibou "`
- `ssd_runner.py`: "LLM in a Flash" layer streaming. Splits `brain_dpo.pt` into `ssd_layers/` and loads one block at a time. Measured on M4: weights in memory `52.9 MB → 18.8 MB`, speed `108 → 33 tok/s`, peak process RAM `403 → 372 MB` (stories identical). `.venv/bin/python ssd_runner.py`
- `mlx_bench.py`: same 13.8M brain in Apple MLX vs PyTorch `mps`. Practice step `596 → 99 ms` (16-bit), generation `635 → 832 tok/s`, greedy story identical. `.venv/bin/python mlx_bench.py`

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

## Next tasks (see `docs/ROADMAP.md`)
### All core $0 MacBook Air milestones are complete ✅!
### Optional $0 Mac experiments (anytime before Dec 2026):
1. ~~Layer-by-layer SSD loading~~ ✅ (`ssd_runner.py`: `52.9 → 18.8 MB` weights in RAM, `108 → 33 tok/s`).
2. ~~Apple MLX vs PyTorch MPS~~ ✅ (`mlx_bench.py`: practice `596 → 99 ms`, generation `635 → 832 tok/s`, same story).
3. ~~Low-resource / second-language Hedwig~~ ✅ (`lang_hedwig.py`, French, val `7.776 → 3.541`).

### Next month (Dec 2026 — $15–100 cloud GPU budget, see `docs/CLOUD_GPU_100.md`):
1. **Phase 3 & 4 Cloud Run**: Rent 1× GPU (~$2 smoke test with `bash cloud_train.sh`), then run `bash cloud_train.sh --124m` (91,286,272 knobs, 500 MB of stories). Local `--dry-run` already passed.
