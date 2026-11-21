# AGENTS.md: context for AI coding assistants (Cursor, Antigravity, Claude Code)

## Project
Hedwig is a learning project: build a ChatGPT-style language model from scratch, in the spirit of Karpathy's nanoGPT / nanochat. The owner is learning, so **explain changes in plain language** and keep code readable over clever.

Long-term goal: a focused **storytelling** model that runs offline on a 16 GB MacBook Air M4 and an iPhone.

## Current state
- `tinygpt.py`: Stage 1 modern BPE-based GPT (4,096 vocab) with RoPE + RMSNorm + SwiGLU + bf16 autocast + `F.scaled_dot_product_attention`, ~13.8M params. `brain.pt` val loss **`1.795`** on the original `data/val.bin` (50 MB run was `2.080`, 200 MB run was `2.010`, then a partial read of the full 1.92 GB TinyStories file reached `1.795` at step 12,250 and was stopped on purpose). `.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "`
- `more_english.py`: extra English reading. `python more_english.py` is the finished 200 MB run. `python more_english.py full` reads the whole TinyStories file (`data/train_full.bin`, 489M pieces, 59,682 steps, ~20 h). That full run was **stopped around step 12,500**. Bookmark is `ckpt_full.pt` (local, gitignored). Do not restart it unless the owner asks. Chat SFT, DPO, and quantization were re-run on top of the 1.795 reader: `brain_chat.pt` manners **`1.438`** (was `1.723`), `brain_dpo.pt` anchored loss **`1.297`** (was `1.440`; peak win-rate **92.5%** / saved **88.8%**).
- `sft.py`: Stage 2 Supervised Fine-Tuning (`13,852` instruction → story pairs from `TinyStories-Instruct` with `-100` prompt loss masking) + interactive chat. `.venv/bin/python sft.py train` / `.venv/bin/python sft.py talk "Write a story..."` / `.venv/bin/python sft.py chat`
- `dpo.py`: Stage 3 Direct Preference Optimization (`13,852` chosen 👍 vs rejected 👎 pairs with fluency anchor; `0% → 92.5%` peak win-rate in 100 steps). `.venv/bin/python dpo.py train` / `.venv/bin/python dpo.py talk "Write a short story about dragon and castle."`
- `moe.py`: Mixture-of-Experts Hedwig (`26.2M` total / `11.6M` active params, 8 experts top-2, upcycled from `brain_dpo.pt`, loss `2.461 → 1.966`).
- `distill.py`: Teacher → Student soft-label knowledge distillation (`13.8M` Teacher → `3.4M` Baby Hedwig `brain_baby.pt` at `6.8 MB`, loss `7.602 → 2.521`).
- `quantize.py`: From-scratch 8-bit and packed 4-bit. Stage 2 chat: `14.0 MB` loss `1.4531`, `7.9 MB` loss `1.4765` (16-bit `1.4530`). Stage 3 taste: `brain_dpo_int8.pt` loss `1.4667`, `brain_dpo_int4.pt` loss `1.4897` (16-bit taste loss `1.4666`).
- `app.py`: Zero-dependency offline Mac Chat UI (`http://127.0.0.1:8000`) with 9 brains (including Stage 3 at 16-bit, 8-bit, and 4-bit), KV-cached word streaming, and clickable **🔬 Brain X-Ray** (Top-5 token probabilities + Top-2 active MoE experts).
- `prepare_data.py`: Downloads TinyStories (`52.4 MB` train + `10.5 MB` val) + 10 public-domain Project Gutenberg novels (`~3.0 MB`), trains 4,096-vocab BPE, and writes `data/train.bin` (`14.1M` tokens) and `data/val.bin` (`2.7M` tokens).
- `tokenizer.py` & `tokenizer.json`: fast word-chunked from-scratch BPE tokenizer (4,096 vocab: 256 bytes + 3,840 merges, `3.87` chars/token)
- `brain.pt`, `brain_chat.pt`, `brain_dpo.pt`, `brain_chat_int8.pt`, `brain_chat_int4.pt`: 13.8M Stage 1 (`27.8 MB`), Stage 2 SFT (`27.8 MB`), Stage 3 DPO (`27.8 MB`), 8-bit (`14.0 MB`), and 4-bit (`7.9 MB`) weights
- `data/input.txt`: Tiny Shakespeare
- `logs/DEVLOG.md`: automatic daily commit log via `.github/workflows/devlog.yml`
- `docs/LEARNINGS.md`: concepts + measured results; `docs/ROADMAP.md`: next steps; `docs/CLOUD_GPU_100.md`: $100 cloud plan
- `how-hedwig-learns.html`: interactive visual explainer updated with all Phase 1–5 & Stage 1–3 measured curves (night-sky redesign: starfield, owl, sticky lesson menu, scroll animations, and both films embedded)
- `web/`: in-browser demo (plain JavaScript, no server), live at `huggingface.co/spaces/ansarbuilds/hedwig-demo`. Same night-sky look; plays `web/media/hedwig-film.mp4` (720p copy of the film). Redeploy with `.venv/bin/python deploy_space.py ansarbuilds/hedwig-demo` (publishes publicly: ask the owner first).
- `film/`: 60-second Remotion launch film. Scenes in `film/src/HedwigFilm.tsx` (10 scenes, Pip the owl), shared set pieces in `film/src/filmKit.tsx`, camera/grade/grain/letterbox in `film/src/cinemaPost.tsx`. `cd film && npm run render` writes `film/out/hedwig-demo.mp4` and re-encodes it under GitHub's 100 MB limit. The old `film/src/HedwigDemo.tsx` is unused.
- `movie/`: *Pip and the Lost Key*, an 84-second storybook film whose story was written word for word by the 286.7M brain (`movie/make_movie.py`). A filter-only polish pass made it look worse, so it was left as is.
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

## How to talk to the owner
- Explain in plain words, with a concrete example from Hedwig's own stories (Lily, a key, an owl, TinyStories). The owner got lost when explanations used a kitchen metaphor. Do not use that metaphor.
- When comparing scale, say it directly: Hedwig is **14 million** knobs. GPT-2 small (2019) is about **124 million**. GPT-3 is about **175 billion**. ChatGPT has read a huge slice of the web. Hedwig has read children's stories and 10 public-domain novels.
- The iPhone export exists, but the owner said to leave the phone and focus on English.
- Do not overwrite `brain.pt`, `brain_chat.pt`, or `brain_dpo.pt` with anything in `cloud_brains/`. Those rented-computer files are a fresh 50 MB training (val **2.100**), which is a worse quiz score than the Mac reader (**1.795**).

## Next tasks (see `docs/ROADMAP.md`)
### Mac experiments done
SSD loading, MLX speed test, French Hedwig, Core ML iPhone export, 200 MB extra English, and a partial full-shelf read (stopped on purpose at val **1.795**).

### Rented GPU
1. ~~Smoke test~~ ✅ 20 Nov 2026, RunPod pod `gleaming_olive_hippopotamus`, 1× RTX A5000, **$0.28/hr**, about **$0.10** spent, balance left **$14.90**. Pod was stopped (`$0.00/hr`). `bash cloud_train.sh` on the 50 MB pile: pretrain **8.470 → 2.100 in 135 s**, SFT **2.073 → 1.758 in 81 s**, DPO **0% → 91.7% in 23 s**. Copies are local only, in `cloud_brains/` (gitignored). Screenshots: `docs/images/`.
2. ~~Scale-up (`--124m`)~~ ✅ 20 Nov 2026, RunPod 1× NVIDIA A40 (**$0.59/hr**, ~66 min, **~$0.65** spent). Trained **97,581,824 knobs** (12 layers, 768 wide, 12 heads, **8,192** vocab) on 614 MB (`366.7 MB` TinyStories + `262.1 MB` FineWeb-Edu + 10 novels), plus `30,930` SFT story + Grammarly CoEdIT pairs: Stage 1 **2.619**, Stage 2 **`1.336`**, Stage 3 **`1.235`** (**96.7%** peak / **95.0%** saved win-rate), 4-bit **55.2 MB** (`1.369` chat / `1.412` taste). Copies + 8k `tokenizer.json` live in `cloud_brains_97m/` (gitignored), backed up to Hugging Face (`ansarbuilds/hedwig-97m`), and wired into `app.py`.
3. ~~Flagship (`--300m`)~~ ✅ 20 Nov 2026, RunPod 1× NVIDIA RTX 4090 (**$0.89/hr**, ~115 min, **~$1.70** spent, ~**$12.50** balance left). Trained **286,663,680 knobs** (20 layers, 1024 wide, 16 heads, **512** context, **16,384** vocab) on 1.22 GB (`523.9 MB` TinyStories + `734.0 MB` FineWeb-Edu + 10 novels, ~205M tokens seen ≈ 0.7 tokens/knob), plus `71,906` SFT pairs (`30,000` CoEdIT) and `52,798` DPO pairs: Stage 1 **2.998**, Stage 2 **`1.341`**, Stage 3 **`1.200`** (**98.3%** win-rate), 4-bit **161.9 MB** (`1.360` chat / `1.425` taste). Copies + 16k `tokenizer.json` live in `cloud_brains_300m/` (gitignored), backed up to Hugging Face (`ansarbuilds/hedwig-300m`), wired into `app.py`, and compared side-by-side against 14M & 97.6M in `docs/STORY_SHOWDOWN.md` (`compare_brains.py`). Don't compare raw loss across 4k/8k/16k vocabs; compare via stories.
