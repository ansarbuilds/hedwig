# Hedwig roadmap

Goal: understand every piece of a ChatGPT-style model by building one, the way Karpathy's nanoGPT and nanochat teach it. The end goal is a focused **storytelling** model that runs offline on a Mac and iPhone.

Budget: **~$100 of cloud GPU time** for learning. Everything else runs on the MacBook Air M4.

## Phase 0: Done ✅
- [x] Character-level GPT (4.8M params) trained on Tiny Shakespeare on the Air (`tinygpt.py`)
- [x] Pushed to GitHub (private)

## Phase 1: Better foundations (Air, $0)
- [x] **Reinstall the env**: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- [x] **Add checkpointing + resume** to `tinygpt.py` (save every 250 steps via atomic `.tmp` rename to `ckpt.pt`, auto-resume)
- [x] **Write a BPE tokenizer from scratch** (`tokenizer.py`). Followed Karpathy / `minbpe` style and trained `tokenizer.json` on Tiny Shakespeare (`vocab_size=512`; can scale to ~4k–8k later).
- [x] **Switch Hedwig to the BPE tokenizer** and compare loss and samples vs the character version (best val `3.007` at step 750, overfit to `3.429` by 3000 — Tiny Shakespeare is only `511k` BPE tokens!)
- [x] **Modernize the architecture** one change at a time, measuring each: ~~RoPE~~ ✅ → ~~RMSNorm~~ ✅ → ~~SwiGLU~~ ✅ → ~~bf16 autocast~~ ✅ → ~~`F.scaled_dot_product_attention`~~ ✅
- [x] (Optional) Port to **MLX** and compare speed vs PyTorch MPS (`mlx_bench.py`: practice step `596 ms → 99 ms` at 16-bit, generation `635 → 832 tok/s`, greedy story identical)

## Phase 2: Pick the data (Air, $0)
- [x] Download **TinyStories** from Hugging Face (`roneneldan/TinyStories` — `52.4 MB` train + `10.5 MB` val)
- [x] Download a subset of **Project Gutenberg** (10 public-domain pre-1929 storybooks: *Alice*, *Looking-Glass*, *Wizard of Oz*, *Peter Pan*, *Grimm*, *Andersen*, *Jungle Book*, *Treasure Island*, *Sherlock Holmes*, *Time Machine* — `~3.0 MB` cleaned)
- [x] Write `prepare_data.py`: clean → train 4,096-vocab BPE (`3.87` chars/token) → save as `uint16` `.bin` shards (`14.1M` train tokens in `data/train.bin`, `2.7M` val tokens in `data/val.bin`)
- [x] Train a ~10–30M model (`13.8M` params) on TinyStories + Gutenberg on the Air (`8.501 → 2.080`, then `more_english.py` on 200 MB → `2.010`, then partial full-shelf read → **`1.795`**). Chat manners refreshed to **`1.438`**, taste anchored loss **`1.297`**.

## Phase 2.5: Full ChatGPT pipeline & shipping on Mac (Air, $0 — Done ✅)
Since the cloud GPU run was planned for later, we built and proved the entire post-training & shipping stack locally on the 13.8M model first:
- [x] **Stage 2 — SFT ("Chat Manners", `sft.py`)**: `13,852` instruction → story pairs from `TinyStories-Instruct` with `-100` prompt loss masking (`brain_chat.pt`, `27.8 MB` `bf16`, response loss `1.806 → 1.438` on the `1.795` reader, interactive `sft.py chat`)
- [x] **8-bit & 4-bit Quantization (`quantize.py`)**: manners brain `1.4530 → 1.4531 (8-bit) → 1.4765 (4-bit)`; taste brain `1.4666 → 1.4667 (8-bit) → 1.4897 (4-bit)` (`brain_dpo_int8.pt`, `brain_dpo_int4.pt`)
- [x] **Offline Mac Chat UI (`app.py`)**: Zero-dependency web UI at `http://127.0.0.1:8000` with live Stage 3 DPO / Stage 2 SFT / 8-bit / 4-bit / Stage 1 brain switching
- [x] **Stage 3 — DPO / Preference Tuning (`dpo.py`, $0 on Mac)**: Trained on `13,852` chosen 👍 vs rejected 👎 story pairs (`0% → 92.5%` peak win-rate at step 100; anchored loss `1.422 → 1.297`; caught and fixed DPO "reward hacking" with an SFT fluency anchor; saved `brain_dpo.pt` at `27.8 MB` — **all 3 ChatGPT stages complete locally!**)
- [x] **Refresh `how-hedwig-learns.html` ($0)**: Updated the interactive visual field guide with real Phase 1–5 & Stage 1–3 measurements
- [x] **Prepare 1-command Cloud GPU script (`cloud_train.sh`, $0)**: Turnkey script, rehearsed with `--dry-run`. `--124m` builds 91,286,272 knobs.

## Phase 3: First cloud run (~$15–30 — Planned for Next Month / Dec 2026)
See `docs/CLOUD_GPU_100.md`.
- [x] Local rehearsal of `cloud_train.sh --dry-run` (2 steps in `dry_run/`, 92 s, real brains untouched). Restarts keep checkpoints. `--124m` is 91,286,272 knobs, and the script stops if that edit does not apply.
- [x] Rent 1× GPU, clone this repo, run a short test. **Done 20 Nov 2026:** RunPod, 1× RTX A5000, `$0.28/hr`, about `$0.10`. `bash cloud_train.sh` scored reading `2.100` in 135 s (Mac first run: `2.080` in ~59 min). Screenshots in `docs/images/`. Copies in local `cloud_brains/` only.
- [ ] Run **Karpathy's nanochat** speedrun end to end to see a full pretrain → SFT → chat pipeline at scale. Read every script.
- [ ] Download the result; chat with it locally

## Phase 4: Hedwig-chat storyteller + grammar scale-up (~$5–20)
- [x] Local 13.8M prototype complete (`prepare_data.py` → `tinygpt.py` → `sft.py` → `quantize.py` → `app.py`)
- [x] Scale up on cloud GPU (`bash cloud_train.sh --124m`): **Done 20 Nov 2026** on 1× NVIDIA A40 (`$0.59/hr`, ~66 min, `~$0.65`). Trained **97.6M knobs** (8,192 vocab) on 614 MB TinyStories + FineWeb-Edu + 30,930 SFT story & Grammarly CoEdIT pairs → Stage 1 `2.619`, Stage 2 **`1.336`**, Stage 3 **`1.235`** (`96.7%` peak win-rate), 4-bit **`55.2 MB`**.
- [x] Download cloud weights and compare against the local 13.8M model in `app.py` (97.6M runs at `54 tok/s` offline on M4; kept `"Harry Potter"` across 4 paragraphs vs 14M's `"Timmy & pot of soup"`!)
- [ ] Flagship scale-up (`bash cloud_train.sh --300m`): **286.7M knobs**, 512 context, 16,384 vocab, 1.2 GB stories + FineWeb-Edu, 30,000 CoEdIT pairs (running on RTX 4090!)
- [ ] Optionally expand SFT / DPO data with an **open** teacher model (Qwen / Mistral, Apache-2.0)

## Phase 5: Ship it ($0)
- [x] Quantize to 8-bit / 4-bit (`quantize.py`); manners `1.4530 → 1.4531 → 1.4765`, taste `1.4666 → 1.4667 → 1.4897` (and 97.6M: `55.2 MB` at 4-bit)
- [x] Run on the Mac with a simple chat UI (`app.py` at `http://127.0.0.1:8000`, live 97.6M Cloud + 14M Mac switcher)
- [x] Convert to mobile / Xcode bundle (`export_iphone.py` → Core ML `iphone/Hedwig.mlpackage`, 27.7 MB, **30/30** greedy pieces match PyTorch; Swift tokenizer matches Python; Xcode clicks in `ios/HedwigApp.swift`)
- [x] Upload large cloud weights (>100 MB) to **Hugging Face** (private) as a backup: **Done** (`https://huggingface.co/ansarbuilds/hedwig-97m` via `upload_hf.py` + MIT `LICENSE`)

## Stretch ideas
- [x] **Mixture-of-Experts Hedwig (8 experts, top-2)** (`moe.py` → `26.2M` total / `11.6M` active knobs, upcycled from `brain_dpo.pt`, loss `2.461 → 1.966` in 300 steps)
- [x] **Distillation with teacher soft-labels** (`distill.py` → `3.4M` "🐣 Baby Hedwig" Student distilled from `13.8M` Teacher via `T=2.0` KL divergence, loss `7.602 → 2.521` in 600 steps, `brain_baby.pt` = `6.8 MB`)
- [x] **Interactive "🔬 Brain X-Ray" in Mac Chat UI** (`app.py` — click any generated word to inspect Top-5 candidate probabilities and Top-2 active MoE experts)
- [x] Layer-by-layer loading from SSD (`ssd_runner.py`: `52.9 MB → 18.8 MB` of weights in memory, `108 → 33 tok/s`, stories identical; process RAM only `403 → 372 MB` because the `28 MB` brain is smaller than PyTorch itself)
- [x] A low-resource-language Hedwig (`lang_hedwig.py`: 2.56M French brain on public-domain Dumas + Verne, val `7.776 → 3.541` in 600 steps / 57 s)

## Reading list
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) + `nanoGPT`
- Karpathy, *Let's build the GPT Tokenizer* + `minbpe`
- Karpathy, *Let's reproduce GPT-2 (124M)* + `llm.c`
- `github.com/karpathy/nanochat`
- `github.com/jingyaogong/minimind` (full pipeline incl. SFT/DPO/RL, mostly Chinese)
- Papers: *Attention Is All You Need* (2017), GPT-2 (2019), Chinchilla (2022), TinyStories (2023), LLM in a Flash (Apple, 2023), BitNet b1.58 (2024)
