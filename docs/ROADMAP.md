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
- [ ] (Optional) Port to **MLX** and compare speed vs PyTorch MPS

## Phase 2: Pick the data (Air, $0)
- [x] Download **TinyStories** from Hugging Face (`roneneldan/TinyStories` — `52.4 MB` train + `10.5 MB` val)
- [x] Download a subset of **Project Gutenberg** (10 public-domain pre-1929 storybooks: *Alice*, *Looking-Glass*, *Wizard of Oz*, *Peter Pan*, *Grimm*, *Andersen*, *Jungle Book*, *Treasure Island*, *Sherlock Holmes*, *Time Machine* — `~3.0 MB` cleaned)
- [x] Write `prepare_data.py`: clean → train 4,096-vocab BPE (`3.87` chars/token) → save as `uint16` `.bin` shards (`14.1M` train tokens in `data/train.bin`, `2.7M` val tokens in `data/val.bin`)
- [x] Train a ~10–30M model (`13.8M` params) on TinyStories + Gutenberg on the Air (`8.501 → 2.080` across 3,000 steps with zero overfitting; writes coherent short stories; `brain.pt` saved in `bfloat16` at `27.8 MB`).

## Phase 2.5: Full ChatGPT pipeline & shipping on Mac (Air, $0 — Done ✅ / In Progress)
Since the cloud GPU run is scheduled for **next month (Dec 2026)**, we built and proved the entire post-training & shipping stack locally on the 13.8M model first:
- [x] **Stage 2 — SFT ("Chat Manners", `sft.py`)**: `13,852` instruction → story pairs from `TinyStories-Instruct` with `-100` prompt loss masking (`brain_chat.pt`, `27.8 MB` `bf16`, response loss `2.077 → 1.7544`, interactive `sft.py chat`)
- [x] **8-bit & 4-bit Quantization (`quantize.py`)**: `55.5 MB` fp32 (`1.7544`) → `27.8 MB` bf16 (`1.7544`) → `14.0 MB` int8 (`1.7547`, `brain_chat_int8.pt`) → `7.9 MB` int4 (`1.7718`, `brain_chat_int4.pt`)
- [x] **Offline Mac Chat UI (`app.py`)**: Zero-dependency web UI at `http://127.0.0.1:8000` with live Stage 3 DPO / Stage 2 SFT / 8-bit / 4-bit / Stage 1 brain switching
- [x] **Stage 3 — DPO / Preference Tuning (`dpo.py`, $0 on Mac)**: Trained on `13,852` chosen 👍 vs rejected 👎 story pairs (`0% → 91.7%` preference win-rate in 100 steps; caught and fixed DPO "reward hacking" with an SFT fluency anchor; saved `brain_dpo.pt` at `27.8 MB` — **all 3 ChatGPT stages complete locally!**)
- [x] **Refresh `how-hedwig-learns.html` ($0)**: Updated the interactive visual field guide with real Phase 1–5 & Stage 1–3 measurements
- [x] **Prepare 1-command Cloud GPU script (`cloud_train.sh`, $0)**: Turnkey script (`bash cloud_train.sh` or `bash cloud_train.sh --124m`) ready for next month's cloud GPU run

## Phase 3: First cloud run (~$15–30 — Planned for Next Month / Dec 2026)
See `docs/CLOUD_GPU_100.md`.
- [ ] Rent 1× GPU, clone this repo, run a short test (~$2)
- [ ] Run **Karpathy's nanochat** speedrun end to end to see a full pretrain → SFT → chat pipeline at scale. Read every script.
- [ ] Download the result; chat with it locally

## Phase 4: Hedwig-chat storyteller scale-up (~$50–70 — Planned for Next Month / Dec 2026)
- [x] Local 13.8M prototype complete (`prepare_data.py` → `tinygpt.py` → `sft.py` → `quantize.py` → `app.py`)
- [ ] Scale up on cloud GPU: run `prepare_data.py` on full `1.9 GB` TinyStories + Gutenberg (optionally FineWeb-Edu) and train a ~124M–560M brain
- [ ] Optionally expand SFT / DPO data with an **open** teacher model (Qwen / Mistral, Apache-2.0)
- [ ] Download cloud weights and compare against the local 13.8M model in `app.py`

## Phase 5: Ship it ($0)
- [x] Quantize to 8-bit / 4-bit (`quantize.py`); measure quality vs size (`55.5 MB` fp32 `1.7544` → `27.8 MB` bf16 `1.7544` → `14.0 MB` int8 `1.7547` → `7.9 MB` int4 `1.7718`)
- [x] Run on the Mac with a simple chat UI (`app.py` at `http://127.0.0.1:8000`, live 16-bit / 8-bit / 4-bit switcher)
- [x] Convert to mobile / Xcode bundle (`export_iphone.py` → `iphone/hedwig_mobile.pt`, `iphone/hedwig_int4.pt`, `iphone/HedwigApp.swift`) ready for an Xcode iPhone app (free Apple ID)
- [ ] Upload large cloud weights (>100 MB) to **Hugging Face** (private) as a backup

## Stretch ideas
- [x] **Mixture-of-Experts Hedwig (8 experts, top-2)** (`moe.py` → `26.2M` total / `11.6M` active knobs, upcycled from `brain_dpo.pt`, loss `2.461 → 1.966` in 300 steps)
- [x] **Distillation with teacher soft-labels** (`distill.py` → `3.4M` "🐣 Baby Hedwig" Student distilled from `13.8M` Teacher via `T=2.0` KL divergence, loss `7.602 → 2.521` in 600 steps, `brain_baby.pt` = `6.8 MB`)
- [x] **Interactive "🔬 Brain X-Ray" in Mac Chat UI** (`app.py` — click any generated word to inspect Top-5 candidate probabilities and Top-2 active MoE experts)
- [x] Layer-by-layer loading from SSD (`ssd_runner.py`: `52.9 MB → 18.8 MB` of weights in memory, `108 → 33 tok/s`, stories identical; process RAM only `403 → 372 MB` because the `28 MB` brain is smaller than PyTorch itself)
- [ ] A low-resource-language Hedwig

## Reading list
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) + `nanoGPT`
- Karpathy, *Let's build the GPT Tokenizer* + `minbpe`
- Karpathy, *Let's reproduce GPT-2 (124M)* + `llm.c`
- `github.com/karpathy/nanochat`
- `github.com/jingyaogong/minimind` (full pipeline incl. SFT/DPO/RL, mostly Chinese)
- Papers: *Attention Is All You Need* (2017), GPT-2 (2019), Chinchilla (2022), TinyStories (2023), LLM in a Flash (Apple, 2023), BitNet b1.58 (2024)
