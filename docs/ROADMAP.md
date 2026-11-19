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

## Phase 3: First cloud run (~$15–30)
See `docs/CLOUD_GPU_100.md`.
- [ ] Rent 1× GPU, clone this repo, run a short test (~$2)
- [ ] Run **Karpathy's nanochat** speedrun end to end to see a full pretrain → SFT → chat pipeline. Read every script.
- [ ] Download the result; chat with it locally

## Phase 4: Hedwig-chat storyteller (Local prototype ✅, ~$50–70 cloud scale-up next)
- [x] Build an SFT set of "instruction → story" pairs (`13,852` train + `2,327` val pairs from `TinyStories-Instruct` in `sft.py`, with `-100` prompt loss masking)
- [x] Train local 13.8M SFT chat model (`brain_chat.pt`, response loss `2.077 → 1.766` in 750 steps) with interactive `sft.py chat` / `sft.py talk`
- [ ] Scale up on cloud GPU (~124M–560M nanochat speedrun on TinyStories + Gutenberg + optionally FineWeb-Edu) and compare with the local 13.8M model

## Phase 5: Ship it ($0)
- [ ] Quantize to 8-bit / 4-bit; measure quality vs size
- [ ] Run on the Mac with a simple chat UI
- [ ] Convert to Core ML and build a tiny iPhone app (Xcode, free Apple ID)
- [ ] Upload the weights to **Hugging Face** (private) as a backup; files >100 MB don't belong on GitHub

## Stretch ideas
- Mixture-of-Experts Hedwig (8 experts, top-2)
- Layer-by-layer loading from SSD (AirLLM / "LLM in a Flash" style)
- Distillation with teacher soft-labels
- A low-resource-language Hedwig

## Reading list
- Karpathy, *Let's build GPT: from scratch, in code, spelled out* (YouTube) + `nanoGPT`
- Karpathy, *Let's build the GPT Tokenizer* + `minbpe`
- Karpathy, *Let's reproduce GPT-2 (124M)* + `llm.c`
- `github.com/karpathy/nanochat`
- `github.com/jingyaogong/minimind` (full pipeline incl. SFT/DPO/RL, mostly Chinese)
- Papers: *Attention Is All You Need* (2017), GPT-2 (2019), Chinchilla (2022), TinyStories (2023), LLM in a Flash (Apple, 2023), BitNet b1.58 (2024)
