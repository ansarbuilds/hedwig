# 🦉 Hedwig

A tiny GPT-style language model built from scratch in pure PyTorch to learn how every piece of an LLM works.

## Progress so far (all trained on a MacBook Air M4 for $0)
- **v1 (Phase 0):** 4.8M-parameter character-level Transformer on Tiny Shakespeare (~16 min, loss `1.481`).
- **v2 (Phase 1 & 2 — Stage 1 Storyteller):** 13.8M-parameter modern Transformer (4,096-vocab BPE, RoPE, RMSNorm, SwiGLU, `bf16`, SDPA) trained 3,000 steps on **TinyStories + 10 Project Gutenberg novels** (`14.1M` tokens, val loss **`2.080`**, `brain.pt` = `27.8 MB`).
- **v3 (Phase 2.5 — Stage 2 SFT + Stage 3 DPO + 8-bit/4-bit Quantization + Mac UI):**
  - **Stage 2 SFT (`sft.py`):** Fine-tuned on `13,852` instruction → story pairs (response loss **`1.7544`**, `brain_chat.pt` = `27.8 MB`).
  - **Stage 3 DPO (`dpo.py`):** Direct Preference Optimization on `13,852` chosen 👍 vs rejected 👎 pairs with a fluency anchor (**`91.7%` preference win-rate**, `brain_dpo.pt` = `27.8 MB`).
  - **Quantization & UI (`quantize.py`, `app.py`):** Quantized from scratch to **8-bit (`14.0 MB`, loss `1.7547`)** and **4-bit (`7.9 MB`, loss `1.7718`)**, served in an offline Mac web UI (`app.py`):

```
🧑 You: Tell me a story where a little girl named Lily finds a shiny key in the garden.
🦉 Hedwig (4-bit, 7.9 MB):
Once upon a time, there was a little girl named Lily. She loved to play outside
and explore. One day, she found a shiny metal key in her yard... When they got
home, Lily put the key in her pocket. Timmy was so happy and hugged her tight.
He said, "Thank you, Lily!" They went back home... dreaming of their next adventure.
```

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                                              # open offline Mac Chat UI (DPO / SFT / 8-bit / 4-bit)
.venv/bin/python dpo.py talk "Write a short story about dragon and castle."
.venv/bin/python sft.py chat                                         # interactive terminal chat (`brain_chat.pt`)
.venv/bin/python sft.py talk "Tell me a story where a little girl named Lily finds a shiny key in the garden."
.venv/bin/python quantize.py                                         # quantize to 8-bit (14 MB) & 4-bit (7.9 MB)
.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "   # base autocomplete model (`brain.pt`)
.venv/bin/python prepare_data.py                                     # download stories & build .bin shards
.venv/bin/python tinygpt.py train                                    # Stage 1 pretraining
.venv/bin/python sft.py train                                        # Stage 2 SFT ("chat manners")
.venv/bin/python dpo.py train                                        # Stage 3 DPO ("preference & taste")
.venv/bin/python export_iphone.py                                    # export mobile model + SwiftUI template (`iphone/`)
bash cloud_train.sh                                                  # 1-command end-to-end pipeline (or `bash cloud_train.sh --124m` on cloud GPU)
```

## Files

| Path | What |
|---|---|
| `tinygpt.py` | Stage 1: ~13.8M-param modern Transformer (RoPE, RMSNorm, SwiGLU, bf16, SDPA), pretraining & autocomplete |
| `sft.py` | Stage 2: Supervised Fine-Tuning (prompt loss masking) & interactive story chat (`talk` / `chat`) |
| `dpo.py` | Stage 3: Direct Preference Optimization (👍 chosen vs 👎 rejected + fluency anchor) |
| `quantize.py` | From-scratch 8-bit (`int8`) and 4-bit (packed `int4`) quantizer + quality benchmark |
| `app.py` | Zero-dependency offline Mac Chat UI (`http://127.0.0.1:8000`) |
| `export_iphone.py` | Exports TorchScript mobile bundle + 4-bit weights + SwiftUI Xcode template (`iphone/`) |
| `cloud_train.sh` | 1-command Cloud GPU training script (`--124m` scales to GPT-2 Small size for Dec 2026) |
| `prepare_data.py` | Downloads TinyStories + 10 Project Gutenberg books, trains 4,096-vocab BPE, writes `uint16` `.bin` shards |
| `tokenizer.py` | From-scratch word-chunked BPE tokenizer (`minbpe` / GPT-2 style) |
| `tokenizer.json` | Learned BPE merges (4,096 vocab on TinyStories + Project Gutenberg) |
| `brain.pt` | Stage 1 pretrained storyteller weights (`27.8 MB` in `bfloat16`, val loss `2.080`) |
| `brain_chat.pt` | Stage 2 SFT chat weights (`27.8 MB` in `bfloat16`, response loss `1.7544`) |
| `brain_dpo.pt` | Stage 3 DPO preference-tuned weights (`27.8 MB` in `bfloat16`, `91.7%` preference win-rate) |
| `brain_chat_int8.pt` | 8-bit quantized chat weights (`14.0 MB`, `3.96×` smaller, response loss `1.7547`) |
| `brain_chat_int4.pt` | 4-bit quantized chat weights (`7.9 MB`, `7.05×` smaller, response loss `1.7718`) |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB) |
| `how-hedwig-learns.html` | Visual explainer: knobs, training, model sizes |
| `docs/LEARNINGS.md` | Concepts and measured results |
| `docs/ROADMAP.md` | What to build next |
| `docs/CLOUD_GPU_100.md` | How to spend $100 of cloud GPU time |
| `logs/DEVLOG.md` | Automatic daily dev log (updated on push) |
| `AGENTS.md` | Context for AI coding assistants |

## Where it's going
- **Next on Mac ($0):** Stage 3 DPO preference tuning (`dpo.py`), updating `how-hedwig-learns.html`, a 1-command cloud training script, and Core ML iPhone export.
- **Next Month (Dec 2026, ~$15–100 cloud GPU):** Scale up to ~124M–560M parameters on the full `1.9 GB` TinyStories + Gutenberg dataset and run Karpathy's `nanochat` speedrun. See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/CLOUD_GPU_100.md](docs/CLOUD_GPU_100.md).
