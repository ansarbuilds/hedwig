# 🦉 Hedwig

A tiny GPT-style language model built from scratch to learn how LLMs work.
- **v1:** 4.8M-parameter character-level Transformer trained on Tiny Shakespeare in ~16 minutes on a MacBook Air M4 (final loss `1.481`).
- **v2 (Phase 2 Storyteller):** 13.8M-parameter modern Transformer (4,096-vocab BPE, RoPE, RMSNorm, SwiGLU, `bf16`, SDPA) trained 3,000 steps on **TinyStories + 10 Project Gutenberg novels** (`14.1M` tokens, val loss **`2.080`**, `brain.pt` = `27.8 MB` in `bfloat16`):

```
Once upon a time, a little owl Zoey lived in a big tree. The little owl
was very brave... Then, his friends saw him struggling. They asked him
what was wrong... Zoey was so happy. He thanked his friends for the
kindness and for helping him. From that day on, Zoey never had to worry again.
```

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                                              # open offline Mac Chat UI (16-bit / 8-bit / 4-bit)
.venv/bin/python sft.py chat                                         # interactive terminal chat (`brain_chat.pt`)
.venv/bin/python sft.py talk "Tell me a story where a little girl named Lily finds a shiny key in the garden."
.venv/bin/python quantize.py                                         # quantize to 8-bit (14 MB) & 4-bit (7.9 MB)
.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "   # base autocomplete model (`brain.pt`)
.venv/bin/python prepare_data.py                                     # download stories & build .bin shards
.venv/bin/python tinygpt.py train                                    # Stage 1 pretraining
.venv/bin/python sft.py train                                        # Stage 2 SFT ("chat manners")
```

## Files

| Path | What |
|---|---|
| `tinygpt.py` | Stage 1: ~13.8M-param modern Transformer (RoPE, RMSNorm, SwiGLU, bf16, SDPA), pretraining & autocomplete |
| `sft.py` | Stage 2: Supervised Fine-Tuning (prompt loss masking) & interactive story chat (`talk` / `chat`) |
| `quantize.py` | From-scratch 8-bit (`int8`) and 4-bit (packed `int4`) quantizer + quality benchmark |
| `app.py` | Zero-dependency offline Mac Chat UI (`http://127.0.0.1:8000`) |
| `prepare_data.py` | Downloads TinyStories + 10 Project Gutenberg books, trains 4,096-vocab BPE, writes `uint16` `.bin` shards |
| `tokenizer.py` | From-scratch word-chunked BPE tokenizer (`minbpe` / GPT-2 style) |
| `tokenizer.json` | Learned BPE merges (4,096 vocab on TinyStories + Project Gutenberg) |
| `brain.pt` | Stage 1 pretrained storyteller weights (`27.8 MB` in `bfloat16`, val loss `2.080`) |
| `brain_chat.pt` | Stage 2 SFT chat weights (`27.8 MB` in `bfloat16`, response loss `1.7544`) |
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
A focused storytelling model that runs offline on a Mac and iPhone. See [docs/ROADMAP.md](docs/ROADMAP.md).
