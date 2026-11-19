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
.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "   # use included brain.pt
.venv/bin/python prepare_data.py                                     # download stories & build .bin shards
.venv/bin/python tinygpt.py train                                    # train or resume from ckpt.pt
```

## Files

| Path | What |
|---|---|
| `tinygpt.py` | ~13.8M-param modern Transformer (RoPE, RMSNorm, SwiGLU, bf16, SDPA), training loop, story generation |
| `prepare_data.py` | Downloads TinyStories + 10 Project Gutenberg books, trains 4,096-vocab BPE, writes `uint16` `.bin` shards |
| `tokenizer.py` | From-scratch word-chunked BPE tokenizer (`minbpe` / GPT-2 style) |
| `tokenizer.json` | Learned BPE merges (4,096 vocab on TinyStories + Project Gutenberg) |
| `brain.pt` | Trained weights |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB) |
| `how-hedwig-learns.html` | Visual explainer: knobs, training, model sizes |
| `docs/LEARNINGS.md` | Concepts and measured results |
| `docs/ROADMAP.md` | What to build next |
| `docs/CLOUD_GPU_100.md` | How to spend $100 of cloud GPU time |
| `logs/DEVLOG.md` | Automatic daily dev log (updated on push) |
| `AGENTS.md` | Context for AI coding assistants |

## Where it's going
A focused storytelling model that runs offline on a Mac and iPhone. See [docs/ROADMAP.md](docs/ROADMAP.md).
