# 🦉 Hedwig

A tiny GPT-style language model built from scratch to learn how LLMs work.
v1 is a 4.8M-parameter character-level Transformer trained on Tiny Shakespeare in ~16 minutes on a MacBook Air M4 (final loss 1.48).

```
ROMEO:
And struck at Petruchio, by what a thou blest,
And like to venge her; our proported service
```

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tinygpt.py talk "ROMEO:"   # use the included brain.pt
.venv/bin/python tinygpt.py train           # or retrain (~16 min on M4, faster on CUDA)
```

## Files

| Path | What |
|---|---|
| `tinygpt.py` | Model, training loop, text generation |
| `brain.pt` | Trained weights (18 MB) |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB) |
| `how-hedwig-learns.html` | Visual explainer: knobs, training, model sizes |
| `docs/LEARNINGS.md` | Concepts and measured results |
| `docs/ROADMAP.md` | What to build next |
| `docs/CLOUD_GPU_100.md` | How to spend $100 of cloud GPU time |
| `AGENTS.md` | Context for AI coding assistants |

## Where it's going
A focused storytelling model that runs offline on a Mac and iPhone. See [docs/ROADMAP.md](docs/ROADMAP.md).
