# 🦉 Hedwig

A tiny GPT-style language model (~4.8M parameters) trained from scratch on a MacBook (Apple M4) using PyTorch.
Hedwig reads Shakespeare one letter at a time and learns to guess the next letter.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch numpy
.venv/bin/python tinygpt.py talk "ROMEO:"   # use the included brain.pt
.venv/bin/python tinygpt.py train           # or retrain it yourself (~15 min on M4)
```

## Files

- `tinygpt.py`: the model, training loop and text generation
- `data/input.txt`: Tiny Shakespeare (~1 MB of text)
- `brain.pt`: Hedwig's trained weights
