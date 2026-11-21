"""
Step 1 of Hedwig's movie: the 286.7M Flagship Hedwig writes the story.
Writes several drafts (different seeds) to movie/drafts.md so we can pick the best one.

Run:  .venv/bin/python movie/write_story.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import app, sft

BRAIN       = "c300_dpo_16"
PROMPT      = "Write a story about a little owl named Pip who loses a golden key and his friend Lily helps him find it."
SEEDS       = [1, 2, 3, 4, 5, 6, 7, 8]
TEMPERATURE = 0.7

model, *_, tok_path, ctx_len = app.get_model(BRAIN)
app.use_tokenizer(tok_path, ctx_len)
out = [f"# Drafts\n\nPrompt: {PROMPT}\n"]
for s in SEEDS:
    torch.manual_seed(s)
    text = sft.reply(model, PROMPT, max_new_tokens=350, temperature=TEMPERATURE)
    out.append(f"## Seed {s}\n\n{text}\n")
    print(f"--- seed {s} ---\n{text}\n", flush=True)
open(os.path.join(os.path.dirname(__file__), "drafts.md"), "w").write("\n".join(out))
