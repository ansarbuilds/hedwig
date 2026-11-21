"""
Fair side-by-side Story & Grammar Showdown across Hedwig's three generations:
  - 14M Mac Hedwig (4,096 vocab, TinyStories + 10 novels)
  - 97.6M Cloud Hedwig (8,192 vocab, 614 MB TinyStories + FineWeb-Edu + 15k CoEdIT)
  - 286.7M Flagship Hedwig (16,384 vocab, 512 context, 1.22 GB + 30k CoEdIT)

Because 4k / 8k / 16k vocab models cannot be compared by raw cross-entropy numbers,
every brain answers the EXACT same 12 fixed prompts with the EXACT same seed and temperature.
Writes the full side-by-side report to docs/STORY_SHOWDOWN.md.

Run:   .venv/bin/python compare_brains.py
"""
import os, time
import torch
import app, sft

TEMPERATURE = 0.65
SEED        = 42
OUT_PATH    = "docs/STORY_SHOWDOWN.md"

BRAINS = [
    ("dpo_16",      "14M Mac Hedwig (DPO 16-bit)"),
    ("c97_dpo_16",  "97.6M Cloud Hedwig (DPO 16-bit)"),
    ("c300_dpo_16", "286.7M Flagship Hedwig (DPO 16-bit)"),
]

PROMPTS = [
    ("Story 1 — Lily & the shiny key",
     "Tell me a story where a little girl named Lily finds a shiny key in the garden."),
    ("Story 2 — Brave owl Pip",
     "Write a short story about a brave little owl named Pip and a golden key."),
    ("Story 3 — Harry Potter name test",
     "Write a story about a boy named Harry Potter."),
    ("Story 4 — Dragon & castle",
     "Write a short story about dragon and castle."),
    ("Story 5 — Lost puppy in the rain",
     "Write a short story about a little puppy who gets lost in the rain and finds his way home."),
    ("Story 6 — Lily shares her red umbrella",
     "Tell me a story where Lily shares her red umbrella with a small frog."),
    ("Story 7 — Owl who was afraid of the dark",
     "Write a short story about a baby owl who was afraid of the dark until he saw the moon."),
    ("Story 8 — Magic wooden box",
     "Write a short story about a boy named Max who opens a magic wooden box in the forest."),
    ("Story 9 — Two friends build a treehouse",
     "Write a short story about two best friends who build a treehouse together."),
    ("Story 10 — Clockwork bird",
     "Write a short story about a girl who fixes a broken toy bird."),
    ("Grammar 1 — Teams message",
     "Fix grammar: hey can we move the call tomorrow im stuck with the update and need 2 hrs"),
    ("Grammar 2 — Past tense & verbs",
     "Improve the grammar of this text: I goes to the office yesterday and send the report to my manager."),
]

def main():
    available = [(k, name) for k, name in BRAINS if os.path.exists(app.BRAIN_SPECS[k][1])]
    results = {k: [] for k, _ in available}

    for k, name in available:
        print(f"\n=== Loading {name} ===", flush=True)
        model, _, _, _, mb, tok_path, ctx_len = app.get_model(k)
        app.use_tokenizer(tok_path, ctx_len)
        for title, prompt in PROMPTS:
            torch.manual_seed(SEED)
            t0 = time.time()
            text = sft.reply(model, prompt, max_new_tokens=200, temperature=TEMPERATURE)
            dt = time.time() - t0
            results[k].append((text, dt))
            print(f"  ✓ {title:36s} ({len(text):4d} chars, {dt:.2f}s)", flush=True)
        # Free MPS memory before loading the next large brain
        app.MODELS.pop(k, None)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    lines = [
        "# 🦉 Hedwig Side-by-Side Story & Grammar Showdown",
        "",
        f"Same **{len(PROMPTS)} fixed prompts**, same random seed (`{SEED}`), same creativity (`temperature = {TEMPERATURE}`).",
        "",
        "Because `14M` (4k vocab), `97.6M` (8k vocab), and `286.7M` (16k vocab) use different dictionaries, raw quiz scores cannot be compared directly. Reading their real answers side-by-side is the fair test.",
        "",
    ]
    for i, (title, prompt) in enumerate(PROMPTS):
        lines.append(f"## {title}")
        lines.append(f"> **Prompt:** `{prompt}`")
        lines.append("")
        for k, name in available:
            text, dt = results[k][i]
            lines.append(f"### {name} *(generated in {dt:.2f}s)*")
            lines.append(text.strip() or "*(empty)*")
            lines.append("")
        lines.append("---")
        lines.append("")

    open(OUT_PATH, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n✅ Saved full side-by-side showdown -> {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
