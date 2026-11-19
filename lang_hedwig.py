"""
A second Hedwig that reads French, without touching the English brain.

The books are public domain (Project Gutenberg, via the GITenberg mirror):
old French novels whose authors died long ago. We train a small brain on them
with the same kind of model as tinygpt.py, just fewer knobs, and we save it
under lang/ so brain.pt stays the English storyteller.

Run:  .venv/bin/python lang_hedwig.py train
      .venv/bin/python lang_hedwig.py talk "Il était une fois un petit hibou "
"""
import os, re, sys, time, urllib.request
import numpy as np
import torch
import tokenizer

# The English dictionary only groups a-z. French needs é, à, ç, and œ to stay in the word
# instead of being dropped on the floor. English tokenizer.json is not rebuilt.
FRENCH_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZÀ-ÖØ-öø-ÿœŒ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"
)
import prepare_data
import tinygpt

# ---- Settings (a smaller brain: about 2.6M knobs, so the Air can finish) ----
lang_name = "French"
vocab_size = 2048
block_size = 128
n_embd = 192
n_head = 4
n_layer = 4
dropout = 0.05
batch_size = 32
steps = 600
lr = 1e-3
ckpt_every = 100
lang_dir = "lang"
tok_path = "lang/tokenizer.json"
train_bin = "lang/train.bin"
val_bin = "lang/val.bin"
brain_path = "lang/brain_fr.pt"
ckpt_path = "lang/ckpt.pt"

# Public-domain French (or we check the words and skip an English translation).
BOOKS = [
    ("mousquetaires.txt", "https://raw.githubusercontent.com/GITenberg/Les-trois-mousquetaires_13951/master/13951-8.txt"),
    ("mousquetaires_2.txt", "https://raw.githubusercontent.com/GITenberg/Les-trois-mousquetaires-Volume-2-of-2_58211/master/58211-0.txt"),
    ("notredame.txt", "https://raw.githubusercontent.com/GITenberg/Notre-Dame-de-Paris_19657/master/19657-8.txt"),
    ("lieues.txt", "https://raw.githubusercontent.com/GITenberg/Vingt-mille-Lieues-Sous-Les-Mers---Complete_5097/master/5097-8.txt"),
    ("lieues_2.txt", "https://raw.githubusercontent.com/GITenberg/Vingt-mille-lieues-sous-les-mers_54873/master/54873-0.txt"),
]

def looks_french(text):
    """A rough door check: real French says 'le/la/les' a lot more than 'the'."""
    low = text.lower()
    fr = sum(low.count(w) for w in (" le ", " la ", " les ", " une ", " et "))
    en = sum(low.count(w) for w in (" the ", " and ", " of "))
    return fr > en * 2 and fr > 100

def download_books():
    os.makedirs(os.path.join(lang_dir, "books"), exist_ok=True)
    parts = []
    for name, url in BOOKS:
        path = os.path.join(lang_dir, "books", name)
        if not os.path.exists(path):
            print(f"  ↓ {name}", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Hedwig/1.0"})
                raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
            except Exception as exc:
                print(f"    skipped ({exc})", flush=True)
                continue
            open(path, "w", encoding="utf-8").write(raw)
        else:
            raw = open(path, encoding="utf-8", errors="ignore").read()
        clean = prepare_data.clean_gutenberg(raw)
        if not looks_french(clean):
            print(f"  · {name}: not French, left out", flush=True)
            continue
        parts.append(clean)
        print(f"  ✓ {name}: {len(clean)/1e6:.2f} MB of French", flush=True)
    if not parts:
        raise SystemExit("No French books downloaded. Check the network and try again.")
    return "\n\n".join(parts)

def prepare():
    """Download the books, learn a French word-piece dictionary, write train.bin and val.bin."""
    os.makedirs(lang_dir, exist_ok=True)
    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(tok_path):
        print("French stories already prepared. Delete lang/train.bin to redo.", flush=True)
        return
    print("1/3 Downloading public-domain French books...", flush=True)
    text = download_books()
    split = int(0.9 * len(text))
    train_text, val_text = text[:split], text[split:]
    print(f"\n2/3 Learning a {vocab_size}-piece French dictionary...", flush=True)
    merges, vocab = tokenizer.train(train_text[:2_000_000], vocab_size=vocab_size, pat=FRENCH_PAT)
    tokenizer.save(merges, tok_path)
    print(f"  ✓ {tok_path} ({len(vocab)} pieces)", flush=True)
    print("\n3/3 Cutting the books into number shards...", flush=True)
    cache = {}
    train_ids = np.array(tokenizer.encode(train_text, merges, cache, pat=FRENCH_PAT), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text, merges, cache, pat=FRENCH_PAT), dtype=np.uint16)
    train_ids.tofile(train_bin + ".tmp"); os.replace(train_bin + ".tmp", train_bin)
    val_ids.tofile(val_bin + ".tmp"); os.replace(val_bin + ".tmp", val_bin)
    print(f"  ✓ train {len(train_ids):,} pieces, val {len(val_ids):,} pieces", flush=True)

def use_french_size():
    """Point the shared TinyGPT builder at this smaller French size. English files are not edited."""
    tinygpt.vocab_size = vocab_size
    tinygpt.block_size = block_size
    tinygpt.n_embd = n_embd
    tinygpt.n_head = n_head
    tinygpt.n_layer = n_layer
    tinygpt.dropout = dropout
    tinygpt.batch_size = batch_size

def load_bins():
    train = torch.from_numpy(np.fromfile(train_bin, dtype=np.uint16).astype(np.int64))
    val = torch.from_numpy(np.fromfile(val_bin, dtype=np.uint16).astype(np.int64))
    return train, val

def get_batch(data):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(tinygpt.device), y.to(tinygpt.device)

def train():
    prepare()
    use_french_size()
    train_data, val_data = load_bins()
    model = tinygpt.TinyGPT().to(tinygpt.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"French brain: {n/1e6:.2f}M knobs on {tinygpt.device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start, best = 0, float("inf")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=tinygpt.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start, best = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"Resuming from step {start} (best {best:.3f})", flush=True)
    t0 = time.time()
    for step in range(start, steps + 1):
        if step % ckpt_every == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
                vl = sum(model(*get_batch(val_data))[1].item() for _ in range(10)) / 10
            print(f"step {step:4d} | French val score {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best:
                best = vl
                sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
                torch.save(sd, brain_path + ".tmp")
                os.replace(brain_path + ".tmp", brain_path)
                print(f"  saved {brain_path} ({best:.3f})", flush=True)
            model.train()
        if step > start and step % ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "best_val": best}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            _, loss = model(*get_batch(train_data))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f"Done. Best French score: {best:.3f}", flush=True)

def talk():
    use_french_size()
    merges, vocab = tokenizer.load(tok_path)
    model = tinygpt.TinyGPT().to(tinygpt.device)
    model.load_state_dict(torch.load(brain_path, map_location=tinygpt.device, weights_only=True))
    model.eval()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Il était une fois un petit hibou "
    ids = tokenizer.encode(prompt, merges, pat=FRENCH_PAT)
    idx = torch.tensor([ids], device=tinygpt.device)
    # Greedy-enough sampling using the English helper's dice, on French pieces.
    out = model.generate(idx, 80, temperature=0.8, top_k=40, rep_penalty=1.1)
    print(tokenizer.decode(out[0].tolist(), vocab))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    train() if cmd == "train" else talk()
