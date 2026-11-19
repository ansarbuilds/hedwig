"""
Give the English Hedwig more stories to read, then refresh chat on top.

The first brain read about 50 MB of TinyStories and stopped improving around
a val score of 2.08. This script downloads a taller slice (200 MB), cuts it
into pieces with the SAME dictionary (tokenizer.json is not retrained), and
keeps teaching from brain.pt. The test pages stay data/val.bin, so the new
score can be compared with 2.08.

If the storyteller gets better or holds steady, Stage 2 and Stage 3 are
trained again so the chat brain, not just the autocomplete brain, knows the
new stories. The previous chat files are copied to *.bak.pt first.

Run:  .venv/bin/python more_english.py
"""
import os, shutil, time, urllib.request
import numpy as np
import torch
import tokenizer
import prepare_data
import tinygpt
import sft
import dpo

# ---- Settings ----
more_mb = 200          # how much TinyStories to read (the first run used 50)
more_steps = 2000      # extra practice steps on the taller pile
lr = 3e-4              # gentler than the first 1e-3, because the brain already knows stories
eval_every = 250
train_bin = "data/train_more.bin"
ckpt_path = "ckpt_more.pt"
text_path = "data/tinystories_200.txt"
old_best = 2.080       # the published Stage 1 score on data/val.bin

def download_more():
    """Fetch a 200 MB slice. The 50 MB file used before is left alone."""
    if os.path.exists(text_path) and os.path.getsize(text_path) > 100_000_000:
        print(f"  ✓ {text_path} already on disk ({os.path.getsize(text_path)/1e6:.0f} MB)", flush=True)
        return open(text_path, encoding="utf-8", errors="ignore").read()
    max_bytes = more_mb * 1024 * 1024
    print(f"  ↓ downloading {more_mb} MB of TinyStories...", flush=True)
    req = urllib.request.Request(
        prepare_data.TINYSTORIES_TRAIN_URL,
        headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "Hedwig/1.0"},
    )
    raw = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", errors="ignore")
    cut = raw.rfind("<|endoftext|>")
    if cut != -1:
        raw = raw[:cut]
    open(text_path, "w", encoding="utf-8").write(raw)
    print(f"  ✓ saved {text_path} ({len(raw)/1e6:.0f} MB)", flush=True)
    return raw

def write_tokens(text, merges, path):
    """Turn a long string into a uint16 file, one chunk at a time so RAM stays calm."""
    cache = {}
    tmp = path + ".tmp"
    total = 0
    with open(tmp, "wb") as out:
        start = 0
        while start < len(text):
            end = min(len(text), start + 500_000)
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start:
                    end = nl + 1
            ids = np.array(tokenizer.encode(text[start:end], merges, cache), dtype=np.uint16)
            ids.tofile(out)
            total += len(ids)
            start = end
            if total and total % 2_000_000 < len(ids):
                print(f"    {total/1e6:.1f}M pieces", flush=True)
    os.replace(tmp, path)
    return total

def build_train_bin():
    if os.path.exists(train_bin) and os.path.getsize(train_bin) > 20_000_000:
        n = os.path.getsize(train_bin) // 2
        print(f"  ✓ {train_bin} already built ({n:,} pieces)", flush=True)
        return n
    print("Building a taller English story pile with the existing dictionary...", flush=True)
    stories = prepare_data.clean_tinystories(download_more())
    print("  adding the same 10 Gutenberg novels...", flush=True)
    books, _ = prepare_data.load_gutenberg_books()
    text = stories + "\n\n" + books
    n = write_tokens(text, tinygpt.merges, train_bin)
    print(f"  ✓ {train_bin}: {n:,} pieces", flush=True)
    return n

def val_score(model, val_data):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
        return sum(model(*tinygpt.get_batch(val_data))[1].item() for _ in range(20)) / 20

def save_brain(model, path):
    sd = {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach()) for k, v in model.state_dict().items()}
    torch.save(sd, path + ".tmp")
    os.replace(path + ".tmp", path)

def train_more():
    build_train_bin()
    train_data = torch.from_numpy(np.fromfile(train_bin, dtype=np.uint16).astype(np.int64))
    val_data = torch.from_numpy(np.fromfile(tinygpt.val_bin, dtype=np.uint16).astype(np.int64))
    print(f"Train pieces {len(train_data):,} | same test pages {len(val_data):,}", flush=True)
    model = tinygpt.TinyGPT().to(tinygpt.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, more_steps)
    start, best = 0, old_best
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=tinygpt.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start, best = ckpt["step"], ckpt.get("best_val", old_best)
        print(f"Resuming extra reading from step {start} (best {best:.3f})", flush=True)
    else:
        model.load_state_dict(torch.load("brain.pt", map_location=tinygpt.device, weights_only=True))
        print("Loaded brain.pt. Measuring the old score on the same test pages...", flush=True)
        start_score = val_score(model, val_data)
        best = min(old_best, start_score)
        print(f"step    0 | val score {start_score:.3f} | published best was {old_best:.3f}", flush=True)
    model.train()
    t0 = time.time()
    for step in range(start, more_steps + 1):
        if step > 0 and step % eval_every == 0:
            vl = val_score(model, val_data)
            print(f"step {step:4d} | val score {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if vl < best:
                best = vl
                save_brain(model, "brain.pt")
                print(f"  saved brain.pt ({best:.3f})", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "best_val": best}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            model.train()
        if step == more_steps:
            break
        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            _, loss = model(*tinygpt.get_batch(train_data))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f"Extra reading done. Best val score {best:.3f} (was {old_best:.3f})", flush=True)
    return best

def backup(path):
    bak = path + ".bak"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copyfile(path, bak)
        print(f"  copied {path} -> {bak}", flush=True)

def refresh_chat():
    """Teach manners and taste again, starting from the new storyteller."""
    print("Refreshing Stage 2 (manners) and Stage 3 (taste) from the new brain.pt ...", flush=True)
    backup("brain_chat.pt")
    backup("brain_dpo.pt")
    for stale in ("ckpt_sft.pt", "ckpt_sft.pt.tmp", "ckpt_dpo.pt", "ckpt_dpo.pt.tmp"):
        if os.path.exists(stale):
            os.remove(stale)
    sft.train()
    dpo.train()

if __name__ == "__main__":
    best = train_more()
    if best <= old_best + 0.02:
        refresh_chat()
    else:
        print("Val score got worse by more than 0.02, so the chat brains were left as they were.", flush=True)
