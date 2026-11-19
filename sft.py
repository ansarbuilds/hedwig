"""
Stage 2 for Hedwig: Supervised Fine-Tuning (SFT) — teaching Hedwig "chat manners".

Stage 1 (tinygpt.py) taught Hedwig how language and stories work (autocomplete).
Stage 2 (sft.py) loads brain.pt and trains on thousands of (User request -> Hedwig story)
pairs from TinyStories-Instruct, masking the prompt loss (ignore_index = -100) so Hedwig
learns to follow instructions like "Write a story about a brave owl and a golden key."

Run:   .venv/bin/python sft.py train                                    (fine-tune brain.pt -> brain_chat.pt)
       .venv/bin/python sft.py talk "Write a story about a little owl." (one-shot story request)
       .venv/bin/python sft.py chat                                     (interactive chat in terminal)
"""
import os, re, sys, time, random, urllib.request
import torch, torch.nn.functional as F
import tinygpt

# ---- SFT Settings ----
steps        = 750               # SFT needs far fewer steps than pretraining
lr           = 2e-4              # gentle learning rate (5x smaller) so we don't forget English
batch_size   = 32
eval_every   = 150
train_mb     = 25                # MB of TinyStories-Instruct-train.txt to download
val_mb       = 4                 # MB of TinyStories-Instruct-valid.txt to download
sft_cache    = "data/sft_cache.pt"
chat_brain   = "brain_chat.pt"
sft_ckpt     = "ckpt_sft.pt"

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt"
VAL_URL   = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt"

NOUNS = {
    "owl", "bird", "bear", "dog", "cat", "puppy", "kitty", "rabbit", "bunny", "fox", "mouse",
    "frog", "lion", "tiger", "monkey", "duck", "fish", "dragon", "king", "queen", "princess",
    "prince", "knight", "wizard", "fairy", "boy", "girl", "friend", "tree", "forest", "woods",
    "park", "garden", "river", "lake", "sea", "beach", "castle", "house", "magic", "key",
    "box", "ball", "car", "train", "boat", "book", "toy", "flower", "star", "moon", "sun",
    "treasure", "crown", "ring", "wand", "balloon", "kite", "cake", "apple",
}

def download_slice(url, path, max_mb):
    """Download up to max_mb MB of TinyStories-Instruct if not already cached."""
    if os.path.exists(path):
        return open(path, encoding="utf-8", errors="ignore").read()
    os.makedirs("data", exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024)
    print(f"  ↓ downloading {max_mb} MB -> {path} ...", flush=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "Hedwig/1.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
    cut = raw.rfind("<|endoftext|>")
    if cut != -1:
        raw = raw[:cut]
    open(path, "w", encoding="utf-8").write(raw)
    return raw

def make_prompt(summary, words_line, story, idx):
    """Turn a TinyStories-Instruct summary/keywords into varied user requests."""
    topics = []
    if words_line:
        topics = [w.strip().lower() for w in words_line.split(",") if w.strip()]
    for w in re.findall(r"[A-Za-z]+", story):
        lw = w.lower()
        if lw in NOUNS and lw not in topics:
            topics.append(lw)
    if idx % 3 == 0 and len(topics) >= 2:
        return f"Write a short story about {topics[0]} and {topics[1]}."
    if idx % 5 == 0 and summary:
        s = summary[0].lower() + summary[1:] if len(summary) > 1 else summary
        return f"Tell me a story where {s}"
    if summary:
        return f"Write a story: {summary}"
    return f"Write a short story about {topics[0] if topics else 'a little friend'}."

def parse_examples(raw_text, word_cache):
    """Build (x, y) token tensors where prompt & padding tokens have target = -100."""
    B_T = tinygpt.block_size
    xs, ys = [], []
    records = raw_text.split("<|endoftext|>")
    for idx, rec in enumerate(records):
        if "Story:" not in rec:
            continue
        head, _, story_raw = rec.partition("Story:")
        paragraphs = [p.strip() for p in story_raw.replace("\r\n", "\n").splitlines()
                      if p.strip() and not p.strip().startswith(("Summary:", "Features:", "Random sentence:", "Words:"))]
        if not paragraphs:
            continue
        story = "\n".join(paragraphs)
        summary, words_line = "", ""
        for line in head.splitlines():
            if line.startswith("Summary:"):
                summary = line[len("Summary:"):].strip()
            elif line.startswith("Words:"):
                words_line = line[len("Words:"):].strip()
        if not summary and not words_line:
            continue

        prompt = make_prompt(summary, words_line, story, idx)
        prefix_ids = tinygpt.tokenizer.encode(f"User: {prompt}\nHedwig: ", tinygpt.merges, word_cache)
        story_ids  = tinygpt.tokenizer.encode(story + "\n\n", tinygpt.merges, word_cache)
        full = prefix_ids + story_ids
        if len(full) > B_T + 1 or len(story_ids) < 25:
            continue

        # Build input x and target y; mask prompt prefix and right-padding with -100
        x = full[:-1] + [0] * (B_T - len(full) + 1)
        y = ([-100] * (len(prefix_ids) - 1)) + story_ids + ([-100] * (B_T - len(full) + 1))
        xs.append(x[:B_T])
        ys.append(y[:B_T])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

def load_sft_data():
    """Load or build cached SFT instruction -> story tensors."""
    if os.path.exists(sft_cache):
        d = torch.load(sft_cache)
        return d["tr_x"], d["tr_y"], d["va_x"], d["va_y"]
    print("Building SFT (User -> Hedwig) story dataset from TinyStories-Instruct...", flush=True)
    tr_raw = download_slice(TRAIN_URL, "data/sft_train.txt", train_mb)
    va_raw = download_slice(VAL_URL,   "data/sft_val.txt",   val_mb)
    word_cache = {}
    tr_x, tr_y = parse_examples(tr_raw, word_cache)
    va_x, va_y = parse_examples(va_raw, word_cache)
    torch.save({"tr_x": tr_x, "tr_y": tr_y, "va_x": va_x, "va_y": va_y}, sft_cache)
    print(f"  ✓ Cached {len(tr_x):,} train pairs and {len(va_x):,} val pairs -> {sft_cache}", flush=True)
    return tr_x, tr_y, va_x, va_y

def get_sft_batch(x_data, y_data):
    ix = torch.randint(len(x_data), (batch_size,))
    return x_data[ix].to(tinygpt.device), y_data[ix].to(tinygpt.device)

def train():
    tr_x, tr_y, va_x, va_y = load_sft_data()
    model = tinygpt.TinyGPT().to(tinygpt.device)
    if os.path.exists("brain.pt"):
        model.load_state_dict(torch.load("brain.pt", map_location=tinygpt.device))
        print(f"Loaded pretrained 13.8M brain.pt | Fine-tuning on {len(tr_x):,} instruction -> story pairs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    start_step, best_val = 0, float("inf")
    if os.path.exists(sft_ckpt):
        ckpt = torch.load(sft_ckpt, map_location=tinygpt.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step, best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"Resuming SFT from step {start_step} (best score: {best_val:.3f})", flush=True)

    t0 = time.time()
    for step in range(start_step, steps + 1):
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
                vl = sum(model(*get_sft_batch(va_x, va_y))[1].item() for _ in range(20)) / 20
            print(f"SFT step {step:4d} | story response score (lower=smarter) {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best_val:
                best_val = vl
                bf16_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
                torch.save(bf16_sd, chat_brain + ".tmp")
                os.replace(chat_brain + ".tmp", chat_brain)
                print(f"  🌟 saved new best {chat_brain} (score {best_val:.3f})", flush=True)
            model.train()

        if step > start_step and step % eval_every == 0:
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best_val": best_val}
            torch.save(ckpt, sft_ckpt + ".tmp")
            os.replace(sft_ckpt + ".tmp", sft_ckpt)

        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            _, loss = model(*get_sft_batch(tr_x, tr_y))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    print(f"Done! Best {chat_brain} score: {best_val:.3f}", flush=True)

@torch.no_grad()
def reply(model, prompt, max_new_tokens=220, temperature=0.75):
    """Format a user prompt, generate Hedwig's story response, and stop at story end (\n\n)."""
    prefix = f"User: {prompt.strip()}\nHedwig: "
    idx = torch.tensor([tinygpt.encode(prefix)], device=tinygpt.device)
    start_len = idx.size(1)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -tinygpt.block_size:])
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        text = tinygpt.decode(idx[0, start_len:].tolist())
        for stop in ("\n\n", "Summary:", "Features:", "Random sentence:", "User:"):
            if stop in text:
                return text.split(stop)[0].strip()
    return tinygpt.decode(idx[0, start_len:].tolist()).strip()

def talk():
    model = tinygpt.TinyGPT().to(tinygpt.device)
    weights = chat_brain if os.path.exists(chat_brain) else "brain.pt"
    model.load_state_dict(torch.load(weights, map_location=tinygpt.device))
    model.eval()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a short story about a little owl and a magic key."
    print(f"🧑 You: {prompt}\n\n🦉 Hedwig:\n{reply(model, prompt)}")

def chat():
    model = tinygpt.TinyGPT().to(tinygpt.device)
    weights = chat_brain if os.path.exists(chat_brain) else "brain.pt"
    model.load_state_dict(torch.load(weights, map_location=tinygpt.device))
    model.eval()
    print("🦉 Hedwig Chat Storyteller (type 'quit' to exit)\n")
    while True:
        try:
            prompt = input("🧑 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not prompt or prompt.lower() in ("quit", "exit"):
            break
        print(f"\n🦉 Hedwig >\n{reply(model, prompt)}\n")

if __name__ == "__main__":
    cmd = (sys.argv[1:] or ["train"])[0]
    train() if cmd == "train" else chat() if cmd == "chat" else talk()
