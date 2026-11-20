"""
Phase 2 data pipeline for Hedwig:
1. Downloads TinyStories (roneneldan/TinyStories) from Hugging Face into data/
2. Downloads 10 classic public-domain Project Gutenberg storybooks into data/gutenberg/
3. Cleans headers/footers, trains a 4,096-piece BPE tokenizer (tokenizer.json)
4. Tokenizes and saves fast uint16 binary shards: data/train.bin and data/val.bin
"""
import os, time, urllib.request
import numpy as np
import tokenizer

# ---- Settings ----
vocab_size           = 4096   # BPE pieces (4096 on Mac, 8192 on cloud --124m; fits cleanly in uint16)
tinystories_train_mb = 50     # MB of TinyStories-train.txt to download (bump anytime up to 1924)
tinystories_val_mb   = 10     # MB of TinyStories-valid.txt to download
fineweb_edu_mb       = 0      # MB of modern English (FineWeb-Edu, ODC-By) to mix in (0 = stories only)
bpe_sample_mb        = 4      # MB of text used to train the BPE tokenizer
data_dir             = "data"
gutenberg_dir        = "data/gutenberg"
train_bin_path       = "data/train.bin"
val_bin_path         = "data/val.bin"
tok_path             = "tokenizer.json"

TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
TINYSTORIES_VAL_URL   = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
FINEWEB_EDU_URL       = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/shard_{:05d}.parquet"

# Public-domain classic storybooks (pre-1929 Project Gutenberg via GITenberg mirror)
GUTENBERG_BOOKS = [
    ("alice.txt",           "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt"),
    ("looking_glass.txt",   "https://raw.githubusercontent.com/GITenberg/Through-the-Looking-Glass_12/master/12-0.txt"),
    ("wizard_of_oz.txt",    "https://raw.githubusercontent.com/GITenberg/The-Wonderful-Wizard-of-Oz_55/master/55.txt"),
    ("peter_pan.txt",       "https://raw.githubusercontent.com/GITenberg/Peter-Pan_16/master/16-0.txt"),
    ("grimm_tales.txt",     "https://raw.githubusercontent.com/GITenberg/Grimms-Fairy-Tales_2591/master/2591-0.txt"),
    ("andersen_tales.txt",  "https://raw.githubusercontent.com/GITenberg/Andersen-s-Fairy-Tales_1597/master/1597.txt"),
    ("jungle_book.txt",     "https://raw.githubusercontent.com/GITenberg/The-Jungle-Book_236/master/236-0.txt"),
    ("treasure_island.txt", "https://raw.githubusercontent.com/GITenberg/Treasure-Island_120/master/120-0.txt"),
    ("sherlock_holmes.txt", "https://raw.githubusercontent.com/GITenberg/The-Adventures-of-Sherlock-Holmes_1661/master/1661.txt"),
    ("time_machine.txt",    "https://raw.githubusercontent.com/GITenberg/The-Time-Machine_35/master/35-0.txt"),
]

def download_slice(url, path, max_mb):
    """Download up to max_mb megabytes from url into path (if not already on disk)."""
    if os.path.exists(path):
        print(f"  ✓ {path} already exists ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
        return open(path, encoding="utf-8", errors="ignore").read()
    max_bytes = int(max_mb * 1024 * 1024)
    print(f"  ↓ downloading {max_mb} MB -> {path} ...", flush=True)
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": "Hedwig/1.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="ignore")
    cut = raw.rfind("<|endoftext|>")
    if cut != -1:
        raw = raw[:cut]
    open(path, "w", encoding="utf-8").write(raw)
    print(f"  ✓ saved {path} ({len(raw)/1e6:.1f} MB)", flush=True)
    return raw

def clean_tinystories(text):
    """Turn <|endoftext|> markers into clean blank-line story dividers."""
    return text.replace("\r\n", "\n").replace("<|endoftext|>", "\n\n").strip()

def clean_gutenberg(text):
    """Strip Project Gutenberg legal header and footer so only story prose remains."""
    text = text.replace("\r\n", "\n")
    for marker in ("*** START OF THIS PROJECT GUTENBERG", "*** START OF THE PROJECT GUTENBERG"):
        idx = text.find(marker)
        if idx != -1:
            nl = text.find("\n", idx)
            text = text[nl + 1:] if nl != -1 else text[idx:]
            break
    for marker in ("*** END OF THIS PROJECT GUTENBERG", "*** END OF THE PROJECT GUTENBERG", "End of the Project Gutenberg", "End of Project Gutenberg"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()

def load_gutenberg_books():
    """Download and clean all classic Project Gutenberg storybooks."""
    os.makedirs(gutenberg_dir, exist_ok=True)
    train_parts, val_parts = [], []
    for name, url in GUTENBERG_BOOKS:
        path = os.path.join(gutenberg_dir, name)
        if not os.path.exists(path):
            raw = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", errors="ignore")
            open(path, "w", encoding="utf-8").write(raw)
        else:
            raw = open(path, encoding="utf-8", errors="ignore").read()
        clean = clean_gutenberg(raw)
        split = int(0.9 * len(clean))
        train_parts.append(clean[:split])
        val_parts.append(clean[split:])
        print(f"  ✓ Gutenberg {name:20s}: {len(clean)/1e3:6.1f} KB cleaned", flush=True)
    return "\n\n".join(train_parts), "\n\n".join(val_parts)

def load_fineweb_edu(max_mb):
    """Download clean modern English from FineWeb-Edu parquet shards (ODC-By license)."""
    if max_mb <= 0:
        return "", ""
    import pyarrow.parquet as pq
    target_bytes = int(max_mb * 1024 * 1024)
    texts, total, shard_idx = [], 0, 0
    while total < target_bytes:
        shard_path = os.path.join(data_dir, f"fineweb_edu_{shard_idx:05d}.parquet")
        if not os.path.exists(shard_path):
            url = FINEWEB_EDU_URL.format(shard_idx)
            print(f"  ↓ downloading FineWeb-Edu shard {shard_idx} -> {shard_path} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Hedwig/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(shard_path + ".tmp", "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(shard_path + ".tmp", shard_path)
        table = pq.read_table(shard_path, columns=["text"])
        for doc in table.column("text").to_pylist():
            if not doc:
                continue
            doc = doc.replace("\r\n", "\n").strip()
            texts.append(doc)
            total += len(doc) + 2
            if total >= target_bytes:
                break
        shard_idx += 1
    full = "\n\n".join(texts)
    split = int(0.95 * len(full))
    print(f"  ✓ FineWeb-Edu: {len(full)/1e6:.1f} MB modern English ({len(texts):,} docs)", flush=True)
    return full[:split], full[split:]

def main():
    t0 = time.time()
    os.makedirs(data_dir, exist_ok=True)

    print("1/4 Downloading TinyStories (roneneldan/TinyStories)...", flush=True)
    ts_train = clean_tinystories(download_slice(TINYSTORIES_TRAIN_URL, "data/tinystories_train.txt", tinystories_train_mb))
    ts_val   = clean_tinystories(download_slice(TINYSTORIES_VAL_URL,   "data/tinystories_val.txt",   tinystories_val_mb))

    print("\n2/4 Downloading public-domain Project Gutenberg novels (+ FineWeb-Edu if enabled)...", flush=True)
    gb_train, gb_val = load_gutenberg_books()
    fw_train, fw_val = load_fineweb_edu(fineweb_edu_mb)

    train_text = ts_train + "\n\n" + gb_train + (("\n\n" + fw_train) if fw_train else "")
    val_text   = ts_val   + "\n\n" + gb_val   + (("\n\n" + fw_val)   if fw_val   else "")
    print(f"\nCombined corpus: train = {len(train_text)/1e6:.1f} MB | val = {len(val_text)/1e6:.1f} MB", flush=True)

    print(f"\n3/4 Training BPE tokenizer (vocab_size = {vocab_size})...", flush=True)
    sample_bytes = int(bpe_sample_mb * 1024 * 1024)
    if fw_train:
        half = sample_bytes // 2
        bpe_sample = ts_train[:half] + "\n\n" + fw_train[:half] + "\n\n" + gb_train
    else:
        bpe_sample = ts_train[:sample_bytes] + "\n\n" + gb_train
    merges, vocab = tokenizer.train(bpe_sample, vocab_size=vocab_size)
    tokenizer.save(merges, tok_path)
    print(f"  ✓ Saved {tok_path} ({len(vocab)} pieces)", flush=True)

    print("\n4/4 Tokenizing corpus into uint16 .bin shards...", flush=True)
    cache = {}
    t_enc = time.time()
    train_ids = np.array(tokenizer.encode(train_text, merges, cache), dtype=np.uint16)
    val_ids   = np.array(tokenizer.encode(val_text,   merges, cache), dtype=np.uint16)
    train_ids.tofile(train_bin_path)
    val_ids.tofile(val_bin_path)

    ratio = (len(train_text) + len(val_text)) / (len(train_ids) + len(val_ids))
    print(f"  ✓ {train_bin_path}: {len(train_ids):,} tokens ({os.path.getsize(train_bin_path)/1e6:.1f} MB)", flush=True)
    print(f"  ✓ {val_bin_path}:   {len(val_ids):,} tokens ({os.path.getsize(val_bin_path)/1e6:.1f} MB)", flush=True)
    print(f"  ✓ Compression: {ratio:.2f} chars/token | encoded in {time.time()-t_enc:.1f}s | total {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
