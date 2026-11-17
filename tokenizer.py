"""
Hedwig's tokenizer: Byte Pair Encoding (BPE), written from scratch.
It learns which chunks of text appear together often and glues them into single pieces.
"""
import os, json

def get_stats(ids):
    """Count how often each pair of neighbours appears."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Replace every copy of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    """Learn (vocab_size - 256) merges from the text."""
    ids = list(text.encode("utf-8"))
    merges = {}                                   # (a, b) -> new_id : the "glue rules"
    vocab = {i: bytes([i]) for i in range(256)}   # new_id -> the actual text it stands for

    for new_id in range(256, vocab_size):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        print(f"merge {new_id - 255:3d}: {vocab[new_id]!r:20} seen {stats[pair]:6d} times", flush=True)

    return merges, vocab, ids

def encode(text, merges):
    """Text -> list of piece numbers, using the learned glue rules."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    """List of piece numbers -> text."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

def save(merges, path="tokenizer.json"):
    """Write the glue rules to a file."""
    json.dump([[a, b, new] for (a, b), new in merges.items()], open(path, "w"))

def load(path="tokenizer.json"):
    """Read the glue rules back and rebuild the dictionary."""
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for a, b, new in json.load(open(path)):
        merges[(a, b)] = new
        vocab[new] = vocab[a] + vocab[b]
    return merges, vocab

if __name__ == "__main__":
    if not os.path.exists("tokenizer.json"):
        text = open("data/input.txt").read()
        merges, vocab, ids = train(text, vocab_size=512)
        save(merges)
        print("Saved tokenizer.json!")

    merges, vocab = load()
    sample = "ROMEO:\nBut soft, what light through yonder window breaks?"
    ids = encode(sample, merges)
    print(f"\n{len(sample)} letters -> {len(ids)} pieces")
    print("pieces:", [decode([i], vocab) for i in ids])
    print("round trip OK:", decode(ids, vocab) == sample)