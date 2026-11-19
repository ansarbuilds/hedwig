"""
Hedwig's tokenizer: Byte Pair Encoding (BPE), written from scratch (GPT-2 / minbpe style).
It splits text into word-sized chunks first, then learns which byte pairs appear together
most often and glues them into single pieces (up to vocab_size = 4096).
"""
import os, re, json
from collections import Counter, defaultdict

# GPT-2 style pre-split pattern so BPE never glues punctuation across separate words
PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+")

def get_stats(ids):
    """Count how often each pair of neighbours appears in a list of IDs."""
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

def train(text, vocab_size=4096, verbose=True, pat=None):
    """Learn (vocab_size - 256) merges fast by counting unique word chunks."""
    finder = pat or PAT
    words = [[list(w.encode("utf-8")), c] for w, c in Counter(finder.findall(text)).items()]
    stats, where = defaultdict(int), defaultdict(set)
    for idx, (ids, c) in enumerate(words):
        for p in zip(ids, ids[1:]):
            stats[p] += c
            where[p].add(idx)

    merges = {}
    vocab = {i: bytes([i]) for i in range(256)}

    for new_id in range(256, vocab_size):
        if not stats:
            break
        pair = max(stats, key=stats.get)
        count = stats[pair]
        if count < 2:
            break
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose and ((new_id - 255) <= 10 or (new_id - 255) % 500 == 0 or new_id == vocab_size - 1):
            print(f"merge {new_id - 255:4d}: {vocab[new_id]!r:20} seen {count:7d} times", flush=True)

        for idx in list(where.pop(pair)):
            ids, c = words[idx]
            for p in zip(ids, ids[1:]):
                stats[p] -= c
                if stats[p] <= 0:
                    stats.pop(p, None)
            out = merge(ids, pair, new_id)
            words[idx][0] = out
            for p in zip(out, out[1:]):
                stats[p] += c
                where[p].add(idx)

    return merges, vocab

def _encode_chunk(chunk_bytes, merges):
    """Apply learned glue rules to one short word chunk."""
    ids = list(chunk_bytes)
    while len(ids) >= 2:
        pair = min(zip(ids, ids[1:]), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def encode(text, merges, cache=None, pat=None):
    """Text -> list of piece numbers, caching each unique word chunk for speed."""
    finder = pat or PAT
    if cache is None:
        cache = {}
    out = []
    for chunk in finder.findall(text):
        ids = cache.get(chunk)
        if ids is None:
            ids = cache[chunk] = _encode_chunk(chunk.encode("utf-8"), merges)
        out.extend(ids)
    return out

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
        merges, vocab = train(text, vocab_size=512)
        save(merges)
        print("Saved tokenizer.json!")

    merges, vocab = load()
    sample = "Once upon a time, a little owl named Hedwig found a shiny golden key."
    ids = encode(sample, merges)
    print(f"\nVocab size: {len(vocab)} | {len(sample)} letters -> {len(ids)} pieces ({len(sample)/len(ids):.2f}x)")
    print("pieces:", [decode([i], vocab) for i in ids])
    print("round trip OK:", decode(ids, vocab) == sample)
