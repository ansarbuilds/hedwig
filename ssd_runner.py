"""
Hedwig from the SSD — "LLM in a Flash" / AirLLM style.

A normal run keeps every thinking layer in RAM at once. This script chops
brain_dpo.pt into one file per layer. While writing a story it loads a single
layer, runs it, then loads the next one over the same slot. The bookmark of
past words (the KV cache) stays in RAM, because each new word still needs it.

Why bother on a 28 MB brain? It barely saves RAM — the lesson is the tradeoff.
The same trick is what lets a phone run a model bigger than its memory.

Run:
  .venv/bin/python ssd_runner.py            # split (if needed) + compare RAM vs SSD
  .venv/bin/python ssd_runner.py split      # only write ssd_layers/
"""
import json, os, resource, subprocess, sys, time
import torch
import torch.nn as nn
import tinygpt

# ---- Settings ----
brain_path = "brain_dpo.pt"          # Stage 3 brain we stream from disk
shard_dir  = "ssd_layers"            # one small file per thinking layer
prompt     = "Write a short story about a dragon and a castle."
n_new      = 80                      # new word-pieces to write (same for both runs)
temperature = 0.75
top_k      = 40
rep_penalty = 1.15
seed       = 7                       # same dice roll so both stories can be compared
device = tinygpt.device

def rss_peak_mb():
    """High-water mark of this process, in MB. macOS reports bytes; Linux reports KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024

def weight_mb(module):
    """How many megabytes of weights + buffers this module is holding right now."""
    n = sum(p.numel() * p.element_size() for p in module.parameters())
    n += sum(b.numel() * b.element_size() for b in module.buffers())
    return n / (1024 * 1024)

def sync():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

def split_brain():
    """Cut the brain into a small 'always in RAM' piece plus one file per layer."""
    if not os.path.exists(brain_path):
        raise SystemExit(f"Missing {brain_path}. Train it with: .venv/bin/python dpo.py train")
    os.makedirs(shard_dir, exist_ok=True)
    sd = torch.load(brain_path, map_location="cpu", weights_only=True)
    resident, layers = {}, {i: {} for i in range(tinygpt.n_layer)}
    for key, value in sd.items():
        if key.startswith("blocks."):
            _, index, rest = key.split(".", 2)
            layers[int(index)][rest] = value
        else:
            resident[key] = value
    # The word-table and the final 'what word next?' layer stay resident.
    # Each thinking layer is its own file. Write to .tmp, then rename, so a
    # crash never leaves a half-written shard.
    blobs = [("resident.pt", resident)] + [(f"block_{i}.pt", layers[i]) for i in range(tinygpt.n_layer)]
    print(f"Splitting {brain_path} into {shard_dir}/ ({len(sd)} tensors)")
    total = 0
    for name, part in blobs:
        if not part:
            raise SystemExit(f"Refusing to write an empty shard: {name}")
        path = os.path.join(shard_dir, name)
        torch.save(part, path + ".tmp")
        os.replace(path + ".tmp", path)
        size = os.path.getsize(path)
        total += size
        print(f"  {name:16s} {size/1e6:6.2f} MB  ({len(part)} tensors)")
    print(f"  {'total':16s} {total/1e6:6.2f} MB")
    return total

def shards_ready():
    need = ["resident.pt"] + [f"block_{i}.pt" for i in range(tinygpt.n_layer)]
    paths = [os.path.join(shard_dir, name) for name in need]
    if not all(os.path.exists(p) for p in paths):
        return False
    if os.path.exists(brain_path) and os.path.getmtime(paths[0]) < os.path.getmtime(brain_path):
        return False
    return True

class LayerStream(nn.Module):
    """One rented desk for a thinking layer. The other layers live as files."""
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(tinygpt.vocab_size, tinygpt.n_embd)
        self.ln = nn.RMSNorm(tinygpt.n_embd)
        self.head = nn.Linear(tinygpt.n_embd, tinygpt.vocab_size)
        self.block = tinygpt.Block()
        self.paths = [os.path.join(shard_dir, f"block_{i}.pt") for i in range(tinygpt.n_layer)]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.bytes_read = 0
        resident = torch.load(os.path.join(shard_dir, "resident.pt"), map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(resident, strict=False)
        bad = [k for k in missing if not k.startswith("block.")]
        if bad or unexpected:
            raise RuntimeError(f"resident shard did not match the model: missing {bad}, extra {unexpected}")
        self.to(device)
        self.eval()

    def forward(self, idx, kv_cache=None, start_pos=0):
        x = self.tok(idx)
        for i, path in enumerate(self.paths):
            # Drop the previous layer's file and slide this one into the same block.
            layer = torch.load(path, map_location=device, weights_only=True)
            self.block.load_state_dict(layer)
            self.bytes_read += self.sizes[i]
            cache = None if kv_cache is None else kv_cache[i]
            x = self.block(x, kv_cache=cache, start_pos=start_pos)
            del layer
        return self.head(self.ln(x)), None

def load_ram_model():
    model = tinygpt.TinyGPT().to(device)
    model.load_state_dict(torch.load(brain_path, map_location=device, weights_only=True))
    model.eval()
    return model

@torch.no_grad()
def generate(model, n_new_tokens):
    """Write a fixed number of word-pieces. Same loop the normal brain uses."""
    prefix = f"User: {prompt.strip()}\nHedwig: "
    idx = torch.tensor([tinygpt.encode(prefix)], device=device)
    start_len = idx.size(1)
    kv_cache = [[None, None] for _ in range(tinygpt.n_layer)]
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    sync()
    t0 = time.perf_counter()
    for step in range(n_new_tokens):
        pos = start_len + step - 1
        if step == 0:
            logits, _ = model(idx[:, -tinygpt.block_size:], kv_cache=kv_cache, start_pos=0)
        elif pos < tinygpt.block_size:
            logits, _ = model(idx[:, -1:], kv_cache=kv_cache, start_pos=pos)
        else:
            logits, _ = model(idx[:, -tinygpt.block_size:])
        recent = idx[0, start_len:].tolist()
        nxt = tinygpt.sample_next(logits, recent, temperature, top_k, rep_penalty)
        idx = torch.cat([idx, nxt], dim=1)
    sync()
    elapsed = time.perf_counter() - t0
    story = tinygpt.decode(idx[0, start_len:].tolist())
    return story, elapsed

def run_mode(mode):
    if mode == "ssd":
        model = LayerStream()
        # Resident weights = word table + final layer + the ONE rented block.
        held = weight_mb(model)
    else:
        model = load_ram_model()
        held = weight_mb(model)
    story, elapsed = generate(model, n_new)
    tok_s = n_new / elapsed if elapsed > 0 else 0.0
    mps_mb = None
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
    report = {
        "mode": mode,
        "device": device,
        "seconds": round(elapsed, 3),
        "tok_s": round(tok_s, 2),
        "weight_mb": round(held, 2),
        "rss_peak_mb": round(rss_peak_mb(), 1),
        "mps_mb": None if mps_mb is None else round(mps_mb, 1),
        "disk_mb_read": round(getattr(model, "bytes_read", 0) / 1e6, 1),
        "story": story,
    }
    print(json.dumps(report))

def bench():
    if not shards_ready():
        split_brain()
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")
    print(f"Writing {n_new} word-pieces, seed {seed}, two fresh processes so peak RAM is honest.\n")
    reports = []
    for mode in ("ram", "ssd"):
        print(f"--- running {mode} ---", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "run", mode],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            raise SystemExit(f"{mode} run failed (exit {proc.returncode})")
        # The child prints exactly one JSON line. Anything else is a warning.
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        report = json.loads(line)
        reports.append(report)
        if proc.stderr.strip():
            print(proc.stderr.strip())
    ram, ssd = reports
    print("\nSame story, two ways of holding the brain:\n")
    print(f"{'':22s} {'all in RAM':>14s} {'one layer from SSD':>20s}")
    print(f"{'weights held (MB)':22s} {ram['weight_mb']:14.2f} {ssd['weight_mb']:20.2f}")
    print(f"{'peak process RAM (MB)':22s} {ram['rss_peak_mb']:14.1f} {ssd['rss_peak_mb']:20.1f}")
    if ram["mps_mb"] is not None:
        print(f"{'MPS tensors (MB)':22s} {ram['mps_mb']:14.1f} {ssd['mps_mb']:20.1f}")
    print(f"{'time for {0} tokens (s)'.format(n_new):22s} {ram['seconds']:14.3f} {ssd['seconds']:20.3f}")
    print(f"{'word-pieces / sec':22s} {ram['tok_s']:14.2f} {ssd['tok_s']:20.2f}")
    print(f"{'read off SSD (MB)':22s} {'0':>14s} {ssd['disk_mb_read']:20.1f}")
    print("\n🦉 in-RAM story:\n" + ram["story"])
    print("\n🦉 SSD story:\n" + ssd["story"])
    same = ram["story"] == ssd["story"]
    print(f"\nStories match: {same}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "split":
        split_brain()
    elif cmd == "run":
        run_mode(sys.argv[2])
    elif cmd == "bench":
        bench()
    else:
        raise SystemExit("Use: ssd_runner.py [bench|split|run ram|run ssd]")
