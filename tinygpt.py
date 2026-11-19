"""
Hedwig - your very own mini LLM!

It reads TinyStories + classic novels as 4,096 BPE word-pieces and learns to write stories.
Run:   python prepare_data.py       (download stories, train 4k BPE, build uint16 .bin shards)
       python tinygpt.py train      (teach the brain, saves brain.pt)
       python tinygpt.py talk       (let the brain write a story)
"""
import os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import tokenizer

# ---- Settings (the "size" of the brain: ~13.8M knobs) ----
block_size = 256   # how many pieces it can remember at once (~1,000 letters of story!)
n_embd     = 384   # how big each "thought" is
n_head     = 6     # how many ways it looks at the text at once (64 per head)
n_layer    = 6     # how many thinking layers stacked up
dropout    = 0.05
batch_size = 32
steps      = 3000
lr         = 1e-3
ckpt_path  = "ckpt.pt"   # the save-point file
ckpt_every = 250         # save every N steps
tok_path   = "tokenizer.json"
train_bin  = "data/train.bin"
val_bin    = "data/val.bin"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_bf16 = device in ("cuda", "mps")  # 16-bit bfloat16 math on GPU (faster, half the activation memory)

# ---- The tokenizer: turn text into numbers (0..vocab_size-1) ----
merges, vocab = tokenizer.load(tok_path)
vocab_size = len(vocab)
encode = lambda s: tokenizer.encode(s, merges)
decode = lambda l: tokenizer.decode(l, vocab)

def load_data():
    """Load uint16 token shards built by prepare_data.py (or fall back to data/input.txt)."""
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        tr = torch.from_numpy(np.fromfile(train_bin, dtype=np.uint16).astype(np.int64))
        va = torch.from_numpy(np.fromfile(val_bin,   dtype=np.uint16).astype(np.int64))
        return tr, va
    data = torch.tensor(encode(open("data/input.txt").read()), dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:]

def get_batch(d):
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- The brain ----
def apply_rope(x, cos, sin):
    """Rotate pairs of features by position angle so attention sees relative distance."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE (no peeking ahead)."""
    def __init__(self):
        super().__init__()
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(block_size).float(), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)
        cos = self.cos[:, :, start_pos:start_pos + T]
        sin = self.sin[:, :, start_pos:start_pos + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if kv_cache is not None:
            if kv_cache[0] is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[0], kv_cache[1] = k, v
        # MPS SDPA doesn't support attention dropout, so use 0.0 on mps (residual dropout still applies)
        attn_drop = dropout if (self.training and x.device.type != "mps") else 0.0
        causal = (kv_cache is None) or (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))

class SwiGLU(nn.Module):
    """Gated MLP (Llama style): one path acts as a learned dimmer switch for the other."""
    def __init__(self):
        super().__init__()
        hidden = ((int(8 * n_embd / 3) + 63) // 64) * 64  # ~2/3 of 4*n_embd to match 2-layer knob count
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up   = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = SwiGLU()
    def forward(self, x, kv_cache=None, start_pos=0):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        return x + self.mlp(self.ln2(x))

def sample_next(logits, recent_ids=None, temperature=0.75, top_k=40, rep_penalty=1.15, return_top5=False):
    """Pick the next word-piece using repetition penalty (no loops) + top-k filtering (no weird typos)."""
    logits = logits[:, -1, :].float()
    if recent_ids and rep_penalty > 1.0:
        for tok_id in set(recent_ids[-64:]):
            logits[0, tok_id] = logits[0, tok_id] / rep_penalty if logits[0, tok_id] > 0 else logits[0, tok_id] * rep_penalty
    logits = logits / max(temperature, 1e-4)
    if top_k and top_k < logits.size(-1):
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, 1)
    if not return_top5:
        return nxt
    top_p, top_i = torch.topk(probs[0], 5)
    top5 = [(decode([ int(i) ]), round(float(p) * 100.0, 1)) for p, i in zip(top_p, top_i)]
    return nxt, top5

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None, kv_cache=None, start_pos=0):
        x = self.tok(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=None if kv_cache is None else kv_cache[i], start_pos=start_pos)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n, temperature=0.75, top_k=40, rep_penalty=1.15):
        kv_cache = [[None, None] for _ in range(n_layer)]
        T0 = idx.size(1)
        logits, _ = self(idx[:, -block_size:], kv_cache=kv_cache, start_pos=0)
        nxt = sample_next(logits, idx[0].tolist(), temperature, top_k, rep_penalty)
        idx = torch.cat([idx, nxt], dim=1)
        for step in range(1, n):
            pos = T0 + step - 1
            if pos < block_size:
                logits, _ = self(idx[:, -1:], kv_cache=kv_cache, start_pos=pos)
            else:
                logits, _ = self(idx[:, -block_size:])
            nxt = sample_next(logits, idx[0].tolist(), temperature, top_k, rep_penalty)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

def train():
    train_data, val_data = load_data()
    model = TinyGPT().to(device)
    print(f"Brain size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M knobs, on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    # ---- Resume if a save point exists ----
    start_step, best_val = 0, float("inf")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resuming from step {start_step} (best score so far: {best_val:.3f})")

    stop_step = int(sys.argv[2]) if len(sys.argv) > 2 else steps
    t0 = time.time()
    for step in range(start_step, stop_step + 1):
        if step % ckpt_every == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                vl = sum(model(*get_batch(val_data))[1].item() for _ in range(20)) / 20
            print(f"step {step:5d} | test score (lower=smarter) {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best_val:
                best_val = vl
                bf16_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
                torch.save(bf16_sd, "brain.pt.tmp")
                os.replace("brain.pt.tmp", "brain.pt")
                print(f"  🌟 saved new best brain.pt (score {best_val:.3f})", flush=True)
            model.train()

        # ---- Save point ----
        if step > start_step and step % ckpt_every == 0:
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best_val": best_val}
            torch.save(ckpt, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"  💾 saved checkpoint at step {step}", flush=True)

        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(*get_batch(train_data))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    print(f"Done! Best brain.pt score: {best_val:.3f}", flush=True)

def talk():
    model = TinyGPT().to(device)
    model.load_state_dict(torch.load("brain.pt", map_location=device))
    model.eval()
    start = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time, a little owl "
    idx = torch.tensor([encode(start)], device=device)
    print("🦉 Hedwig says:\n")
    print(decode(model.generate(idx, 200)[0].tolist()))

if __name__ == "__main__":
    train() if (sys.argv[1:] or ["train"])[0] == "train" else talk()
