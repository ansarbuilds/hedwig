"""
Hedwig - your very own mini LLM!

It reads Shakespeare as BPE word-pieces and learns to guess the next piece.
Run:   python tinygpt.py train      (teach the brain, saves brain.pt)
       python tinygpt.py talk       (let the brain write something)
"""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import tokenizer

# ---- Settings (the "size" of the brain) ----
block_size = 128   # how many pieces it can remember at once
n_embd     = 256   # how big each "thought" is
n_head     = 4     # how many ways it looks at the text at once
n_layer    = 6     # how many thinking layers stacked up
dropout    = 0.1
batch_size = 64
steps      = 3000
lr         = 1e-3
ckpt_path  = "ckpt.pt"   # the save-point file
ckpt_every = 250         # save every N steps
tok_path   = "tokenizer.json"
cache_path = "data/tokens_512.pt"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_bf16 = device in ("cuda", "mps")  # 16-bit bfloat16 math on GPU (faster, half the activation memory)

# ---- The tokenizer: turn text into numbers (0..vocab_size-1) ----
merges, vocab = tokenizer.load(tok_path)
vocab_size = len(vocab)
encode = lambda s: tokenizer.encode(s, merges)
decode = lambda l: tokenizer.decode(l, vocab)

def load_data():
    """Encode Shakespeare once with BPE and cache it so training starts fast."""
    if os.path.exists(cache_path):
        data = torch.load(cache_path)
    else:
        print("Encoding data/input.txt with BPE (one-time ~30s, caching for future runs)...", flush=True)
        data = torch.tensor(encode(open("data/input.txt").read()), dtype=torch.long)
        torch.save(data, cache_path)
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

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos[:, :, :T], self.sin[:, :, :T])
        k = apply_rope(k, self.cos[:, :, :T], self.sin[:, :, :T])
        # MPS SDPA doesn't support attention dropout, so use 0.0 on mps (residual dropout still applies)
        attn_drop = dropout if (self.training and x.device.type != "mps") else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, is_causal=True)
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
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        x = self.tok(idx)
        logits = self.head(self.ln(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
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
        if step % 250 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                vl = sum(model(*get_batch(val_data))[1].item() for _ in range(20)) / 20
            print(f"step {step:5d} | test score (lower=smarter) {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best_val:
                best_val = vl
                torch.save(model.state_dict(), "brain.pt.tmp")
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
    start = sys.argv[2] if len(sys.argv) > 2 else "ROMEO:"
    idx = torch.tensor([encode(start)], device=device)
    print("🦉 Hedwig says:\n")
    print(decode(model.generate(idx, 500)[0].tolist()))

if __name__ == "__main__":
    train() if (sys.argv[1:] or ["train"])[0] == "train" else talk()
