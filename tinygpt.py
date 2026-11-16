"""
Hedwig - your very own mini LLM!

It reads Shakespeare one letter at a time and learns to guess the next letter.
Run:   python tinygpt.py train      (teach the brain, saves brain.pt)
       python tinygpt.py talk       (let the brain write something)
"""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F

# ---- Settings (the "size" of the brain) ----
block_size = 128   # how many letters it can remember at once
n_embd     = 256   # how big each "thought" is
n_head     = 4     # how many ways it looks at the text at once
n_layer    = 6     # how many thinking layers stacked up
dropout    = 0.1
batch_size = 64
steps      = 3000
lr         = 1e-3
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ---- The book: turn letters into numbers ----
text = open("data/input.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
data = torch.tensor(encode(text))
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

def get_batch(d):
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---- The brain ----
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)  # no peeking ahead!
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(chars), n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, len(chars))
    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
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
    model = TinyGPT().to(device)
    print(f"Brain size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M knobs, on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps + 1):
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*get_batch(val_data))[1].item() for _ in range(20)) / 20
            print(f"step {step:5d} | test score (lower=smarter) {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            model.train()
        _, loss = model(*get_batch(train_data))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.save(model.state_dict(), "brain.pt")
    print("Saved brain.pt!")

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
