"""
Speed test: Apple's MLX versus the PyTorch brain Hedwig already trains with.

Same brain on purpose: 6 layers, 384 wide, 6 heads, 4,096 word-pieces.
We time two chores:
  1. One practice step (read 32 snippets, guess the next pieces, adjust the knobs).
  2. Writing 80 new word-pieces, one at a time, remembering the earlier ones (KV cache).

PyTorch stays the real brain. This script only measures.

Run:  .venv/bin/python mlx_bench.py
"""
import time
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_opt
import tinygpt

# ---- Settings (same size as tinygpt.py) ----
block_size = tinygpt.block_size
n_embd = tinygpt.n_embd
n_head = tinygpt.n_head
n_layer = tinygpt.n_layer
dropout = tinygpt.dropout
batch_size = tinygpt.batch_size
lr = tinygpt.lr
vocab_size = tinygpt.vocab_size
head_dim = n_embd // n_head
hidden = ((int(8 * n_embd / 3) + 63) // 64) * 64
warmup_steps = 2
timed_steps = 5
n_new = 80
prompt = "Once upon a time, a little owl "
brain_path = "brain.pt"

def sync_torch():
    if tinygpt.device == "mps":
        torch.mps.synchronize()
    elif tinygpt.device == "cuda":
        torch.cuda.synchronize()

def tree_nbytes(params):
    if isinstance(params, dict):
        return sum(tree_nbytes(v) for v in params.values())
    if isinstance(params, list):
        return sum(tree_nbytes(v) for v in params)
    return params.nbytes

# ---- The same brain, written in MLX ----
class Attention(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = mlx_nn.Linear(n_embd, 3 * n_embd)
        self.proj = mlx_nn.Linear(n_embd, n_embd)
        self.rope = mlx_nn.RoPE(head_dim, traditional=False)
        self.drop = mlx_nn.Dropout(dropout)

    def __call__(self, x, cache=None, offset=0):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(B, T, n_embd)
        return self.drop(self.proj(out)), (k, v)

class SwiGLU(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = mlx_nn.Linear(n_embd, hidden, bias=False)
        self.up = mlx_nn.Linear(n_embd, hidden, bias=False)
        self.down = mlx_nn.Linear(hidden, n_embd, bias=False)
        self.drop = mlx_nn.Dropout(dropout)

    def __call__(self, x):
        return self.drop(self.down(mlx_nn.silu(self.gate(x)) * self.up(x)))

class Block(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = mlx_nn.RMSNorm(n_embd)
        self.attn = Attention()
        self.ln2 = mlx_nn.RMSNorm(n_embd)
        self.mlp = SwiGLU()

    def __call__(self, x, cache=None, offset=0):
        a, new_cache = self.attn(self.ln1(x), cache=cache, offset=offset)
        x = x + a
        return x + self.mlp(self.ln2(x)), new_cache

class HedwigMLX(mlx_nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = mlx_nn.Embedding(vocab_size, n_embd)
        self.blocks = [Block() for _ in range(n_layer)]
        self.ln = mlx_nn.RMSNorm(n_embd)
        self.head = mlx_nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, cache=None, offset=0):
        x = self.tok(idx)
        new_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[i]
            x, layer_new = block(x, cache=layer_cache, offset=offset)
            new_cache.append(layer_new)
        return self.head(self.ln(x)), new_cache

def as_mx(tensor):
    return mx.array(tensor.detach().float().cpu().numpy())

def load_brain_into_mlx(model, path):
    """Copy the saved PyTorch knobs into the MLX brain. Same numbers, different toolbox."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.tok.weight = as_mx(sd["tok.weight"])
    model.ln.weight = as_mx(sd["ln.weight"])
    model.head.weight = as_mx(sd["head.weight"])
    model.head.bias = as_mx(sd["head.bias"])
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        block.ln1.weight = as_mx(sd[p + "ln1.weight"])
        block.attn.qkv.weight = as_mx(sd[p + "attn.qkv.weight"])
        block.attn.qkv.bias = as_mx(sd[p + "attn.qkv.bias"])
        block.attn.proj.weight = as_mx(sd[p + "attn.proj.weight"])
        block.attn.proj.bias = as_mx(sd[p + "attn.proj.bias"])
        block.ln2.weight = as_mx(sd[p + "ln2.weight"])
        block.mlp.gate.weight = as_mx(sd[p + "mlp.gate.weight"])
        block.mlp.up.weight = as_mx(sd[p + "mlp.up.weight"])
        block.mlp.down.weight = as_mx(sd[p + "mlp.down.weight"])
    mx.eval(model.parameters())

def cast_mlx(model, dtype):
    def rec(tree):
        if isinstance(tree, dict):
            return {k: rec(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rec(v) for v in tree]
        return tree.astype(dtype)
    model.update(rec(model.parameters()))
    mx.eval(model.parameters())

def one_batch():
    """32 real story snippets, reused so the timer is the brain and not the disk."""
    if __import__("os").path.exists(tinygpt.train_bin):
        data = np.fromfile(tinygpt.train_bin, dtype=np.uint16).astype(np.int64)
    else:
        data = np.random.randint(0, vocab_size, size=100_000, dtype=np.int64)
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y

def time_pytorch_train(x_np, y_np):
    model = tinygpt.TinyGPT().to(tinygpt.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x = torch.tensor(x_np, device=tinygpt.device)
    y = torch.tensor(y_np, device=tinygpt.device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup_steps):
        step()
    sync_torch()
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        loss = step()
    sync_torch()
    ms = (time.perf_counter() - t0) * 1000 / timed_steps
    return ms, float(loss.detach().float())

def time_mlx_train(x_np, y_np, dtype):
    model = HedwigMLX()
    model.train()
    if dtype != mx.float32:
        cast_mlx(model, dtype)
    opt = mlx_opt.AdamW(learning_rate=lr)
    x = mx.array(x_np)
    y = mx.array(y_np)

    def loss_fn(m, xb, yb):
        logits, _ = m(xb)
        return mx.mean(mlx_nn.losses.cross_entropy(logits, yb))

    grad_fn = mlx_nn.value_and_grad(model, loss_fn)

    @mx.compile
    def step(xb, yb):
        loss, grads = grad_fn(model, xb, yb)
        opt.update(model, grads)
        return loss

    for _ in range(warmup_steps):
        mx.eval(step(x, y))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        loss = step(x, y)
        mx.eval(loss)
    mx.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / timed_steps
    return ms, float(loss)

@torch.no_grad()
def generate_torch(model, ids, n):
    model.eval()
    idx = torch.tensor([ids], device=tinygpt.device)
    cache = [[None, None] for _ in range(n_layer)]
    T0 = idx.size(1)
    logits, _ = model(idx, kv_cache=cache, start_pos=0)
    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([idx, nxt], dim=1)
    sync_torch()
    t0 = time.perf_counter()
    for step in range(1, n):
        pos = T0 + step - 1
        logits, _ = model(idx[:, -1:], kv_cache=cache, start_pos=pos)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    sync_torch()
    # The clock starts after the prompt is read, so this is "new pieces per second".
    elapsed = time.perf_counter() - t0
    return idx[0, T0:].tolist(), (n - 1) / elapsed

def generate_mlx(model, ids, n):
    model.eval()
    idx = mx.array([ids])
    logits, cache = model(idx, cache=None, offset=0)
    mx.eval(logits)
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    out = [int(nxt.item())]
    T0 = len(ids)
    token = nxt.reshape(1, 1)
    mx.synchronize()
    t0 = time.perf_counter()
    for step in range(1, n):
        logits, cache = model(token, cache=cache, offset=T0 + step - 1)
        mx.eval(logits)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(int(nxt.item()))
        token = nxt.reshape(1, 1)
    mx.synchronize()
    return out, (n - 1) / elapsed_from(t0)

def elapsed_from(t0):
    return time.perf_counter() - t0

def main():
    knobs = sum(p.numel() for p in tinygpt.TinyGPT().parameters())
    mlx_model = HedwigMLX()
    mlx_knobs = tree_nbytes(mlx_model.parameters()) // 4  # float32, 4 bytes each, at init
    # tree_nbytes counts bytes; parameters are float32 so divide by 4 for a knob count.
    print(f"PyTorch knobs: {knobs:,}    MLX knobs: {mlx_knobs:,}    device: {tinygpt.device} / MLX gpu")
    if knobs != mlx_knobs:
        raise SystemExit("The two brains do not have the same number of knobs. Refusing to time them.")

    print(f"\nPractice step: batch {batch_size}, {block_size} pieces, {warmup_steps} warmups then {timed_steps} timed steps.")
    x_np, y_np = one_batch()
    pt_ms, pt_loss = time_pytorch_train(x_np, y_np)
    print(f"  PyTorch mps, 16-bit math, 32-bit knobs: {pt_ms:.0f} ms/step  (loss {pt_loss:.3f})")
    mx_ms, mx_loss = time_mlx_train(x_np, y_np, mx.float32)
    print(f"  MLX 32-bit:                            {mx_ms:.0f} ms/step  (loss {mx_loss:.3f})")
    try:
        mx16_ms, mx16_loss = time_mlx_train(x_np, y_np, mx.bfloat16)
        print(f"  MLX 16-bit (bfloat16):                 {mx16_ms:.0f} ms/step  (loss {mx16_loss:.3f})")
    except Exception as exc:
        mx16_ms = None
        print(f"  MLX 16-bit skipped: {exc}")

    print("\nChecking the MLX brain against the saved PyTorch brain (most-likely next piece)...")
    pt = tinygpt.TinyGPT().to(tinygpt.device)
    pt.load_state_dict(torch.load(brain_path, map_location=tinygpt.device, weights_only=True))
    pt.eval()
    mx_model = HedwigMLX()
    load_brain_into_mlx(mx_model, brain_path)
    mx_model.eval()
    ids = tinygpt.encode(prompt)
    with torch.no_grad():
        logits_pt, _ = pt(torch.tensor([ids], device=tinygpt.device))
    logits_mx, _ = mx_model(mx.array([ids]))
    mx.eval(logits_mx)
    diff = (torch.tensor(np.array(logits_mx)) - logits_pt.float().cpu()).abs().max().item()
    print(f"  Biggest logit difference on the prompt: {diff:.5f}")

    for _ in range(1):
        generate_torch(pt, ids, 8)
        generate_mlx(mx_model, ids, 8)
    story_pt, tok_pt = generate_torch(pt, ids, n_new)
    story_mx, tok_mx = generate_mlx(mx_model, ids, n_new)
    text_pt = tinygpt.decode(story_pt)
    text_mx = tinygpt.decode(story_mx)
    same = story_pt == story_mx

    mx16_tok = None
    if mx16_ms is not None:
        mx16 = HedwigMLX()
        load_brain_into_mlx(mx16, brain_path)
        cast_mlx(mx16, mx.bfloat16)
        generate_mlx(mx16, ids, 8)
        _, mx16_tok = generate_mlx(mx16, ids, n_new)

    print("\nSame chore, two toolboxes:\n")
    print(f"{'':28s} {'PyTorch mps':>14s} {'MLX 32-bit':>14s} {'MLX 16-bit':>14s}")
    print(f"{'practice step (ms)':28s} {pt_ms:14.0f} {mx_ms:14.0f} {('—' if mx16_ms is None else f'{mx16_ms:.0f}'):>14s}")
    print(f"{'new word-pieces / sec':28s} {tok_pt:14.1f} {tok_mx:14.1f} {('—' if mx16_tok is None else f'{mx16_tok:.1f}'):>14s}")
    print(f"\nStories match piece for piece: {same}")
    print("\n🦉 PyTorch:\n" + text_pt[:400])
    print("\n🦉 MLX:\n" + text_mx[:400])

if __name__ == "__main__":
    main()
