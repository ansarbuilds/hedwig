"""
Stretch Goal #1 for Hedwig: Mixture-of-Experts (MoE — 8 Experts, Top-2 Routing).

In plain English:
How frontier models (DeepSeek, Mixtral, GPT-4) grow huge without getting slow:
Instead of one single "thinking room" (SwiGLU MLP) inside each layer, we build
8 specialist Expert rooms + a tiny "Traffic Cop" (Router).
For every word, the Traffic Cop picks only the 2 best Experts (Top-2) to wake up
and process that word, while the other 6 Experts sleep!

We use "Upcycling" (how Mixtral & Qwen-MoE are built): we copy the trained attention
and word dictionary from brain_dpo.pt, seed the 8 experts from the trained SwiGLU
weights with slight variations so each expert can specialize, and train with a
load-balancing loss so all 8 experts share the work evenly.

Run:   .venv/bin/python moe.py train
       .venv/bin/python moe.py talk "Write a short story about dragon and castle."
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import tinygpt, sft

# ---- MoE Settings ----
n_experts   = 8         # 8 specialist experts per layer
top_k_exp   = 2         # wake up only the top 2 experts per word
exp_hidden  = 352       # each expert is half the width of the dense 704-wide SwiGLU (so 2 active experts = same compute!)
aux_weight  = 0.02      # load-balancing penalty so the router uses all 8 experts evenly
steps       = 300
lr          = 1e-4
moe_brain   = "brain_moe.pt"

class ExpertSwiGLU(nn.Module):
    """One specialist Expert (half-width SwiGLU so 2 active experts cost the same compute as 1 dense MLP)."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(tinygpt.n_embd, exp_hidden, bias=False)
        self.up   = nn.Linear(tinygpt.n_embd, exp_hidden, bias=False)
        self.down = nn.Linear(exp_hidden, tinygpt.n_embd, bias=False)
        self.drop = nn.Dropout(tinygpt.dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class SparseMoE(nn.Module):
    """8 Experts + a Traffic Cop (Router) that sends each word to its Top-2 Experts."""
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(tinygpt.n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList([ExpertSwiGLU() for _ in range(n_experts)])
        self.last_top2 = None
        self.last_aux_loss = 0.0

    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(-1, C)                                  # (N, C) where N = B * T
        router_logits = self.router(flat_x)                     # (N, 8)
        router_probs  = F.softmax(router_logits, dim=-1)        # (N, 8)
        top_w, top_i  = torch.topk(router_probs, top_k_exp, dim=-1)  # (N, 2)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)         # normalize the 2 winning weights to sum to 1.0

        # Load-balancing loss: encourage the router to send ~1/8 of words to each expert
        usage = F.one_hot(top_i[:, 0], n_experts).float().mean(dim=0)
        mean_prob = router_probs.mean(dim=0)
        self.last_aux_loss = n_experts * (usage * mean_prob).sum()

        # Record which 2 experts handled the most recent token (for the live UI Brain X-Ray!)
        last_ids = top_i[-1].tolist()
        last_wts = top_w[-1].tolist()
        self.last_top2 = [(int(last_ids[0]), round(float(last_wts[0]) * 100)),
                          (int(last_ids[1]), round(float(last_wts[1]) * 100))]

        out = torch.zeros_like(flat_x)
        for slot in range(top_k_exp):
            exp_idx = top_i[:, slot]
            weight  = top_w[:, slot].unsqueeze(-1)
            for e_id, expert in enumerate(self.experts):
                mask = (exp_idx == e_id)
                if mask.any():
                    out[mask] = out[mask] + weight[mask] * expert(flat_x[mask]).to(out.dtype)
        return out.view(B, T, C)

class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.RMSNorm(tinygpt.n_embd)
        self.attn = tinygpt.CausalSelfAttention()
        self.ln2  = nn.RMSNorm(tinygpt.n_embd)
        self.mlp  = SparseMoE()
    def forward(self, x, kv_cache=None, start_pos=0):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        return x + self.mlp(self.ln2(x))

class MoEGPT(nn.Module):
    """Hedwig with 8 Experts per layer (Top-2 active per word)."""
    def __init__(self):
        super().__init__()
        self.tok    = nn.Embedding(tinygpt.vocab_size, tinygpt.n_embd)
        self.blocks = nn.ModuleList([MoEBlock() for _ in range(tinygpt.n_layer)])
        self.ln     = nn.RMSNorm(tinygpt.n_embd)
        self.head   = nn.Linear(tinygpt.n_embd, tinygpt.vocab_size)
        self.last_experts = None

    def forward(self, idx, targets=None, kv_cache=None, start_pos=0):
        x = self.tok(idx)
        aux_total = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=None if kv_cache is None else kv_cache[i], start_pos=start_pos)
            aux_total = aux_total + block.mlp.last_aux_loss
        self.last_experts = self.blocks[-1].mlp.last_top2
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + aux_weight * (aux_total / len(self.blocks))
        return logits, loss

def upcycle_from_dense(moe_model, dense_path="brain_dpo.pt"):
    """Copy trained attention/embeddings from dense Hedwig and slice its SwiGLU into 8 experts."""
    if not os.path.exists(dense_path):
        dense_path = "brain_chat.pt"
    sd = torch.load(dense_path, map_location="cpu")
    with torch.no_grad():
        moe_model.tok.weight.copy_(sd["tok.weight"])
        moe_model.ln.weight.copy_(sd["ln.weight"])
        moe_model.head.weight.copy_(sd["head.weight"])
        moe_model.head.bias.copy_(sd["head.bias"])
        for i, block in enumerate(moe_model.blocks):
            block.ln1.weight.copy_(sd[f"blocks.{i}.ln1.weight"])
            block.ln2.weight.copy_(sd[f"blocks.{i}.ln2.weight"])
            for k in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias"):
                getattr(block.attn, k.split(".")[0])._parameters[k.split(".")[1]].copy_(sd[f"blocks.{i}.attn.{k}"])
            g_full = sd[f"blocks.{i}.mlp.gate.weight"].float()  # (1024, 384)
            u_full = sd[f"blocks.{i}.mlp.up.weight"].float()
            d_full = sd[f"blocks.{i}.mlp.down.weight"].float()  # (384, 1024)
            H_full = g_full.size(0)
            for e_id, exp in enumerate(block.mlp.experts):
                # Give each expert a shifted 352-neuron slice of the 1024-neuron trained MLP
                idx = (torch.arange(exp_hidden) + e_id * (H_full // n_experts)) % H_full
                exp.gate.weight.copy_(g_full[idx] + 0.01 * torch.randn_like(g_full[idx]))
                exp.up.weight.copy_(u_full[idx]   + 0.01 * torch.randn_like(u_full[idx]))
                exp.down.weight.copy_(d_full[:, idx])
            nn.init.normal_(block.mlp.router.weight, std=0.02)

def train():
    tr_x, tr_y, va_x, va_y = sft.load_sft_data()
    model = MoEGPT()
    upcycle_from_dense(model)
    model = model.to(tinygpt.device)
    total_p = sum(p.numel() for p in model.parameters())
    # Active params per token = total minus 6 sleeping experts per layer
    sleep_p = 6 * tinygpt.n_layer * (3 * tinygpt.n_embd * exp_hidden)
    active_p = total_p - sleep_p
    print(f"MoE Brain: {total_p/1e6:.1f}M total knobs ({active_p/1e6:.1f}M active per word, 8 experts top-2)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best_val = float("inf")
    t0 = time.time()

    for step in range(steps + 1):
        if step % 75 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
                vl = sum(model(*sft.get_sft_batch(va_x, va_y))[1].item() for _ in range(15)) / 15
            print(f"MoE step {step:4d} | story score (lower=smarter) {vl:.3f} | top-2 experts {model.last_experts} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best_val:
                best_val = vl
                bf16_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in model.state_dict().items()}
                torch.save(bf16_sd, moe_brain + ".tmp")
                os.replace(moe_brain + ".tmp", moe_brain)
                print(f"  🌟 saved new best {moe_brain} (score {best_val:.3f})", flush=True)
            model.train()

        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            _, loss = model(*sft.get_sft_batch(tr_x, tr_y))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    print(f"Done! Best {moe_brain} score: {best_val:.3f}", flush=True)

def talk():
    model = MoEGPT().to(tinygpt.device)
    model.load_state_dict(torch.load(moe_brain, map_location=tinygpt.device))
    model.eval()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a short story about dragon and castle."
    story, xray = "", []
    for text, steps_info in sft.reply_stream(model, prompt, with_xray=True):
        story, xray = text, steps_info
    print(f"🧑 You: {prompt}\n\n🦉 Hedwig (MoE 8-Experts, Top-2):\n{story}\n")
    sample_exps = [f"{s['piece'].strip() or '␣'}→E{s.get('experts',[(0,0)])[0][0]}/E{s.get('experts',[(0,0),(1,0)])[1][0]}" for s in xray[:12]]
    print("Traffic Cop routing on first 12 words:", " · ".join(sample_exps))

if __name__ == "__main__":
    train() if (sys.argv[1:] or ["train"])[0] == "train" else talk()
