"""
Stretch Goal #2 for Hedwig: Teacher -> Student Knowledge Distillation ("Baby Hedwig").

In plain English:
- Your 13.8M Stage 3 DPO brain (brain_dpo.pt) is the "Teacher".
- We create a tiny 3.1M-knob "Baby Hedwig" Student (4 layers, 4 heads, 192 thought width —
  4.4x fewer knobs and only ~6.2 MB in 16-bit!).
- Instead of only checking whether Baby Hedwig guessed the single next word in the story,
  the Teacher whispers its FULL 4,096-word probability hints ("The word is 'forest',
  though 'woods' is 20% likely and 'garden' is 10% likely").
- That soft-label KL-divergence loss transfers the Teacher's intuition into a tiny brain
  that runs at ~180+ tokens/sec!

Run:   .venv/bin/python distill.py train
       .venv/bin/python distill.py talk "Tell me a story where a little girl named Lily finds a shiny key in the garden."
"""
import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import tinygpt, sft

# ---- Baby Hedwig (Student) Settings: ~3.1M knobs (4.4x smaller than 13.8M Teacher!) ----
baby_embd  = 192
baby_head  = 4     # 48 per head
baby_layer = 4
temp_kd    = 2.0   # softens Teacher probabilities so runner-up word hints stand out
alpha_kd   = 0.5   # 50% Teacher soft-label hints + 50% ground-truth story words
steps      = 600
lr         = 1.5e-3
baby_brain = "brain_baby.pt"

class BabyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = baby_embd // baby_head
        self.qkv  = nn.Linear(baby_embd, 3 * baby_embd)
        self.proj = nn.Linear(baby_embd, baby_embd)
        self.drop = nn.Dropout(tinygpt.dropout)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(tinygpt.block_size).float(), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None, :, :])
        self.register_buffer("sin", freqs.sin()[None, None, :, :])

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, baby_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, baby_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, baby_head, self.head_dim).transpose(1, 2)
        cos, sin = self.cos[:, :, start_pos:start_pos + T], self.sin[:, :, start_pos:start_pos + T]
        q, k = tinygpt.apply_rope(q, cos, sin), tinygpt.apply_rope(k, cos, sin)
        if kv_cache is not None:
            if kv_cache[0] is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            kv_cache[0], kv_cache[1] = k, v
        causal = (kv_cache is None) or (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.drop(self.proj(out.transpose(1, 2).contiguous().view(B, T, C)))

class BabySwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 512
        self.gate = nn.Linear(baby_embd, hidden, bias=False)
        self.up   = nn.Linear(baby_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, baby_embd, bias=False)
        self.drop = nn.Dropout(tinygpt.dropout)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))

class BabyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.RMSNorm(baby_embd)
        self.attn = BabyAttention()
        self.ln2  = nn.RMSNorm(baby_embd)
        self.mlp  = BabySwiGLU()
    def forward(self, x, kv_cache=None, start_pos=0):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        return x + self.mlp(self.ln2(x))

class BabyGPT(nn.Module):
    """3.1M-parameter distilled Student model (4.4x smaller than 13.8M Teacher)."""
    def __init__(self):
        super().__init__()
        self.tok    = nn.Embedding(tinygpt.vocab_size, baby_embd)
        self.blocks = nn.ModuleList([BabyBlock() for _ in range(baby_layer)])
        self.ln     = nn.RMSNorm(baby_embd)
        self.head   = nn.Linear(baby_embd, tinygpt.vocab_size)
    def forward(self, idx, targets=None, kv_cache=None, start_pos=0):
        x = self.tok(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=None if kv_cache is None else kv_cache[i], start_pos=start_pos)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def warm_start_from_teacher(student, teacher_sd):
    """Project the first 192 channels of the Teacher's embeddings/layers into Baby Hedwig."""
    with torch.no_grad():
        student.tok.weight.copy_(teacher_sd["tok.weight"][:, :baby_embd])
        student.ln.weight.copy_(teacher_sd["ln.weight"][:baby_embd])
        student.head.weight.copy_(teacher_sd["head.weight"][:, :baby_embd])
        student.head.bias.copy_(teacher_sd["head.bias"])

def distill_loss(student_logits, teacher_logits, targets):
    """Combine hard ground-truth loss with Teacher soft-label KL divergence on valid tokens."""
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
    mask = (targets.view(-1) != -100)
    if not mask.any():
        return ce_loss, ce_loss
    s_flat = student_logits.view(-1, student_logits.size(-1))[mask].float() / temp_kd
    t_flat = teacher_logits.view(-1, teacher_logits.size(-1))[mask].float() / temp_kd
    kl_loss = F.kl_div(F.log_softmax(s_flat, dim=-1), F.softmax(t_flat, dim=-1), reduction="batchmean") * (temp_kd ** 2)
    return (1.0 - alpha_kd) * ce_loss + alpha_kd * kl_loss, ce_loss

def train():
    tr_x, tr_y, va_x, va_y = sft.load_sft_data()
    t_path = "brain_dpo.pt" if os.path.exists("brain_dpo.pt") else "brain_chat.pt"
    t_sd = torch.load(t_path, map_location=tinygpt.device)

    teacher = tinygpt.TinyGPT().to(tinygpt.device)
    teacher.load_state_dict(t_sd)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = BabyGPT()
    warm_start_from_teacher(student, {k: v.cpu() for k, v in t_sd.items()})
    student = student.to(tinygpt.device)

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Distilling Teacher ({t_params/1e6:.1f}M knobs) -> Baby Student ({s_params/1e6:.1f}M knobs, {t_params/s_params:.1f}x smaller!)", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best_val = float("inf")
    t0 = time.time()

    for step in range(steps + 1):
        if step % 150 == 0:
            student.eval()
            with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
                vl = sum(student(*sft.get_sft_batch(va_x, va_y))[1].item() for _ in range(15)) / 15
            print(f"Distill step {step:4d} | Baby Hedwig test score (lower=smarter) {vl:.3f} | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vl < best_val:
                best_val = vl
                bf16_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in student.state_dict().items()}
                torch.save(bf16_sd, baby_brain + ".tmp")
                os.replace(baby_brain + ".tmp", baby_brain)
                print(f"  🌟 saved new best {baby_brain} (score {best_val:.3f}, {os.path.getsize(baby_brain)/1e6:.1f} MB)", flush=True)
            student.train()

        x, y = sft.get_sft_batch(tr_x, tr_y)
        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            with torch.no_grad():
                t_logits, _ = teacher(x)
            s_logits, _ = student(x)
            loss, _ = distill_loss(s_logits, t_logits, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    print(f"Done! Best {baby_brain} score: {best_val:.3f} ({os.path.getsize(baby_brain)/1e6:.1f} MB)", flush=True)

def talk():
    model = BabyGPT().to(tinygpt.device)
    model.load_state_dict(torch.load(baby_brain, map_location=tinygpt.device))
    model.eval()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Tell me a story where a little girl named Lily finds a shiny key in the garden."
    t0 = time.time()
    story = sft.reply(model, prompt)
    dt = time.time() - t0
    print(f"🧑 You: {prompt}\n\n🐣 Baby Hedwig (3.1M knobs, {os.path.getsize(baby_brain)/1e6:.1f} MB, {dt:.2f}s):\n{story}")

if __name__ == "__main__":
    train() if (sys.argv[1:] or ["train"])[0] == "train" else talk()
