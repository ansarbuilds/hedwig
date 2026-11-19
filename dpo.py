"""
Stage 3 for Hedwig: Direct Preference Optimization (DPO) — teaching "taste & judgment".

Why Stage 3 exists:
- Stage 1 (tinygpt.py) taught Hedwig English and how stories flow (Pretraining).
- Stage 2 (sft.py) taught Hedwig to answer "User: ... -> Hedwig: ..." requests (SFT).
- Still, a Stage 2 model only ever saw good examples—it was never penalized for two classic
  small-LLM bad habits:
    1. Going off-topic (you ask for an OWL story and it writes about a DOG).
    2. Repeating the same sentence or phrase in a loop.

How DPO fixes this (in plain English):
For each user prompt, we show Hedwig TWO candidate stories side-by-side:
  👍 Chosen (Thumbs-Up):   The real story that matches the prompt cleanly.
  👎 Rejected (Thumbs-Down): Either an off-topic story from a different prompt, or a story
                             with clumsy sentence/phrase repetition.
We keep a frozen copy of brain_chat.pt as an anchor ("ref") so Hedwig doesn't forget English,
and nudge the active brain ("policy") to score 👍 higher and 👎 lower!

Run:   .venv/bin/python dpo.py train                                    (train brain_chat.pt -> brain_dpo.pt)
       .venv/bin/python dpo.py talk "Write a short story about owl and tree."
"""
import os, sys, time, random
import torch, torch.nn.functional as F
import tinygpt, sft

# ---- DPO Settings ----
steps      = 150        # DPO only needs ~150 gentle preference steps
lr         = 2e-5       # very gentle knob turns so we refine taste without breaking grammar
beta       = 0.5        # how strongly we anchor to the frozen Stage 2 reference brain
nll_weight = 0.5        # fluency anchor on 👍 chosen story (prevents DPO "keyword-stuffing" reward hacking!)
batch_size = 16         # 16 (prompt, 👍 chosen, 👎 rejected) triplets per step
eval_every = 50
dpo_cache  = "data/dpo_cache.pt"
dpo_brain  = "brain_dpo.pt"
dpo_ckpt   = "ckpt_dpo.pt"

def make_repetitive(story):
    """Create a 👎 rejected story by repeating its first sentence/clause in a clumsy loop."""
    lines = [p.strip() for p in story.splitlines() if p.strip()]
    first = lines[0]
    dot = first.find(".")
    clause = first[:dot + 1] if dot != -1 else first
    return f"{clause} {clause} {clause} {story}"

def build_pairs_from_sft(x_tensor, y_tensor, seed=42):
    """
    Turn SFT (x, y) tensors into DPO preference triplets:
      (chosen_x, chosen_y, rejected_x, rejected_y)
    Half of 👎 rejected examples are OFF-TOPIC (a story from another prompt),
    and half are REPETITIVE (clumsy repeated sentences).
    """
    rng = random.Random(seed)
    B_T = tinygpt.block_size
    n = len(x_tensor)
    c_xs, c_ys, r_xs, r_ys = [], [], [], []
    word_cache = {}

    # Pre-extract prefix_ids and story_ids from each SFT row
    items = []
    for i in range(n):
        x_row = x_tensor[i].tolist()
        y_row = y_tensor[i].tolist()
        # In sft.py, y_row is -100 on the prompt prefix (except the last prefix token predicts story[0])
        valid_pos = [pos for pos, val in enumerate(y_row) if val != -100]
        if len(valid_pos) < 25:
            continue
        first_pos, last_pos = valid_pos[0], valid_pos[-1]
        prefix_ids = x_row[:first_pos + 1]
        story_ids  = y_row[first_pos:last_pos + 1]
        items.append((prefix_ids, story_ids))

    for idx, (prefix_ids, chosen_story_ids) in enumerate(items):
        if idx % 2 == 0:
            # 👎 Rejected type 1: Off-topic story (from a completely different prompt)
            other_idx = (idx + rng.randint(17, max(18, len(items) - 1))) % len(items)
            rej_story_ids = items[other_idx][1]
        else:
            # 👎 Rejected type 2: Repetitive loop at the start of the story
            story_text = tinygpt.decode(chosen_story_ids)
            rej_story_ids = tinygpt.tokenizer.encode(make_repetitive(story_text) + "\n\n", tinygpt.merges, word_cache)

        c_full = prefix_ids + chosen_story_ids
        r_full = prefix_ids + rej_story_ids
        if len(c_full) > B_T + 1:
            c_full = c_full[:B_T + 1]
        if len(r_full) > B_T + 1:
            r_full = r_full[:B_T + 1]

        c_story_len = len(c_full) - len(prefix_ids)
        r_story_len = len(r_full) - len(prefix_ids)
        if c_story_len < 15 or r_story_len < 15:
            continue

        cx = c_full[:-1] + [0] * (B_T - len(c_full) + 1)
        cy = ([-100] * (len(prefix_ids) - 1)) + c_full[len(prefix_ids):] + ([-100] * (B_T - len(c_full) + 1))
        rx = r_full[:-1] + [0] * (B_T - len(r_full) + 1)
        ry = ([-100] * (len(prefix_ids) - 1)) + r_full[len(prefix_ids):] + ([-100] * (B_T - len(r_full) + 1))

        c_xs.append(cx[:B_T]); c_ys.append(cy[:B_T])
        r_xs.append(rx[:B_T]); r_ys.append(ry[:B_T])

    return (torch.tensor(c_xs, dtype=torch.long), torch.tensor(c_ys, dtype=torch.long),
            torch.tensor(r_xs, dtype=torch.long), torch.tensor(r_ys, dtype=torch.long))

def load_dpo_data():
    """Load or build cached DPO (prompt, 👍 chosen, 👎 rejected) tensors."""
    if os.path.exists(dpo_cache):
        d = torch.load(dpo_cache)
        return d["tr"], d["va"]
    print("Building Stage 3 DPO (👍 chosen vs 👎 rejected) preference pairs...", flush=True)
    tr_x, tr_y, va_x, va_y = sft.load_sft_data()
    tr = build_pairs_from_sft(tr_x, tr_y, seed=42)
    va = build_pairs_from_sft(va_x, va_y, seed=99)
    torch.save({"tr": tr, "va": va}, dpo_cache)
    print(f"  ✓ Cached {len(tr[0]):,} train pairs & {len(va[0]):,} val pairs -> {dpo_cache}", flush=True)
    return tr, va

def seq_logprob(model, x, y):
    """Average log-probability per valid response token (ignoring -100 prompt & padding tokens)."""
    logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = (y != -100)
    safe_y = y.clamp(min=0)
    token_lp = log_probs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1) * mask
    return token_lp.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

def dpo_step_loss(policy, ref, cx, cy, rx, ry):
    """Compute anchored DPO loss and preference accuracy (% of times policy prefers 👍 over 👎)."""
    with torch.no_grad():
        ref_c = seq_logprob(ref, cx, cy)
        ref_r = seq_logprob(ref, rx, ry)
    pol_c = seq_logprob(policy, cx, cy)
    pol_r = seq_logprob(policy, rx, ry)
    # How much more the student favors 👍 vs 👎 compared to the frozen reference brain
    margin = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    dpo_loss = -F.logsigmoid(margin).mean()
    nll_loss = -pol_c.mean()  # keeps English grammar fluent on 👍 stories (prevents reward hacking)
    loss = dpo_loss + nll_weight * nll_loss
    win_rate = (margin > 0).float().mean().item() * 100.0
    return loss, dpo_loss, win_rate

def get_dpo_batch(split_tuple):
    cx, cy, rx, ry = split_tuple
    ix = torch.randint(len(cx), (batch_size,))
    dev = tinygpt.device
    return cx[ix].to(dev), cy[ix].to(dev), rx[ix].to(dev), ry[ix].to(dev)

def train():
    tr, va = load_dpo_data()
    src = "brain_chat.pt" if os.path.exists("brain_chat.pt") else "brain.pt"

    policy = tinygpt.TinyGPT().to(tinygpt.device)
    policy.load_state_dict(torch.load(src, map_location=tinygpt.device))

    ref = tinygpt.TinyGPT().to(tinygpt.device)
    ref.load_state_dict(torch.load(src, map_location=tinygpt.device))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print(f"Loaded {src} into active student + frozen anchor | Training on {len(tr[0]):,} 👍/👎 pairs", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    start_step, best_loss = 0, float("inf")
    if os.path.exists(dpo_ckpt):
        ckpt = torch.load(dpo_ckpt, map_location=tinygpt.device)
        policy.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step, best_loss = ckpt["step"], ckpt.get("best_val", float("inf"))
        print(f"Resuming DPO from step {start_step} (best loss: {best_loss:.3f})", flush=True)

    t0 = time.time()
    for step in range(start_step, steps + 1):
        if step % eval_every == 0:
            policy.eval()
            with torch.no_grad(), torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
                totals, dpos, wins = zip(*[dpo_step_loss(policy, ref, *get_dpo_batch(va)) for _ in range(15)])
                vl = sum(x.item() for x in dpos) / len(dpos)
                vt = sum(x.item() for x in totals) / len(totals)
                vw = sum(wins) / len(wins)
            print(f"DPO step {step:4d} | DPO loss {vl:.3f} (anchored {vt:.3f}) | 👍 win-rate {vw:5.1f}% | {time.time()-t0:.0f}s", flush=True)
            if step > 0 and vt < best_loss:
                best_loss = vt
                bf16_sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in policy.state_dict().items()}
                torch.save(bf16_sd, dpo_brain + ".tmp")
                os.replace(dpo_brain + ".tmp", dpo_brain)
                print(f"  🌟 saved new best {dpo_brain} (DPO loss {vl:.3f}, win-rate {vw:.1f}%)", flush=True)
            policy.train()

        if step > start_step and step % eval_every == 0:
            ckpt = {"model": policy.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "best_val": best_loss}
            torch.save(ckpt, dpo_ckpt + ".tmp")
            os.replace(dpo_ckpt + ".tmp", dpo_ckpt)

        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            loss, _, _ = dpo_step_loss(policy, ref, *get_dpo_batch(tr))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    print(f"Done! Best {dpo_brain} DPO loss: {best_loss:.3f}", flush=True)

def talk():
    model = tinygpt.TinyGPT().to(tinygpt.device)
    weights = dpo_brain if os.path.exists(dpo_brain) else "brain_chat.pt"
    model.load_state_dict(torch.load(weights, map_location=tinygpt.device))
    model.eval()
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Write a short story about owl and tree."
    print(f"🧑 You: {prompt}\n\n🦉 Hedwig (Stage 3 DPO):\n{sft.reply(model, prompt)}")

if __name__ == "__main__":
    train() if (sys.argv[1:] or ["train"])[0] == "train" else talk()
