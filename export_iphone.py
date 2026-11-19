"""
Build a Hedwig the iPhone can actually run.

The phone gets a fixed desk: 64 word-pieces. Real words sit on the left,
empty seats (zeros) pad the right. The brain only reads the last real word's
scores, so the empty seats do not change the story. Core ML is the format
Xcode understands. We check it against PyTorch before we call it done.

Run:  .venv/bin/python export_iphone.py
"""
import os, shutil
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import tinygpt

SEQ = 64
OUT = "iphone"
SRC = "brain_dpo.pt" if os.path.exists("brain_dpo.pt") else "brain_chat.pt"

def phone_attn(attn, x):
    """Same attention as tinygpt.py, with the length baked in as the number 64."""
    n_embd, n_head = tinygpt.n_embd, tinygpt.n_head
    head_dim = n_embd // n_head
    q, k, v = attn.qkv(x).split(n_embd, dim=-1)
    q = q.view(1, SEQ, n_head, head_dim).transpose(1, 2)
    k = k.view(1, SEQ, n_head, head_dim).transpose(1, 2)
    v = v.view(1, SEQ, n_head, head_dim).transpose(1, 2)
    cos, sin = attn.cos[:, :, :SEQ], attn.sin[:, :, :SEQ]
    def rope(t):
        a, b = t.chunk(2, dim=-1)
        return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    q, k = rope(q), rope(k)
    scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    mask = torch.triu(torch.ones(SEQ, SEQ, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, -1e4)
    out = torch.softmax(scores, dim=-1) @ v
    out = out.transpose(1, 2).contiguous().view(1, SEQ, n_embd)
    return attn.proj(out)

class PhoneHedwig(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, idx):
        x = self.base.tok(idx.long())
        for block in self.base.blocks:
            x = x + phone_attn(block.attn, block.ln1(x))
            x = x + block.mlp(block.ln2(x))
        return self.base.head(self.base.ln(x))

def pad(ids):
    ids = ids[:SEQ]
    return ids + [0] * (SEQ - len(ids)), len(ids)

def greedy(predict, ids, n_new):
    """Always pick the most likely next piece. predict(padded) -> logits [SEQ, V]."""
    out = list(ids)
    for _ in range(n_new):
        if len(out) >= SEQ:
            break
        padded, n = pad(out)
        logits = predict(padded)
        out.append(int(logits[n - 1].argmax()))
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("ios", exist_ok=True)
    print(f"1/4 Loading {SRC} ...", flush=True)
    base = tinygpt.TinyGPT().cpu().eval()
    base.load_state_dict(torch.load(SRC, map_location="cpu", weights_only=True))
    phone = PhoneHedwig(base).eval()
    example = torch.zeros(1, SEQ, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(phone, example)

    print("2/4 Converting to Core ML (this is the file Xcode runs)...", flush=True)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=(1, SEQ), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS16,
    )
    model_path = os.path.join(OUT, "Hedwig.mlpackage")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    ml.save(model_path)

    prompt = "Once upon a time, a little owl "
    ids = tinygpt.encode(prompt)

    def predict_torch(padded):
        with torch.no_grad():
            logits = phone(torch.tensor([padded]))
        return logits[0].float()

    def predict_coreml(padded):
        got = ml.predict({"ids": np.array([padded], dtype=np.int32)})
        return torch.tensor(np.array(got["logits"][0]), dtype=torch.float32)

    story_pt = greedy(predict_torch, ids, 30)
    story_ml = greedy(predict_coreml, ids, 30)
    new_pt, new_ml = story_pt[len(ids):], story_ml[len(ids):]
    same = sum(a == b for a, b in zip(new_pt, new_ml))
    print(f"3/4 Core ML matched PyTorch on {same}/{len(new_pt)} new pieces.")
    print("   PyTorch:", tinygpt.decode(new_pt)[:180])
    print("   Core ML:", tinygpt.decode(new_ml)[:180])

    shutil.copyfile("tokenizer.json", os.path.join(OUT, "tokenizer.json"))
    shutil.copyfile("ios/HedwigTokenizer.swift", os.path.join(OUT, "HedwigTokenizer.swift"))
    shutil.copyfile("ios/HedwigApp.swift", os.path.join(OUT, "HedwigApp.swift"))
    # A few sentences so we can check the Swift dictionary against Python.
    import json
    checks = ["Once upon a time, a little owl ", "Hello, Hedwig!"]
    json.dump([{"text": s, "ids": tinygpt.encode(s)} for s in checks],
              open(os.path.join(OUT, "bpe_check.json"), "w"), ensure_ascii=False, indent=2)
    mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(model_path) for f in fs) / 1e6
    print(f"4/4 Wrote {model_path} ({mb:.1f} MB), tokenizer, and the Swift app.")
    print("Xcode steps are at the top of ios/HedwigApp.swift.")

if __name__ == "__main__":
    main()
