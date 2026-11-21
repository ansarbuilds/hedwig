"""
Export a Hedwig brain so it can run inside a web browser (no Python, no server).

The browser can't read PyTorch's .pt files, so this writes two plain files per brain:
  web/brains/<name>/model.json  - the brain's shape (layers, width, heads) + where each weight lives
  web/brains/<name>/model.bin   - the weights themselves, squeezed to 8-bit
and copies the matching tokenizer.json next to them.

8-bit squeezing: every row of a weight table gets one "scale" number. Each knob is stored as a
whole number from -127 to 127, and the browser multiplies it back by the scale. That makes the
14M brain ~14 MB to download instead of ~55 MB, with almost no change to its stories.

It also writes a small test file (web/brains/<name>/check.json): the exact next-piece numbers
PyTorch picks greedily for a test prompt, so test_web.mjs can prove the JavaScript brain matches.

Run:
    .venv/bin/python export_web.py            # 14M Mac Hedwig (Stage 3 DPO)
    .venv/bin/python export_web.py 97m        # 97.6M Cloud Hedwig (Stage 3 DPO)
"""
import json, os, shutil, sys
import numpy as np, torch
import tokenizer

BRAINS = {
    "14m": {"src": "brain_dpo.pt", "tok": "tokenizer.json",
            "label": "Mac Hedwig 13.8M (Stage 3 DPO)", "n_head": 6, "block_size": 256},
    "97m": {"src": "cloud_brains_97m/brain_dpo.pt", "tok": "cloud_brains_97m/tokenizer.json",
            "label": "Cloud Hedwig 97.6M (Stage 3 DPO)", "n_head": 12, "block_size": 256},
}
CHECK_PROMPT = "Write a short story about Lily and a shiny key."

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "14m"
    cfg = BRAINS[name]
    out_dir = os.path.join("web", "brains", name)
    os.makedirs(out_dir, exist_ok=True)

    sd = torch.load(cfg["src"], map_location="cpu")
    sd = {k: v.float() for k, v in sd.items() if not k.endswith((".cos", ".sin"))}  # browser recomputes RoPE angles

    tensors, chunks, offset = [], [], 0
    restored = {}  # what the browser will see after un-squeezing, used for the PyTorch check below
    for k, v in sd.items():
        if v.dim() == 2:   # weight tables: 8-bit, one scale per row
            scale = (v.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
            q = torch.round(v / scale[:, None]).clamp(-127, 127).to(torch.int8)
            restored[k] = q.float() * scale[:, None]
            scale_bytes = scale.numpy().astype("<f4").tobytes()
            q_bytes = q.numpy().tobytes()
            tensors.append({"name": k, "shape": list(v.shape), "kind": "int8",
                            "scales": offset, "data": offset + len(scale_bytes)})
            chunks += [scale_bytes, q_bytes]
            offset += len(scale_bytes) + len(q_bytes)
        else:              # small 1-D lists (norm weights, biases): keep as normal 32-bit numbers
            restored[k] = v
            b = v.numpy().astype("<f4").tobytes()
            tensors.append({"name": k, "shape": list(v.shape), "kind": "f32", "data": offset})
            chunks.append(b)
            offset += len(b)
        pad = (-offset) % 4  # keep every list starting on a 4-byte boundary so JS can read floats directly
        if pad:
            chunks.append(b"\0" * pad)
            offset += pad

    n_embd = sd["tok.weight"].shape[1]
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    config = {"label": cfg["label"], "vocab_size": sd["tok.weight"].shape[0], "n_embd": n_embd,
              "n_head": cfg["n_head"], "n_layer": n_layer, "block_size": cfg["block_size"],
              "hidden": sd["blocks.0.mlp.gate.weight"].shape[0]}
    with open(os.path.join(out_dir, "model.bin"), "wb") as f:
        for c in chunks:
            f.write(c)
    json.dump({"config": config, "tensors": tensors}, open(os.path.join(out_dir, "model.json"), "w"))
    shutil.copy(cfg["tok"], os.path.join(out_dir, "tokenizer.json"))
    print(f"🦉 {cfg['label']}: {offset/1e6:.1f} MB -> {out_dir}/", flush=True)

    write_check(cfg, config, restored, out_dir)

def write_check(cfg, config, restored, out_dir):
    """Run the un-squeezed weights in PyTorch on CPU and save the greedy next pieces + logits."""
    import tinygpt
    merges, vocab = tokenizer.load(cfg["tok"])
    tinygpt.vocab_size = config["vocab_size"]
    tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size = (
        config["n_embd"], config["n_head"], config["n_layer"], config["block_size"])
    model = tinygpt.TinyGPT()
    model.load_state_dict(restored, strict=False)  # cos/sin come from the model's own buffers
    model.eval()

    prefix = f"User: {CHECK_PROMPT}\nHedwig: "
    ids = tokenizer.encode(prefix, merges)
    prompt_ids = list(ids)
    first_logits = None
    with torch.no_grad():
        for _ in range(40):
            logits, _ = model(torch.tensor([ids]))
            last = logits[0, -1]
            if first_logits is None:
                first_logits = last[:10].tolist()
            ids.append(int(last.argmax()))
    check = {"prefix": prefix, "prompt_ids": prompt_ids, "greedy_ids": ids[len(prompt_ids):],
             "first_logits": first_logits, "greedy_text": tokenizer.decode(ids[len(prompt_ids):], vocab)}
    json.dump(check, open(os.path.join(out_dir, "check.json"), "w"))
    print("   PyTorch greedy:", repr(check["greedy_text"][:120]), flush=True)

if __name__ == "__main__":
    main()
