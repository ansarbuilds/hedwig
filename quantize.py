"""
Phase 5 for Hedwig: From-scratch 8-bit (int8) and 4-bit (int4) quantization!

In plain English:
- A 32-bit knob uses 4 bytes (like writing 0.37482910).
- A 16-bit knob (bfloat16) uses 2 bytes (like writing 0.375).
- An 8-bit knob (int8) squishes each row into whole numbers from -127 to +127 (1 byte per knob).
- A 4-bit knob (int4) squishes groups of 32 knobs into whole numbers from -7 to +7 and packs
  TWO knobs into a single byte (0.5 bytes per knob — nearly 8x smaller than 32-bit!).

Run:   .venv/bin/python quantize.py          (quantize brain_chat.pt -> int8 & int4, measure score & size)
       .venv/bin/python quantize.py talk 4   (generate a story from the 4-bit 7.4 MB brain!)
"""
import os, sys, time
import torch
import tinygpt, sft

INT8_PATH = "brain_chat_int8.pt"
INT4_PATH = "brain_chat_int4.pt"
GROUP_SIZE = 32  # for 4-bit: share one scale factor across every 32 knobs

def quantize_int8(state_dict):
    """Per-row absmax 8-bit quantization: float -> int8 (-127..127) + float16 scale per row."""
    out = {}
    for k, w in state_dict.items():
        if not w.is_floating_point() or w.ndim < 2:
            out[k] = w.to(torch.bfloat16) if w.is_floating_point() else w
            continue
        wf = w.float()
        scale = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = (wf / scale).round().clamp(-127, 127).to(torch.int8)
        out[k] = {"q8": q, "scale": scale.to(torch.float16)}
    return out

def quantize_int4(state_dict):
    """Group-wise 4-bit quantization (group_size=32), packing two int4 values into each uint8 byte."""
    out = {}
    for k, w in state_dict.items():
        if not w.is_floating_point() or w.ndim < 2 or w.numel() % GROUP_SIZE != 0:
            out[k] = w.to(torch.bfloat16) if w.is_floating_point() else w
            continue
        wf = w.float()
        orig_shape = wf.shape
        grouped = wf.reshape(-1, GROUP_SIZE)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        # Map -7..+7 to 1..15 (unsigned 0..15) so two 4-bit nibbles pack cleanly into one uint8 byte
        q = ((grouped / scale).round().clamp(-7, 7).to(torch.int16) + 8).to(torch.uint8)
        packed = (q[:, 0::2] << 4) | (q[:, 1::2] & 0x0F)
        out[k] = {"q4": packed, "scale": scale.to(torch.float16), "shape": orig_shape}
    return out

def dequantize(q_state_dict):
    """Unpack an int8 or int4 state_dict back into float tensors so TinyGPT can run it."""
    out = {}
    for k, v in q_state_dict.items():
        if isinstance(v, dict) and "q8" in v:
            out[k] = v["q8"].float() * v["scale"].float()
        elif isinstance(v, dict) and "q4" in v:
            packed = v["q4"]
            high = (packed >> 4) & 0x0F
            low  = packed & 0x0F
            unpacked = torch.stack([high, low], dim=-1).reshape(-1, GROUP_SIZE).to(torch.float32) - 8.0
            out[k] = (unpacked * v["scale"].float()).reshape(v["shape"])
        else:
            out[k] = v
    return out

@torch.no_grad()
def evaluate_sft_loss(model, va_x, va_y, n_batches=30):
    """Measure story response loss on the SFT validation set (seeded so every bit-width sees identical batches)."""
    model.eval()
    g = torch.Generator().manual_seed(42)
    total = 0.0
    for _ in range(n_batches):
        ix = torch.randint(len(va_x), (sft.batch_size,), generator=g)
        x, y = va_x[ix].to(tinygpt.device), va_y[ix].to(tinygpt.device)
        with torch.autocast(device_type=tinygpt.device, dtype=torch.bfloat16, enabled=tinygpt.use_bf16):
            total += model(x, y)[1].item()
    return total / n_batches

def main():
    src = "brain_chat.pt" if os.path.exists("brain_chat.pt") else "brain.pt"
    print(f"Loading {src} ({os.path.getsize(src)/1e6:.2f} MB on disk)...", flush=True)
    sd_bf16 = torch.load(src, map_location="cpu")
    sd_fp32 = {k: (v.float() if v.is_floating_point() else v) for k, v in sd_bf16.items()}

    print("Quantizing to 8-bit (int8) and 4-bit (packed int4)...", flush=True)
    q8 = quantize_int8(sd_fp32)
    q4 = quantize_int4(sd_fp32)
    torch.save(q8, INT8_PATH)
    torch.save(q4, INT4_PATH)

    _, _, va_x, va_y = sft.load_sft_data()
    model = tinygpt.TinyGPT().to(tinygpt.device)
    prompt = "Tell me a story where a little girl named Lily finds a shiny key in the garden."

    rows = [
        ("32-bit (float32)", sd_fp32, sum(v.numel() * v.element_size() for v in sd_fp32.values()) / 1e6),
        ("16-bit (bfloat16)", sd_bf16, os.path.getsize(src) / 1e6),
        (" 8-bit (int8)",     dequantize(torch.load(INT8_PATH, map_location="cpu")), os.path.getsize(INT8_PATH) / 1e6),
        (" 4-bit (int4)",     dequantize(torch.load(INT4_PATH, map_location="cpu")), os.path.getsize(INT4_PATH) / 1e6),
    ]

    print(f"\n{'Precision':20s} | {'Size (MB)':10s} | {'Shrink':8s} | {'Test Score (lower=smarter)':26s}")
    print("-" * 74)
    base_mb = rows[0][2]
    samples = {}
    for label, sd, mb in rows:
        model.load_state_dict(sd)
        loss = evaluate_sft_loss(model, va_x, va_y)
        print(f"{label:20s} | {mb:7.2f} MB | {base_mb/mb:6.2f}x  | {loss:.4f}", flush=True)
        torch.manual_seed(7)
        samples[label.strip()] = sft.reply(model, prompt)

    for label in ("16-bit (bfloat16)", "8-bit (int8)", "4-bit (int4)"):
        print(f"\n--- Sample from {label} ---")
        print(samples[label])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "talk":
        bits = sys.argv[2] if len(sys.argv) > 2 else "4"
        path = INT4_PATH if "4" in bits else INT8_PATH
        model = tinygpt.TinyGPT().to(tinygpt.device)
        model.load_state_dict(dequantize(torch.load(path, map_location="cpu")))
        model.eval()
        p = sys.argv[3] if len(sys.argv) > 3 else "Write a short story about a brave little owl and a golden key."
        print(f"🦉 Hedwig ({bits}-bit, {os.path.getsize(path)/1e6:.1f} MB):\n{sft.reply(model, p)}")
    else:
        main()
