# What I learned building Hedwig

Notes from the first session (Nov 2026). Numbers marked "rough" are estimates, not measurements.

## 1. What an LLM is

- An LLM plays one game: **guess the next token** (a letter, word piece, or word).
- A chatbot answer is many guesses in a row, each fed back in as input.
- **Parameters ("knobs")** are the numbers inside the model. Training tunes them.
- Hedwig, MiniMind, GPT-2 and GPT-5 all use the same core design: the **Transformer**. The difference is scale, data and post-training.

## 2. What training is

One training step:

1. **Read**: take a chunk of text (Hedwig v1: 128 characters × 64 chunks).
2. **Guess**: predict every next character with the current knobs.
3. **Check**: measure how wrong it was (cross-entropy loss, the "test score").
4. **Nudge**: backpropagation computes which way to turn every knob; the optimizer (AdamW) turns them a little.

Repeat thousands to billions of times.

## 3. The three stages of a ChatGPT-style model

| Stage | Name | What it teaches | Hedwig v1 |
|---|---|---|---|
| 1 | Pretraining | Language, facts, style (learn to talk) | ✅ done |
| 2 | SFT (supervised fine-tuning) | Answer questions, follow instructions (manners) | ❌ |
| 3 | RLHF / DPO / RL | Give *better* answers | ❌ |

## 4. Hedwig v1 results (measured)

| | |
|---|---|
| Parameters | 4,805,185 (≈4.8M) |
| Layout | 6 layers, 256 embedding size, 4 attention heads, 128 context |
| Tokenizer | character-level, 65 symbols |
| Data | Tiny Shakespeare, 1.1 MB |
| Hardware | MacBook Air M4 (16 GB RAM, 8-core GPU, PyTorch MPS) |
| Time | 977 s (~16 min) for 3,000 steps |
| Loss | 4.388 → 1.483 |
| Brain file | `brain.pt`, 18 MB |

Loss log: 0: 4.388 · 250: 1.964 · 500: 1.738 · 750: 1.632 · 1000: 1.569 · 1250: 1.523 · 1500: 1.522 · 1750: 1.495 · 2000: 1.485 · 2250: 1.488 · 2500: 1.502 · 2750: 1.492 · 3000: 1.483

Loss flattened after ~2,000 steps, so the model is now limited by its size and data, not by training time.

## 5. Memory vs compute (the key insight)

- **To run** a model, the whole brain must sit in RAM, because every token touches every active parameter.
  - Running from SSD works (memory-mapping, llama.cpp) but is ~40× slower than RAM.
  - 16 GB Mac → roughly a 6B model at 16-bit, 12B at 8-bit, 20B at 4-bit (rough).
- **To train**, each parameter needs ~16 bytes (weights + gradients + 2 Adam states) plus activations.
  - 16 GB Mac → fits ~500M–750M parameters for training (rough).
- **The real limit is compute speed, not memory.** Training FLOPs ≈ 6 × params × tokens.
  - Chinchilla rule: tokens ≈ 20 × params for compute-optimal training.
  - 124M model ≈ 2×10¹⁸ FLOPs ≈ 2–3 weeks on the Air (rough); ~1–2 hours on one H100.
  - 2× parameters ≈ 4× training time (because you should also read 2× more data).

## 6. Why the finished brain is small

The model learns *patterns*, not copies. 500 GB of training text can produce a 6 GB brain. Proof: Hedwig v1's brain (18 MB) is *bigger* than its data (1.1 MB), and it cannot recite the plays; it writes new Shakespeare-ish lines.

## 7. Making small models smart

- **Focus**: TinyStories (Microsoft, 2023) showed a ~28M model can beat GPT-2 XL (1.5B) at simple stories.
- **Better tokenizer**: word pieces (BPE) instead of characters, so there's ~3× more text per step.
- **Modern parts**: RoPE, RMSNorm, SwiGLU, bf16, flash attention.
- **Distillation**: a big open model (Qwen, Mistral, DeepSeek; check licenses) acts as a teacher, either writing training data or providing soft probabilities. Do NOT distill from GPT/Claude outputs; their terms forbid building competing models.
- **Quantization**: 16-bit → 8/4-bit makes it 2–4× smaller and faster with little quality loss. BitNet goes to 1.58-bit.
- **Mixture of Experts**: only a few "expert" sub-networks run per token.

## 8. Budget ladder (rough, cloud H100 ≈ $2–4/hr)

| Budget | Model | Like | File (16-bit) |
|---|---|---|---|
| $0 (Air) | 124M | GPT-2 Small | ~250 MB |
| ~$15–100 | 124M–560M (nanochat) | GPT-2 grade + chat | ~0.25–1 GB |
| ~$1,000 | 1.5B | GPT-2 XL | ~3 GB |
| ~$10,000 | 3–7B | small GPT-3 | ~6–14 GB |
| ~$1B+ | frontier | GPT-5 | not public |

Everything up to ~$10k produces a brain that runs on a 16 GB Mac (7B only at 4-bit).

## 9. Hardware and business notes

- Mac mini M6 (Sept 2026, from $899) has GPU neural accelerators; Apple claims up to 4× AI vs M4. Probably ~2–4× faster training than the Air (unverified).
- Renting beats buying for occasional big runs.
- GPU rental as a business is weak at current prices (an RTX 5090 cost ~CAD $6–13k in late 2026 during a shortage). Building *with* AI is the better path.
- Don't use the company laptop for personal projects.

## Checkpointing (save + resume)

- Saves model + optimizer + scheduler + step to `ckpt.pt` every 250 steps

- Writes to `.tmp` first, then renames, so a crash mid-save can't corrupt it

- Test: Ctrl+C after step 750 (score 1.619) → reran → "Resuming from step 750", score 1.622 ✅

- Final score after 3000 steps: 1.481 (vs 1.48 without interruption, so resume works ✅)

- MacBook Air slows down as it heats up (no fan): ~66s → ~110s per 250 steps

- Start a fresh run: `rm ckpt.pt`, otherwise `train` resumes from the last save
