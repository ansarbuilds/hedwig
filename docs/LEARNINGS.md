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

| Stage | Name | What it teaches | Hedwig |
|---|---|---|---|
| 1 | Pretraining | Language, facts, style (learn to talk) | ✅ done (`tinygpt.py` → `brain.pt`) |
| 2 | SFT (supervised fine-tuning) | Answer questions, follow instructions (manners) | ✅ done (`sft.py` → `brain_chat.pt`) |
| 3 | RLHF / DPO / RL | Give *better* answers (preference & taste) | ✅ done (`dpo.py` → `brain_dpo.pt`) |

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

## 10. Checkpointing (save + resume)

- Saves model + optimizer + scheduler + step to `ckpt.pt` every 250 steps.
- Writes to `.tmp` first, then renames (`os.replace`), so a crash mid-save can't corrupt it.
- Test: Ctrl+C after step 750 (score 1.619) → reran → "Resuming from step 750", score 1.622 ✅
- Final score after 3,000 steps: 1.481 (vs 1.48 without interruption, so resume works ✅).
- MacBook Air slows down as it heats up (no fan): ~66s → ~110s per 250 steps.
- Start a fresh run: `rm ckpt.pt`, otherwise `train` resumes from the last save.

## 11. BPE tokenizer (512 vocab) wired into `tinygpt.py`

- Built from scratch in `tokenizer.py`: starts from 256 raw UTF-8 byte tokens and merges the 256 most common pairs (`vocab_size = 512`, saved in `tokenizer.json`).
- Compresses `data/input.txt` (`1,115,394` chars) to `568,210` tokens (**1.96× shorter**; `511,389` train / `56,821` val), cached to `data/tokens_512.pt`.
- Brain grew from `4.8M` to `5,034,496` (`~5.0M`) knobs; trained 3,000 steps in `928 s` on M4 (`mps`).
- **Measured val loss:** `0: 6.388 · 250: 3.473 · 500: 3.102 · 750: 3.007 · 1000: 3.020 · 1250: 3.086 · 1500: 3.182 · 1750: 3.211 · 2000: 3.330 · 2250: 3.398 · 2500: 3.457 · 2750: 3.432 · 3000: 3.429`
- **Key lesson (overfitting):** Best score was at **step 750 (`3.007`**, ≈ `1.534` per-char equivalent when divided by `1.96`), then test score climbed to `3.429`. Because BPE compressed the book to `511k` tokens, 3,000 steps (~48 passes) made the 5M-knob brain memorize training lines instead of generalizing.
- **Samples:** Cleaner spelling and real Shakespeare names/scenes (`Mercutio`, `FRIAR PETER`, `DUKE VINCENTIO`, `SICINIUS`, `MENENIUS`).

## 12. Best-val saving + RoPE (Rotary Position Embeddings)

- Replaced learned `pos` table (`32,768` knobs) and black-box `nn.MultiheadAttention` with explicit `CausalSelfAttention` + `apply_rope` (`5,001,728` knobs).
- Updated `train()` to save `brain.pt` only when validation loss hits a new best (and persist `best_val` in `ckpt.pt`).
- **Measured val loss (with RoPE):** `0: 6.434 · 250: 3.183 · 500: 3.013 (best 🌟) · 750: 3.078 · 1000: 3.157 · 1250: 3.267 · 1500: 3.385` (stopped early via `Ctrl+C`, kept step-500 `brain.pt`).
- **Before vs after RoPE:**
  - At step 250: **`3.183`** vs `3.473` (0.29 lower!).
  - Hit the `~3.01` floor at **step 500 (`127 s`)** instead of step 750 (`188 s`) — learned roughly **1/3 faster**.
  - Floor stayed around `~3.01` because Tiny Shakespeare (`511k` BPE tokens) is now the bottleneck, not the model.

## 13. RMSNorm (replacing `LayerNorm`)

- Swapped all 13 `nn.LayerNorm(n_embd)` layers (`ln1`, `ln2`, `ln`) for `nn.RMSNorm(n_embd)` (`4,998,400` knobs — dropped `3,328` bias knobs).
- **Measured val loss (1,000 steps, same 3,000-step LR schedule):** `0: 6.427 · 250: 3.168 · 500: 3.041 (best 🌟) · 750: 3.067 · 1000: 3.128`
- **Quality:** Matches `LayerNorm` within normal batch noise (`3.168` vs `3.183` at step 250; `3.041` vs `3.013` at step 500; `3.067` vs `3.078` at step 750).
- **Surprise Apple Silicon (`mps`) timing lesson:** Even though RMSNorm does less math on paper, a side-by-side 20-step benchmark on M4 (`torch 2.15.0`) measured **`419 ms/step` for `RMSNorm` vs `323 ms/step` for `LayerNorm`** (~30% slower on `mps`, and `458 ms` vs `195 ms` standalone), because Apple's MPS backend has a fused Metal kernel for `LayerNorm` but not yet for `RMSNorm`. Keeping `RMSNorm` for modern architecture parity with CUDA/Llama; `bf16` and `F.scaled_dot_product_attention` are next for speed.

## 14. SwiGLU (replacing `GELU` MLP)

- Replaced the 2-layer `GELU` MLP (`256 → 1024 → 256`) with Llama-style gated `SwiGLU` (`gate`, `up`, `down` with width `704`, no bias) — `5,089,024` (`~5.1M`) knobs.
- **Measured val loss (750 steps, same 3,000-step LR schedule):** `0: 6.388 · 250: 3.162 · 500: 3.021 (best 🌟) · 750: 3.101` (`305 s` to step 500, same speed as RMSNorm + GELU's `304 s`).
- **Before vs after SwiGLU:** Best step-250 score yet (`3.162`) and `0.020` lower at step 500 (`3.021` vs `3.041`), right at Tiny Shakespeare's `~3.01` data floor.

## 15. `bf16` autocast (mixed precision)

- Wrapped forward + eval in `torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16)` on GPU (`mps` / `cuda`), keeping master weights in `float32`.
- **Measured val loss (750 steps, same 3,000-step LR schedule):** `0: 6.424 · 250: 3.152 · 500: 3.002 (best 🌟) · 750: 3.080`
- **Before vs after `bf16`:**
  - **Faster:** `262 s` to step 500 (vs `305 s` in `fp32`) and **`368 s` for 750 steps** (vs `487 s` — **~24% faster** on M4, and cuts activation memory in half!).
  - **Best overall BPE score yet:** **`3.002`** at step 500 (zero quality loss from 16-bit scratchpad math).

## 16. `F.scaled_dot_product_attention` (FlashAttention-ready) — Phase 1 complete ✅

- Replaced the 4 manual attention lines + `self.mask` buffer with `F.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop, is_causal=True)`.
- **Apple MPS quirk:** PyTorch `mps` SDPA does not support non-zero `dropout_p`, so `tinygpt.py` uses `0.0` on `mps` (residual/MLP dropout stays on) and `dropout` on `cuda`/`cpu`.
- **Measured val loss (750 steps, same 3,000-step LR schedule):** `0: 6.410 · 250: 3.159 (122 s) · 500: 3.086 (269 s, best 🌟) · 750: 3.213 (410 s)`
- **Before vs after SDPA:**
  - Slightly higher val loss (`3.086` vs `3.002` at step 500; `3.213` vs `3.080` at step 750) because `attn_drop = 0.0` meant slightly less regularization on tiny `511k`-token Shakespeare (so it overfit a touch sooner).
  - Microbenchmark on M4 `mps` (`bf16`): manual attention was faster locally (`69 ms` vs `138 ms` at `T=128`; `1007 ms` vs `2067 ms` at `T=512`), end-to-end `410 s` vs `368 s` for 750 steps. Kept SDPA because it is cleaner, drops the `T × T` mask buffer, and automatically enables FlashAttention on NVIDIA CUDA in Phase 3/4.
- **Phase 1 takeaway:** Hedwig now has the full modern Llama/nanochat building-block stack (BPE + RoPE + RMSNorm + SwiGLU + `bf16` + SDPA + checkpoint/resume + best-val saving). All five runs hit the exact same `~3.00–3.08` floor at step 500 because Tiny Shakespeare (`511k` tokens) is too small — time for Phase 2 (TinyStories + Gutenberg)!

## 17. Phase 2 data pipeline: TinyStories + Project Gutenberg + 4,096-vocab BPE (`prepare_data.py`)

- **Speeding up pure-Python BPE (~130× faster):** Upgraded `tokenizer.py` with GPT-2 style word-chunk pre-splitting (`PAT`) and a unique-word frequency counter / lookup cache so each distinct word is solved once instead of re-scanning millions of characters.
- **Data (`65.0 MB` cleaned stories):**
  - `roneneldan/TinyStories`: `52.4 MB` train + `10.5 MB` val.
  - 10 pre-1929 public-domain Project Gutenberg storybooks (*Alice in Wonderland*, *Through the Looking-Glass*, *Wizard of Oz*, *Peter Pan*, *Grimm's Fairy Tales*, *Andersen's Fairy Tales*, *The Jungle Book*, *Treasure Island*, *Sherlock Holmes*, *The Time Machine*): `~3.0 MB` after stripping headers/footers.
- **4,096-vocab BPE (`tokenizer.json`):** Trained 3,840 merges (`256` bytes + `3,840` merges = `4,096` pieces) in ~4 seconds. Compresses stories by **`3.87 chars/token`** (nearly 4× shorter than char-level; common whole words like `'Once'`, `' upon'`, `' little'`, `' owl'`, `' golden'` are single tokens).
- **`uint16` `.bin` shards:** Tokenized all `65 MB` in `2.7 s` (`25.6 s` total including downloads):
  - `data/train.bin`: **`14,069,966` tokens** (`28.1 MB`, ~27× more tokens than Tiny Shakespeare!)
  - `data/val.bin`: **`2,726,648` tokens** (`5.5 MB`)

## 18. Phase 2 storyteller training: 13.8M knobs on TinyStories + Gutenberg ✅

- **Model (`13,780,864` / `~13.8M` knobs):** `vocab_size = 4096`, `block_size = 256` (~1,000 chars of story context), `n_embd = 384`, `n_head = 6`, `n_layer = 6`, `batch_size = 32`, `dropout = 0.05`, `lr = 1e-3` (`CosineAnnealingLR` over `3,000` steps).
- **Measured val loss across all 3,000 steps (~24.6M tokens seen, ~59 min total on M4 `mps`, resumed cleanly at step 1,500):**
  `0: 8.501 · 250: 3.113 🌟 · 500: 2.741 🌟 · 750: 2.535 🌟 · 1000: 2.405 🌟 · 1250: 2.313 🌟 · 1500: 2.275 🌟 · 1750: 2.209 🌟 · 2000: 2.156 🌟 · 2250: 2.132 🌟 · 2500: 2.113 🌟 · 2750: 2.080 🌟 (best) · 3000: 2.089`
- **Why it never overfit:** Validation loss dropped from `8.501` all the way to **`2.080`** (`2.080 ÷ 3.87 ≈ 0.537` per character —nearly **3× better** than Shakespeare's `1.481`!).
- **What improved from step 1,500 (`2.276`) → step 2,750 (`2.080`):**
  - At step 1,500, `"Once upon a time, a little owl "` briefly drifted into calling the owl a monkey that *"barked happily"*, and `"found a magic "` produced `"magic icero"`.
  - At step 2,750 (`2.080`), **Zoey the little owl stays an owl living in a big tree for the entire story** (`"...thanked his friends for the kindness and for helping him. From that day on, Zoey never had to worry again."`), and `"found a magic "` produces **`"magic ingredient"`** and **`"magic spell"`**.
- **16-bit weight saving (`bfloat16`):** Saving `brain.pt` in `bfloat16` cuts the file size in half from `53.0 MB` down to **`27.8 MB`** (well under GitHub's 50 MB warning) with zero quality loss.

## 19. Stage 2 — Supervised Fine-Tuning (SFT: "Chat Manners") (`sft.py`) ✅

- **What SFT changes:** Stage 1 (`tinygpt.py`) only knows how to autocomplete text (and rambles into a second story after finishing the first). Stage 2 (`sft.py`) loads `brain.pt` (`13.8M` knobs) and trains on **`13,852` `User: <request>\nHedwig: <story>\n\n` pairs** (`2,327` val pairs) built from `roneneldan/TinyStoriesInstruct`.
- **Two key SFT mechanics:**
  1. **Prompt loss masking (`ignore_index = -100`):** All tokens in `User: ...\nHedwig:` (and right-padding) have target `-100`, so `F.cross_entropy` only grades Hedwig on the story response.
  2. **End-of-story stopping (`\n\n`):** Every SFT story ends with `\n\n`, so `sft.py talk` and `sft.py chat` stop cleanly as soon as the single requested story finishes.
- **Measured SFT response loss (750 steps at `lr = 2e-4`, `746 s` on M4 `mps`):**
  `0: 2.077 · 150: 1.825 🌟 · 300: 1.769 🌟 · 450: 1.777 · 600: 1.781 · 750: 1.766 🌟 (best)`
- **Result (`brain_chat.pt`, `27.8 MB` in `bfloat16`):** Follows story instructions (e.g. `"Tell me a story where a little girl named Lily finds a shiny key in the garden"` or `"Write a story: A little bird named Pip learns to fly high above the forest with his mom"`) and stops at the end of the story.

## 20. Phase 5 — 8-bit & 4-bit Quantization (`quantize.py`) + Local Mac Chat UI (`app.py`) ✅

- **How from-scratch quantization works (`quantize.py`):**
  - **8-bit (`int8`, `brain_chat_int8.pt`):** Scales each weight matrix row by `max(|row|) / 127` and rounds to 1-byte integers (`-127..+127`) plus a `float16` row scale.
  - **4-bit (`int4`, `brain_chat_int4.pt`):** Splits weights into groups of `32` knobs, scales each group to `-7..+7`, and packs **two 4-bit knobs into one `uint8` byte** (`0.5` bytes per knob!).
- **Measured quality vs file size on the SFT validation set (30 seeded batches):**

| Precision | File size | Shrink vs 32-bit | SFT test score (lower = smarter) |
|---|---|---|---|
| 32-bit (`float32`) | `55.52 MB` | `1.00×` | `1.7544` |
| 16-bit (`bfloat16`, `brain_chat.pt`) | `27.78 MB` | `2.00×` | `1.7544` (`+0.0000`) |
| 8-bit (`int8`, `brain_chat_int8.pt`) | **`14.00 MB`** | **`3.96×`** | **`1.7547`** (`+0.0003` — virtually identical!) |
| 4-bit (`int4`, `brain_chat_int4.pt`) | **`7.87 MB`** | **`7.05×`** | **`1.7718`** (`+0.0174` — still writes full coherent stories at 7.9 MB!) |

- **Local Mac Chat UI (`app.py`):** Zero-dependency web app at `http://127.0.0.1:8000` with live switching between 16-bit (`27.8 MB`), 8-bit (`14.0 MB`), 4-bit (`7.9 MB`), and Stage 1 Base (`27.8 MB`) brains, story-starter chips, and a creativity (temperature) slider.

## 21. Stage 3 — Direct Preference Optimization (DPO: "Taste & Reward Hacking") (`dpo.py`) ✅

- **What DPO teaches:** Built `13,852` `(prompt, 👍 chosen, 👎 rejected)` preference triplets (`2,327` val) where 👎 rejected stories are either off-topic (from a different prompt) or have clumsy sentence repetition. Kept a frozen copy of `brain_chat.pt` as the reference anchor (`ref`) and trained the active student (`policy`) to widen the log-probability gap between 👍 and 👎.
- **Run 1 — Pure DPO without fluency anchor (`beta = 0.2, lr = 5e-5, 300 steps`) exposed "Reward Hacking":**
  - DPO loss dropped `0.693 → 0.269` and preference win-rate jumped `0.0% → 89.6%`.
  - It fixed the off-topic bug (asking for `"owl and tree"` or `"dragon and castle"` actually produced stories about an owl/tree and dragon/castle instead of Spot the dog!), **but** by step 300 it discovered a classic RL loophole (**reward hacking**): cramming `"Lily's"` into every 3rd word (`"Lily loves Lily's key. Lily does Lily's mom's idea..."`) to artificially inflate the `pol_c - pol_r` margin at the expense of normal grammar!
- **Run 2 — Anchored DPO (`beta = 0.5, lr = 2e-5, nll_weight = 0.5, 150 steps`) fixed the reward hack:**
  - Added a `0.5 * chosen_nll` fluency anchor (Llama-3 / RPO style) so Hedwig must prefer 👍 over 👎 **while** maintaining fluent English on 👍 stories.
  - **Measured curve:** `step 0: DPO 0.693 (win 0.0%) · step 50: DPO 0.639 (win 87.9% 🌟) · step 100: DPO 0.598 (win 91.7% 🌟) · step 150: DPO 0.602 (win 89.6%)`
  - **Result (`brain_dpo.pt`, `27.8 MB` in `bfloat16`):** Reached **`91.7%` preference win-rate** in 100 steps (`103 s`) and writes clean, grammatical, on-topic stories (`"Once upon a time, there was a little girl named Lily. She was very happy because she found a shiny key... From that day on, Lily and Lily played together and had lots of fun. The end."`). All 3 stages of a modern ChatGPT-style LLM (`Pretrain → SFT → DPO`) are now complete from scratch!
