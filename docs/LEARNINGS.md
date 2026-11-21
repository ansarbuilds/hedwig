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

## 22. KV Cache (~2× faster inference) + Top-k + Repetition Penalty + Word-by-Word Streaming ✅

- **KV Cache ("Bookmarking past words" in `tinygpt.py` & `sft.py`):**
  - Without a KV cache, generating word #180 re-computes Queries, Keys, and Values for all 179 earlier words (`47.2 tokens/sec`, `3.81 s` for 180 tokens on M4 `mps`).
  - With a KV cache (`kv_cache = [[k_past, v_past], ...]`), Hedwig saves the rotated Keys and Values of earlier words and only computes the **1 newest word** at each step: **`89.0 tokens/sec` (`2.02 s` for 180 tokens — `1.89×` faster!)**.
- **Top-k (`top_k = 40`) + Repetition Penalty (`rep_penalty = 1.15` in `sample_next`):**
  - Gently divides the logit score of tokens used in the last 64 steps by `1.15` (stopping `"the dragon... a big dragon... give the dragon"` loops) and restricts sampling to the 40 most sensible next words (cutting off 0.01% nonsense typos).
- **Live Word-by-Word Streaming (`app.py` at `http://127.0.0.1:8000`):**
  - Streams NDJSON chunks from `sft.reply_stream()` over `ReadableStream` so stories type out word-by-word on screen at `~89 tok/s` with a live tokens/sec counter.

## 23. Stretch Goals Complete — Mixture-of-Experts (`moe.py`), Teacher → Student Distillation (`distill.py`), & Live Brain X-Ray (`app.py`) ✅

- **Mixture-of-Experts (`moe.py` → `brain_moe.pt`, 8 Experts, Top-2 Routing):**
  - Replaced each layer's dense 704-wide `SwiGLU` MLP with **8 specialist 352-wide `ExpertSwiGLU` modules** and a linear **Router (`Traffic Cop`)** + load-balancing auxiliary loss (`n_experts * sum(usage * mean_prob)`).
  - **26.2M total knobs, only 11.6M active per word** (6 of 8 experts sleep on every token!).
  - **Upcycling (`2.461 → 1.966` in 300 steps):** Initialized from `brain_dpo.pt` and sliced the trained MLP across the 8 experts. Real routing on `"Once upon a time, there was a little boy named Timmy"`:
    `Once→E5/E0 · upon→E3/E4 · a→E5/E4 · time→E4/E0 · ,→E7/E1 · there→E5/E4 · was→E3/E1 · a→E5/E4 · little→E4/E1 · boy→E4/E3 · named→E6/E7 · Timmy→E1/E0`
- **Teacher → Student Knowledge Distillation (`distill.py` → `brain_baby.pt`, 3.4M knobs, `6.8 MB`):**
  - Built a 4-layer, 4-head, 192-width **"🐣 Baby Hedwig" Student (`3.4M` knobs — `4.1×` smaller than the `13.8M` Teacher!)**.
  - Trained with **50% Teacher soft-label KL divergence (`T = 2.0`) + 50% ground-truth cross-entropy** over 600 steps (`7.602 → 3.128 → 2.722 → 2.607 → 2.521` in `350 s`). Writes complete multi-paragraph stories from a `6.8 MB` 16-bit file (`~1.8 MB` at 4-bit!).
- **Live "🔬 Brain X-Ray" in `app.py` (`http://127.0.0.1:8000`):**
  - Clicking any word in Hedwig's story displays a `%` probability bar chart of the **Top 5 candidate word-pieces** Hedwig was choosing between at that step, plus the **Top-2 active MoE Experts** when using the MoE brain.

## 24. Layer-by-layer SSD loading ("LLM in a Flash") (`ssd_runner.py`) ✅

Picture six notebooks stacked on your desk. Normally Hedwig keeps all six open (the whole brain in RAM) and flips through them for every new word. `ssd_runner.py` leaves five notebooks in the drawer (the SSD). For the next word it opens notebook 1, thinks, puts it back, opens notebook 2, and so on. One notebook slot is reused. The bookmark of words already written (the KV cache) stays on the desk, because each new word still needs it.

`brain_dpo.pt` (`27.79 MB` on disk) was split into `ssd_layers/resident.pt` (`6.30 MB`: word table + final predictor) and six `block_0.pt` … `block_5.pt` files (`3.58 MB` each). Both runs wrote **80** word-pieces from the same prompt and the same random seed, in a fresh process, on M4 `mps`. The stories matched word for word.

| | all layers in RAM | one layer loaded from SSD |
|---|---:|---:|
| weights held | `52.94 MB` (fp32 copy of the bf16 file) | `18.84 MB` (`2.8×` less) |
| MPS tensor memory | `52.9 MB` | `18.8 MB` |
| peak process RAM (Activity Monitor style) | `402.5 MB` | `371.8 MB` (only `30.7 MB` less) |
| time for 80 word-pieces | `0.739 s` | `2.436 s` |
| speed | `108.2 tok/s` | `32.8 tok/s` (`3.3×` slower) |
| bytes read from SSD | `0` | `1,719 MB` (`21.5 MB` per word: all 6 layer files) |

What this teaches: on a `28 MB` brain the drawer trick barely moves the process RAM number, because PyTorch itself is already ~`370 MB`. The speed cost is real — every new word re-reads six small files (`80 × 6 × 3.58 MB`). The same idea pays off when the brain is bigger than RAM (a multi-gigabyte model on a phone, or the Dec 2026 cloud brain if it outgrows a small machine). Run it again with `.venv/bin/python ssd_runner.py`.

## 25. Cloud script rehearsal (`cloud_train.sh --dry-run`) ✅

The rented-GPU script had never been run as one command. A full run on the Mac would also have overwritten `brain.pt`. The rehearsal copies the scripts into `dry_run/` (gitignored), trains for **2 steps** on a **4 MB** story slice, and leaves the real brains alone. Their file timestamps were identical before and after (`brain.pt`, `brain_chat.pt`, `brain_dpo.pt`).

What changed in the script, and what the rehearsal measured:

| | Before | After this rehearsal |
|---|---|---|
| Launch on the Mac | Would train in the real folder | Refuses, and points at `--dry-run` |
| Restart after a crash | Deleted `ckpt.pt`, `ckpt_sft.pt`, `ckpt_dpo.pt` and started over | Keeps the saves. Second launch printed `Resuming from step 2`, `Resuming SFT from step 2`, `Resuming DPO from step 2` |
| Second data prep | Re-downloaded and re-tokenized | Printed `Stories already prepared` and skipped |
| `--124m` text replace | If a line did not match, it kept training the small model and said nothing | Stops if the line is missing. A throwaway copy really became **91,286,272 knobs** (`768` wide, `12` heads, `12` layers, `6,000` steps) |
| Best-brain save | Hardcoded to every 250 steps, so a 2-step run never wrote `brain.pt` | Uses `ckpt_every`. On the real Mac config that is still `250`, so a normal training run saves on the same steps as before |

The 2-step sandbox run finished in **92 s** on the M4 and wrote all five brain files plus the three checkpoints. Scores are meaningless at 2 steps (pretrain val `8.499 → 7.556`). The point of the run was that every stage started, saved, and resumed.

## 26. Before renting: study length, door key, and a small copy of the taste brain ✅

**How long the big brain studies.** `6,000` steps would let a `91.3M` knob brain read about `49M` word-pieces, roughly one third of the `500 MB` pile. The Mac brain (`13.8M` knobs) read `24.6M` pieces in `3,000` steps, about **2 pieces per knob**, and reached val loss `2.080`. `--124m` is now **`20,000` steps**: `20,000 × 32 × 256 = 163.8M` pieces, again about **2 per knob**, one pass through the pile. Chat lessons stay at `750` SFT steps and `150` DPO steps, because those reuse the same `13,852` requests. A bookmark is still written every `250` steps, so stopping the machine early keeps the best brain.

**The door key.** This repo is private. The Mac is already logged in to GitHub. The rented computer needs a 7-day, read-only token for this repo only. The clicks are in `docs/CLOUD_GPU_100.md`. The token is not stored in the project.

**Shrinking the taste brain (`quantize.py`, 81 s on M4).** Same story-score test as Stage 2. Lower is a closer match to the real TinyStories replies.

| Brain | 16-bit (27.8 MB) | 8-bit (14.0 MB) | 4-bit (7.87 MB) |
|---|---:|---:|---:|
| Stage 2 manners `brain_chat.pt` | 1.7544 | 1.7547 | 1.7718 |
| Stage 3 taste `brain_dpo.pt` | 1.7637 | 1.7639 | 1.7827 |

8-bit barely moves the score. 4-bit costs about `0.02`, the same gap the manners brain already had. The Mac chat menu can now pick the taste brain at 16-bit, 8-bit, or 4-bit. The cloud script's last step squeezes both brains, because it calls this same `quantize.py`.

## 27. Apple MLX vs PyTorch on the same 13.8M brain (`mlx_bench.py`) ✅

Picture two calculators and the same homework. PyTorch is the calculator Hedwig already uses. MLX is Apple's calculator, built for the M4. The brain is the same size (`13,780,864` knobs: 6 layers, 384 wide, 6 heads). Both read a batch of `32` snippets, `256` pieces long.

The practice step in MLX is compiled into one GPU recipe. PyTorch uses the same 16-bit math the real training uses (`bf16` autocast) with 32-bit knobs. Five timed steps after two warmups, on a batch kept in memory so the clock is the brain and not the disk:

| | PyTorch on `mps` | MLX 32-bit | MLX 16-bit |
|---|---:|---:|---:|
| One practice step | `596 ms` | `135 ms` (`4.4×` faster) | `99 ms` (`6.0×` faster) |
| New word-pieces / sec (most likely piece, KV cache) | `635` | `708` | `832` |

The loss numbers during the practice timing (`5.14` vs `8.45`) are not a quality score. Each toolbox starts from its own random knobs. The fair check copies `brain.pt` into MLX. The biggest logit gap on the prompt was `0.012`, and the next `80` most-likely pieces matched exactly, including `"The owl was walking through the forest when he saw a big, scary wolf."`

Writing is only a little faster because each new word is a tiny job, and Python asks for the next word one at a time. Practice is where MLX wins: that step updates every knob, and `596 ms → 99 ms` is the whole class finishing the worksheet six times quicker. The chat app's earlier `~108 tok/s` also rolls dice (top-k, repetition penalty) and streams words to the browser. This test skips the dice and only times the thinking. Run it again with `.venv/bin/python mlx_bench.py`.

## 28. A phone brain that matches PyTorch (`export_iphone.py`) ✅

The old iPhone file was a screen with a pretend sentence behind the button. Core ML also refused the live brain, because that brain asks “how long is this sentence?” in a way the phone converter cannot follow.

The phone version sits at a desk with **64 fixed seats**. Real word-pieces sit on the left. Empty seats are zeros on the right. The next-word scores are read at the last real seat, so the zeros do not change the choice. Same knobs as `brain_dpo.pt`, 16-bit Core ML, **27.7 MB**, saved as `iphone/Hedwig.mlpackage` (that folder stays off git; rebuild with `.venv/bin/python export_iphone.py`).

Check: 30 new pieces, always the most likely one, prompt `"Once upon a time, a little owl "`. **Core ML matched PyTorch 30/30**, both continuing `"igloo came to visit. The owl had a big, shiny crown..."`.

The Swift dictionary (`ios/HedwigTokenizer.swift`) was compiled on the Mac and matched Python’s piece numbers on `"Once upon a time, a little owl "` (9 pieces) and `"Hello, Hedwig!"` (7 pieces). The Xcode clicks for a free Apple ID are written at the top of `ios/HedwigApp.swift`. A physical phone was not plugged in here, so the last step (press Play in Xcode) is yours.

## 29. French Hedwig (`lang_hedwig.py`) ✅

A separate small brain, so the English `brain.pt` is untouched. Books are public domain via GITenberg: *Les Trois Mousquetaires* and *Vingt mille lieues sous les mers* (about **3.7 MB** of French). A Notre-Dame file turned out to be an English translation, so it was left out.

The English dictionary drops letters like `é` because it only groups `a-z`. French uses its own pattern that keeps `é à ç œ`, and its own `lang/tokenizer.json` (2,048 pieces). Round-trip of `"Il était une fois un cœur à Noël."` stays intact.

Brain: **2.56M** knobs (4 layers, 192 wide). 600 steps, about **57 s** on the M4. Val score on held-out French:

`step 0: 7.776 · 100: 4.524 · 200: 4.050 · 300: 3.803 · 400: 3.649 · 500: 3.541 (best) · 600: 3.588`

Prompt `"Il était une fois un petit hibou "` continued in French, accents included, grammar still wobbly: `"Il était une fois un petit hibou ; le servectivement, un petit sant de ses eaux..."`. That is one hour of class, not a finished novelist. Run `.venv/bin/python lang_hedwig.py talk "Il était une fois un petit hibou "`.

## 30. More English stories (`more_english.py`) ✅

The first English brain had read about **50 MB** of TinyStories (plus 10 novels) and leveled off at val **`2.080`**. Picture a student who has finished one thin storybook and keeps rereading it. This run handed that same student a **200 MB** stack, cut with the **same** 4,096-piece dictionary, and the test pages stayed `data/val.bin`. Extra practice was **2,000 steps** at a gentler learning rate (`3e-4`), about **33 minutes** of the reading plus the chat lessons after.

| | Before | After more stories |
|---|---:|---:|
| Storytelling val score (same test pages) | 2.080 | **2.010** (best at step 1,500) |
| Chat manners score (Stage 2) | 1.7544 | **1.723** (best at step 600) |
| Taste win-rate (Stage 3) | 91.7% | **92.5%** at step 50, saved file is step 150 at **90.4%** because the fluency-anchored score kept falling (`1.541 → 1.440`) |

The reading curve: `0: 2.080 · 250: 2.105 · 500: 2.090 · 750: 2.040 · 1000: 2.038 · 1250: 2.045 · 1500: 2.010 · 1750: 2.030 · 2000: 2.046`. It got a little worse at first (the new stories tugged the knobs), then beat the old score. The train pile is **54.0M** pieces, up from **14.1M**.

Asked for an owl in a tree, the new taste brain wrote a full little story. Names still wander (Tweet, then Spark, then Sparky), which is the same small-brain habit, but the score on held-out stories is lower, which means closer to the real TinyStories pages. Previous chat files are in `brain_chat.pt.bak` and `brain_dpo.pt.bak` if you want the old ones back. Run `.venv/bin/python more_english.py` to resume; it will not rebuild the 200 MB file.

## 31. First rented GPU (`bash cloud_train.sh` on RunPod) ✅

One RTX A5000, **$0.28/hr**, about **$0.10** spent, balance left **$14.90**. The pod was stopped and the page showed **$0.00/hr**. This trained a **new** 13.8M brain from scratch on the **50 MB** TinyStories pile plus the 10 novels. It did not continue the Mac's `brain.pt`.

| | Mac, first 50 MB lesson | Rented A5000, same 50 MB lesson |
|---|---|---|
| Reading quiz | 8.501 → **2.080** in ~59 min | 8.470 → **2.100** in **135 s** |
| Manners | **1.754** (later 1.723 after more stories) | **1.758** in 81 s |
| Taste win-rate | **91.7%** | **91.7%** in 23 s |

Reading curve on the A5000: `0: 8.470 · 250: 3.090 · 500: 2.730 · 750: 2.512 · 1000: 2.466 · 1250: 2.377 · 1500: 2.249 · 1750: 2.236 · 2000: 2.197 · 2250: 2.100 · 2500: 2.100 · 2750: 2.104 · 3000: 2.111`.

Lily sample from the rented taste brain: "Once upon a time, there was a little girl named Lily. She had long hair that she loved to brush every day. One day, she found her big key in the garden and picked it up with her hands."

The copies are in `cloud_brains/` on the Mac only. They are a twin of the early Mac student. The main `brain.pt` is the later reader at **1.795**, from a partial pass over the full 1.92 GB file (stopped on purpose around step 12,500, because the rest of the ~20 hours does not help the bigger rented brain). Screenshots: `docs/images/runpod-pod-running.png`, `docs/images/runpod-first-train.png`.

## 32. Chat manners, story taste, and compression on the 1.795 reader (`sft.py` + `dpo.py` + `quantize.py`) ✅

After the partial full-shelf read brought `brain.pt` down to **`1.795`**, we re-ran Stage 2 (`sft.py`), Stage 3 (`dpo.py`), and 8-bit/4-bit compression (`quantize.py`) from scratch on top of that stronger reader (backing up the 200 MB chat brains to `.bak` and clearing `ckpt_sft.pt` / `ckpt_dpo.pt`).

| Stage | Previous (200 MB reader) | New (1.795 reader) |
|---|---:|---:|
| Stage 2 chat manners (`brain_chat.pt`) | `1.723` | **`1.438`** (best at step 450, 508 s) |
| Stage 3 taste anchored loss (`brain_dpo.pt`) | `1.541 → 1.440` | **`1.422 → 1.297`** (156 s; peak win-rate **92.5%** at step 100, saved step 150 at **88.8%**) |
| Stage 2 quantized (`16-bit / 8-bit / 4-bit`) | `1.7544 / 1.7547 / 1.7718` | **`1.4530 / 1.4531 / 1.4765`** |
| Stage 3 quantized (`16-bit / 8-bit / 4-bit`) | `1.7637 / 1.7639 / 1.7827` | **`1.4666 / 1.4667 / 1.4897`** |

Stage 2 curve: `0: 1.806 · 150: 1.493 · 300: 1.480 · 450: 1.438 (best) · 600: 1.465 · 750: 1.475`.
Stage 3 curve: `0: 0.693 (1.422, 0.0%) · 50: 0.629 (1.363, 92.1%) · 100: 0.588 (1.332, 92.5%) · 150: 0.587 (1.297, 88.8%)`.

Sample from the new `brain_dpo.pt` on `"Write a short story about a brave little owl and a golden key."`: *"Once upon a time, there was a wise owl named Hoot. He lived in the forest with his friends. One day, while he was flying by, he saw something shiny on the ground. It was a key! Hoot wanted to see what it could do..."*

## 33. First 97.6M Cloud Hedwig — Stories + Modern English + Grammar (`bash cloud_train.sh --124m`) ✅

Trained on RunPod **1× NVIDIA A40** (`48 GB VRAM`, **`$0.59/hr`**) in **~66 minutes (~$0.65 total)**. Upgraded to **97,581,824 knobs** (`12` layers, `768` wide, `12` heads, **`8,192`-piece BPE dictionary**) on **614 MB** of combined text (`366.7 MB` TinyStories + `262.1 MB` / `55,035` FineWeb-Edu modern English docs + 10 Gutenberg novels), followed by `30,930` mixed story + **Grammarly CoEdIT** (Apache-2.0) SFT pairs and `22,236` DPO pairs.

| Stage | 14M Mac Hedwig | 97.6M Cloud Hedwig (A40) | Time on A40 |
|---|---:|---:|---:|
| Stage 1 reading (`brain.pt`) | `1.795` (4k vocab, stories only) | **`2.619`** (8k vocab, stories + FineWeb-Edu) | 57 min (20,000 steps) |
| Stage 2 manners + grammar (`brain_chat.pt`) | `1.438` (stories only) | **`1.336`** (stories + 15k CoEdIT grammar) | 5.6 min (1,500 steps) |
| Stage 3 taste (`brain_dpo.pt`) | `1.297` (92.5% peak win) | **`1.235`** (**96.7%** peak / **95.0%** saved) | 52 s (150 steps) |
| 16-bit / 8-bit / 4-bit chat (`brain_chat`) | `27.8 / 14.0 / 7.9 MB` (`1.453 / 1.453 / 1.477`) | **`195.6 / 98.2 / 55.2 MB`** (`1.357 / 1.358 / 1.369`) | ~2 min |
| 16-bit / 8-bit / 4-bit taste (`brain_dpo`) | `27.8 / 14.0 / 7.9 MB` (`1.467 / 1.467 / 1.490`) | **`195.6 / 98.2 / 55.2 MB`** (`1.400 / 1.400 / 1.412`) | ~2 min |

**Real before/after proof on `"Write a story about a boy named Harry Potter."`:**
- **14M Mac Hedwig:** Chopped `"Potter"` into `"pot"` + `"ter"`, ignored the name, defaulted to *Timmy*, and wrote about *potatoes and a big pot of soup*.
- **97.6M Cloud Hedwig:** Kept **`Harry Potter`** and **`Harry`** through the entire story (*"Harry Potter was a small boy who liked to play outside. He had a red ball and he loved to kick it around the yard. One day, Harry saw something shiny in the grass..."*).

**Real grammar test:** Recognized `Fix grammar:` / `Improve the grammar of this text:` and replied with a single edited sentence instead of a bedtime story (fixing `"I goes"` → `"I go"`), while still truncating or missing past-tense details on longer Teams messages — showing both that multi-task SFT works and why production work-grammar rewriting benefits from larger models.

All 8 files (`brain*.pt` + 8,192-vocab `tokenizer.json`) live locally in `cloud_brains_97m/` (gitignored because 16-bit weights are 195.6 MB > GitHub's 100 MB limit) and are selectable side-by-side in `app.py`.

## 34. Flagship 286.7M Hedwig (`bash cloud_train.sh --300m`) + 14M vs 97.6M vs 286.7M Story Showdown ✅

Trained on RunPod **1× NVIDIA RTX 4090** (`24 GB VRAM`, **`$0.89/hr`**) in **~1 hr 55 min (~$1.70 total)**. Upgraded to **286,663,680 knobs** (`20` layers, `1024` wide, `16` heads, **`512` context window**, **`16,384`-piece BPE dictionary**) on **1.22 GB** (`523.9 MB` TinyStories + `734.0 MB` / `154,768` FineWeb-Edu docs + 10 novels = `294.6M` tokens; `25,000` steps ≈ `205M` tokens seen), followed by `71,906` SFT pairs (`TinyStories-Instruct` + `30,000` Grammarly CoEdIT) and `52,798` DPO pairs.

| Metric | 14M Mac Hedwig | 97.6M Cloud Hedwig (A40) | 286.7M Flagship Hedwig (4090) |
|---|---:|---:|---:|
| Knobs / Context / Vocab | `13.8M` / `256` / `4,096` | `97.6M` / `256` / `8,192` | **`286.7M` / `512` / `16,384`** |
| Tokens seen (ratio per knob) | ~125M (~9 / knob) | ~164M (~1.7 / knob) | ~205M (~0.7 / knob, intentionally under-read for budget) |
| Cloud cost | `$0` (Mac M4) | `~$0.65` (A40, 66 min) | **`~$1.70`** (RTX 4090, 115 min) |
| Stage 1 val *(different vocabs!)* | `1.795` (4k) | `2.619` (8k) | `2.998` (16k) |
| Stage 2 SFT (`brain_chat.pt`) | `1.438` | `1.336` | **`1.341`** (71,906 pairs) |
| Stage 3 DPO win-rate / anchored | `92.5%` / `1.297` | `96.7%` / `1.235` | **`98.3%` / `1.200`** (52,798 pairs) |
| 16-bit / 4-bit file size | `27.8 / 7.9 MB` | `195.6 / 55.2 MB` | **`574.7 / 161.9 MB`** |

**Fair Side-by-Side Story & Grammar Showdown (`compare_brains.py` → `docs/STORY_SHOWDOWN.md`, 12 fixed prompts, seed `42`, temp `0.65`):**
- **Multi-part prompt following (Clear win for 286.7M):**
  - *Lily + red umbrella + small frog:* Both 14M and 97.6M forgot the frog completely; **286.7M** included Lily, the small frog, and the red umbrella.
  - *Two best friends build a treehouse:* 14M wrote about one girl; 97.6M got confused and wrote `"What are you doing, Jack?" asked Jack`; **286.7M** kept two distinct friends (*Bob and Sue*) straight through a thunderstorm.
  - *Baby owl afraid of the dark until he saw the moon:* 14M started with an egg; 97.6M used a cloud; **286.7M** hit baby owl + dark + stars + moon.
  - *Harry Potter:* 14M turned him into a puppy talking to `"Potus"`; both **97.6M** and **286.7M** kept `"Harry Potter"` and `"Harry"` as a boy throughout.
- **Chinchilla "under-read" lesson visible in Grammar mode:** 14M ignored grammar prompts and wrote bedtime stories; both 97.6M and 286.7M immediately switched into single-sentence edit mode, and the better-read 97.6M fixed `"I goes"` → `"I go"` while the under-read 286.7M (`~0.7` tokens/knob) still left `"I goes"` or drifted `"stuck"` → `"snapped"`.

Weights + 16k `tokenizer.json` live locally in `cloud_brains_300m/` (gitignored), backed up to Hugging Face (`ansarbuilds/hedwig-300m`), and wired into `app.py` side-by-side with 97.6M and 14M.
