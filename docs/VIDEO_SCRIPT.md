# 🦉 Hedwig — Video & Full Documentation Storyboard

A complete storyboard telling Hedwig's real journey in plain English: trying to build an LLM from scratch on a MacBook Air, hitting the laptop wall, renting NVIDIA GPUs that run 26× faster, growing to 286.7M knobs for $2.45, and comparing honestly against GPT & Claude.

---

## Act 1 — The Hook: My Try at Building an LLM From Scratch
* **Opening line:** *"Everyone uses ChatGPT and Claude — so I wanted to try building my own LLM from scratch in pure PyTorch, starting from zero knobs on a MacBook Air, to see how it really works."*
* **On screen:** Live recording of `app.py` (`http://127.0.0.1:8000`) streaming a story word-by-word, clicking a word to open **🔬 Brain X-Ray**, and the 60-second launch film (`film/out/hedwig-demo.mp4`).

## Act 2 — What I Built on My MacBook Air M4 (`$0.00`)
* **Day 1 (`v1`, 4.8M):** Character-level GPT guessing 1 letter at a time on Tiny Shakespeare (`loss 4.388 → 1.483`).
* **From-scratch BPE Tokenizer (`tokenizer.py`):** 4,096 word-pieces (`3.87` chars/piece), packing ~1,000 characters into 256 slots.
* **Modernizing the Transformer (`tinygpt.py`):** Upgraded one piece at a time (`RoPE → RMSNorm → SwiGLU → bf16 → SDPA`), measuring each — catching that `RMSNorm` ran 30% slower than `LayerNorm` on Apple `mps`.
* **All 3 ChatGPT stages on Mac (13.8M):**
  * **Pretrain (`tinygpt.py` + `more_english.py`):** `8.501 → 2.080 → 2.010 → 1.795` on TinyStories + 10 Gutenberg novels.
  * **SFT (`sft.py`):** `13,852` instruction pairs with `-100` prompt loss masking (`1.438`).
  * **DPO (`dpo.py`) & Reward Hacking blooper:** Pure DPO cheated by cramming `"Lily's"` into every 3rd word (*"Lily loves Lily's key. Lily does Lily's mom's idea..."*). A Llama-3-style fluency anchor closed the cheat (`92.5%` peak win-rate).
  * **4-bit compression (`quantize.py`) & Mac Lab:** Shrank 7× (`27.8 MB → 7.9 MB`), KV cache doubled speed (`47 → 89 tok/s`), plus MoE (`moe.py`), Baby Hedwig distillation (`distill.py`), Apple MLX (`6×` faster), SSD streaming (`ssd_runner.py`), Core ML iPhone export, and French Hedwig.

## Act 3 — Why the Mac Wasn't Enough
* **The Compute Wall:** Training work = `6 × knobs × tokens`. Full TinyStories takes ~20 hours on M4 and the fanless Air thermally throttles (`66s → 110s` per 250 steps). A 100M+ brain would take weeks.
* **The 4k Vocabulary Ceiling (Funny Failure):** Asking 14M Hedwig for **"Harry Potter"** chopped `"Potter"` into `"pot"` + `"ter"` — so Hedwig wrote about **Timmy and a big pot of soup** (or turned Harry into a puppy talking to `"Potus"`), and dropped the frog in Lily's red umbrella prompt.

## Act 4 — Why Rented GPUs Are 26× Faster & What We Improved (`$2.45`)
* **Zero wasted minutes:** Rehearsed locally first (`bash cloud_train.sh --dry-run`, 92s).
* **The 26× speed jump:** On the exact same 50 MB lesson, 1× RTX A5000 (`$0.10`) finished in **135 seconds vs ~59 minutes on the Mac Air** thanks to `bf16` Tensor Cores, native CUDA FlashAttention, dedicated high-bandwidth VRAM, and active cooling.
* **Scaling up (`97.6M` & `286.7M`):**
  * **97.6M on 1× NVIDIA A40 (`$0.65`, 66 min):** 8k vocab, 614 MB TinyStories + FineWeb-Edu + CoEdIT grammar → SFT `1.336`, DPO **`96.7%`**. Kept `"Harry Potter"` intact and switched to single-sentence grammar edits!
  * **286.7M on 1× RTX 4090 (`$1.70`, 115 min):** 16k vocab, 512 context, 1.22 GB + 71,906 SFT pairs → DPO **`98.3%`**. Nailed multi-part stories (Lily + red umbrella + frog; Bob & Sue building a treehouse through a storm; baby owl + dark + moon) and wrote the 84s movie *Pip and the Lost Key*.

## Act 5 — Hedwig vs. GPT & Claude (How Frontier Models Train)
* **Visual scale comparison:** Same Transformer and 3-stage recipe (`Pretrain → SFT → DPO/RLHF`).
* **The cluster & reading gap:** Flagship Hedwig used **1× RTX 4090** across **~205M tokens (`0.7` tokens/knob)** for `$1.70`. Llama 3 405B used **16,384× H100 GPUs** across **15.6 trillion tokens (`~38.5` tokens/knob)**. Frontier GPT-4/5 and Claude Opus models use **10,000–100,000+ GPUs** across **10T–20T+ tokens** + deep RL post-training for **$50M–$500M+**.

## Act 6 — What I Learnt & Open Source
* Five plain lessons: (1) change one thing and measure, (2) rising loss/win-rate can hide reward hacking, (3) reading tokens/knob beats raw parameter count (`97.6M` beat under-read `286.7M` on grammar), (4) compare stories across vocabs not raw loss, (5) dry-run locally before renting GPUs.
* MIT code on GitHub, weights on Hugging Face (`hedwig-97m`, `hedwig-300m`), live browser demo written in pure JS (`40/40` match with PyTorch).
