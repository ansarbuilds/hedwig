# 🦉 Hedwig — Video & Full Documentation Storyboard

A ready-to-film outline for telling the whole Hedwig story in plain English, from Day 1 on a MacBook Air M4 to scaling up on a rented cloud GPU and open-sourcing it.

---

## Act 1 — The Hook (0:00–0:45)
* **Opening line:** *"Can you build a tiny ChatGPT from scratch on a regular MacBook Air — no Hugging Face Trainer, no APIs, just pure PyTorch — and actually watch how it thinks?"*
* **On screen:** Live recording of `app.py` (`http://127.0.0.1:8000`) streaming a bedtime story at ~89 tokens/sec, then clicking a word to open the **🔬 Brain X-Ray** showing the top-5 next word-pieces.
* **The scale anchor:** Hedwig is **14 million** knobs (GPT-2 small is **124 million**, GPT-3 is **175 billion**, ChatGPT has read a huge slice of the web). Hedwig fits in **7.9 MB** at 4-bit and runs 100% offline.

## Act 2 — Day 1 to Modern Transformer on a MacBook Air (0:45–2:30)
* **Day 1 (`v1`):** 4.8M character-level GPT guessing one letter at a time on Tiny Shakespeare (`loss 4.388 → 1.481`).
* **Teaching it words (`tokenizer.py`):** Built a 4,096-piece BPE tokenizer from scratch (`3.87` chars/piece).
  * *Great visual moment:* Show `"Once upon a time, a little owl named Hedwig..."` split into pills in `how-hedwig-learns.html`.
* **Modernizing the engine (`tinygpt.py`):** Upgraded one piece at a time (2018 GPT-2 → 2026 Llama/nanochat style): `RoPE → RMSNorm → SwiGLU → bf16 → SDPA`.
* **More stories = smarter brain (`more_english.py`):**
  * 50 MB TinyStories + 10 public-domain novels → quiz score **2.080**
  * 200 MB stack → **2.010**
  * Partial read of the full 1.92 GB shelf → **1.795**

## Act 3 — The 3 Stages of ChatGPT + Funny Failures (2:30–4:30)
* **Stage 1 (Pretrain):** Autocomplete storyteller (`brain.pt`, `1.795`).
* **Stage 2 (SFT / Chat Manners, `sft.py`):** Taught `User: ... → Hedwig: ...` on 13,852 story prompts with `-100` prompt masking (`1.806 → 1.438`).
* **Stage 3 (DPO / Taste, `dpo.py`):** Showed 👍 on-topic stories vs 👎 off-topic/repetitive stories.
  * *Funny blooper #1 — Reward Hacking:* Pure DPO discovered a loophole and wrote `"Lily loves Lily's key. Lily does Lily's mom's idea..."` to game the score! Adding a fluency anchor fixed it cleanly (`1.422 → 1.297`, **92.5%** peak win-rate).
* **Shrinking it 7× (`quantize.py`):** Packed 2 knobs per byte (4-bit) — `27.8 MB → 7.9 MB` with almost zero quality loss (`1.4666 → 1.4897`).
* *Funny blooper #2 — Why 14M is still tiny:* Asking for **"Harry Potter"** produced a story about **Timmy and a big pot of soup** — because `"potter"` got chopped into `pot` + `ter`, and Hedwig latched onto `pot` → `potatoes` → `pot of soup`!

## Act 4 — Side Quests on the M4 (4:30–5:45)
Quick montage of the 5 Mac experiments (all `$0`):
1. **Mixture-of-Experts (`moe.py`):** 8 specialist experts, only 2 awake per word (`26.2M` total / `11.6M` active).
2. **Baby Hedwig (`distill.py`):** Distilled a `3.4M` (`6.8 MB`) student from the `13.8M` teacher.
3. **Apple MLX (`mlx_bench.py`):** Same brain ran **6× faster** per training step (`596 ms → 99 ms`) with identical greedy output.
4. **SSD Layer Streaming (`ssd_runner.py`):** Loaded 1 layer at a time from disk (`52.9 MB → 18.8 MB` weights in RAM).
5. **iPhone Core ML (`export_iphone.py`) & French Hedwig (`lang_hedwig.py`):** 30/30 exact piece match on Core ML + a separate 2.56M French brain on Dumas & Verne.

## Act 5 — Renting a Real Cloud GPU & The Big Scale-Up (5:45–7:00)
* **Smoke test on RunPod:** Rented 1× RTX A5000 (`$0.28/hr`, spent `~$0.10`). What took **59 minutes** on the Mac finished in **2 minutes 15 seconds** (**26× faster**). Show `docs/images/runpod-pod-running.png` and `docs/images/runpod-first-train.png`.
* **The Finale (record after your next RunPod run):** `bash cloud_train.sh --124m` — scaling Hedwig from **14M → 91.3M knobs** (~`$1–$2`), then comparing **14M vs 91M** side-by-side on the exact same prompts in `app.py`.

## Act 6 — End Product & Open Source (7:00–7:30)
* Show `docs/images/hedwig-next-steps-board.png`.
* **Use:** Offline, private, kid-safe bedtime storyteller + interactive Brain X-Ray learning tool.
* **Open Source:** MIT-licensed code on GitHub + weights & live demo on Hugging Face.

---

## B-Roll / Asset Checklist
- [x] `docs/LEARNINGS.md` (all 32 measured milestones)
- [x] `how-hedwig-learns.html` (updated through Stage 3, Mac experiments, & RunPod)
- [x] `docs/images/runpod-pod-running.png` & `docs/images/runpod-first-train.png`
- [x] `docs/images/hedwig-next-steps-board.png` (strategy & next-steps visual board)
- [ ] 20-second screen recording of `app.py` streaming + clicking a word for **🔬 Brain X-Ray**
- [ ] Screenshot of the `"Harry Potter"` → `"potatoes / pot of soup"` prompt + X-Ray
- [ ] Side-by-side recording of **14M vs 91M** after the RunPod `--124m` run
