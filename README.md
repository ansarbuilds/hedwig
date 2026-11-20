# 🦉 Hedwig

A tiny GPT-style language model built from scratch in pure PyTorch to learn how every piece of an LLM works.

## Progress so far (all trained on a MacBook Air M4 for $0)
- **v1 (Phase 0):** 4.8M-parameter character-level Transformer on Tiny Shakespeare (~16 min, loss `1.481`).
- **v2 (Phase 1 & 2 — Stage 1 Storyteller):** 13.8M-parameter modern Transformer. Reading score on the same test pages: `2.080` after 50 MB, `2.010` after 200 MB, **`1.795`** after a partial read of the full TinyStories file (stopped on purpose). `brain.pt` is the 1.795 reader.
- **v3 (Phase 2.5 — Stage 2 SFT + Stage 3 DPO + 8-bit/4-bit Quantization + Mac UI):**
  - **Stage 2 SFT (`sft.py`):** Fine-tuned on `13,852` instruction → story pairs on top of the `1.795` reader (response loss **`1.438`**, down from `1.723` / `1.7544`).
  - **Stage 3 DPO (`dpo.py`):** Preference tuning on the same pairs. Anchored taste score dropped **`1.422 → 1.297`** (was `1.440`); peak win-rate **`92.5%`** (step 100), saved step 150 at **`88.8%`**.
  - **Quantization & UI (`quantize.py`, `app.py`):** Quantized from scratch to **8-bit (`14.0 MB`, chat `1.4531` / taste `1.4667`)** and **4-bit (`7.9 MB`, chat `1.4765` / taste `1.4897`)**, accelerated with **KV Cache (`89 tok/s`, `1.89×` faster)** + Top-k + repetition penalty, and served in an offline Mac web UI (`app.py`) with live word-by-word streaming and clickable **🔬 Brain X-Ray**.
- **v4 (Stretch Goals Complete on Mac for $0):**
  - **Mixture-of-Experts (`moe.py`):** 8 specialist experts, top-2 routing (`26.2M` total / `11.6M` active knobs, upcycled from `brain_dpo.pt`, loss `2.461 → 1.966`).
  - **Knowledge Distillation (`distill.py`):** `13.8M` Teacher → `3.4M` **🐣 Baby Hedwig** Student (`brain_baby.pt`, `6.8 MB`, loss `7.602 → 2.521`).
  - **Mobile & Cloud Packaging (`export_iphone.py`, `cloud_train.sh`, `how-hedwig-learns.html`):** TorchScript + 4-bit + SwiftUI iPhone bundle, 1-command Dec 2026 cloud GPU script, and updated interactive field guide.
- **v5 (SSD streaming, $0):** `ssd_runner.py` loads one thinking layer at a time off the SSD ("LLM in a Flash"). Weights in memory drop `52.9 MB → 18.8 MB`, speed drops `108 → 33 tok/s`, and the story is identical. Peak process RAM only moves `403 → 372 MB`, because a `28 MB` brain is smaller than PyTorch itself.

```
🧑 You: Tell me a story where a little girl named Lily finds a shiny key in the garden.
🦉 Hedwig (4-bit, 7.9 MB):
Once upon a time, there was a little girl named Lily. She loved to play outside
and explore. One day, she found a shiny metal key in her yard... When they got
home, Lily put the key in her pocket. Timmy was so happy and hugged her tight.
He said, "Thank you, Lily!" They went back home... dreaming of their next adventure.
```

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                                              # open offline Mac Chat UI (DPO / SFT / 8-bit / 4-bit)
.venv/bin/python dpo.py talk "Write a short story about dragon and castle."
.venv/bin/python sft.py chat                                         # interactive terminal chat (`brain_chat.pt`)
.venv/bin/python sft.py talk "Tell me a story where a little girl named Lily finds a shiny key in the garden."
.venv/bin/python quantize.py                                         # quantize to 8-bit (14 MB) & 4-bit (7.9 MB)
.venv/bin/python tinygpt.py talk "Once upon a time, a little owl "   # base autocomplete model (`brain.pt`)
.venv/bin/python prepare_data.py                                     # download stories & build .bin shards
.venv/bin/python tinygpt.py train                                    # Stage 1 pretraining
.venv/bin/python sft.py train                                        # Stage 2 SFT ("chat manners")
.venv/bin/python dpo.py train                                        # Stage 3 DPO ("preference & taste")
.venv/bin/python export_iphone.py                                    # Core ML brain + Swift app for Xcode
.venv/bin/python ssd_runner.py                                       # one layer at a time from SSD vs all-in-RAM
.venv/bin/python mlx_bench.py                                        # MLX vs PyTorch speed on the same 13.8M brain
.venv/bin/python lang_hedwig.py talk "Il était une fois un petit hibou "
bash cloud_train.sh --dry-run                                        # 2-step rehearsal in dry_run/ (does not touch real brains)
bash cloud_train.sh                                                  # rented GPU: full pipeline (or `--124m` for the 91.3M brain)
```

## Files

| Path | What |
|---|---|
| `tinygpt.py` | Stage 1: ~13.8M-param modern Transformer (RoPE, RMSNorm, SwiGLU, bf16, SDPA), pretraining & autocomplete |
| `sft.py` | Stage 2: Supervised Fine-Tuning (prompt loss masking) & interactive story chat (`talk` / `chat`) |
| `dpo.py` | Stage 3: Direct Preference Optimization (👍 chosen vs 👎 rejected + fluency anchor) |
| `moe.py` | Stretch: Mixture-of-Experts Hedwig (`26.2M` total / `11.6M` active knobs, 8 experts top-2) |
| `distill.py` | Stretch: Teacher → Student soft-label distillation (`13.8M` Teacher → `3.4M` `6.8 MB` Baby Hedwig) |
| `quantize.py` | From-scratch 8-bit (`int8`) and 4-bit (packed `int4`) quantizer + quality benchmark |
| `app.py` | Offline Mac Chat UI (`http://127.0.0.1:8000`) with 9 brains, word streaming, & 🔬 Live Brain X-Ray |
| `export_iphone.py` | Builds a Core ML iPhone brain and checks it against PyTorch |
| `ios/HedwigApp.swift` | SwiftUI screen, sampler, and the Xcode clicks for a free Apple ID |
| `ios/HedwigTokenizer.swift` | Swift word-piece dictionary, same numbers as `tokenizer.py` |
| `lang_hedwig.py` | French Hedwig (2.56M) on public-domain books, saved under `lang/` |
| `ssd_runner.py` | Loads one thinking layer at a time from `ssd_layers/` and compares RAM vs speed |
| `mlx_bench.py` | Times the same 13.8M brain in Apple MLX and in PyTorch |
| `cloud_train.sh` | Cloud pipeline. `--dry-run` rehearses locally; `--124m` is the 91.3M scale-up; restarts resume |
| `prepare_data.py` | Downloads TinyStories + 10 Project Gutenberg books, trains 4,096-vocab BPE, writes `uint16` `.bin` shards |
| `tokenizer.py` | From-scratch word-chunked BPE tokenizer (`minbpe` / GPT-2 style) |
| `tokenizer.json` | Learned BPE merges (4,096 vocab on TinyStories + Project Gutenberg) |
| `brain.pt` | Stage 1 pretrained storyteller weights (`27.8 MB` in `bfloat16`, val loss `1.795`) |
| `brain_chat.pt` | Stage 2 SFT chat weights (`27.8 MB` in `bfloat16`, response loss `1.438`) |
| `brain_dpo.pt` | Stage 3 DPO preference-tuned weights (`27.8 MB` in `bfloat16`, anchored loss `1.297`, `92.5%` peak win-rate) |
| `brain_chat_int8.pt` | 8-bit quantized chat weights (`14.0 MB`, `3.96×` smaller, response loss `1.4531`) |
| `brain_chat_int4.pt` | 4-bit quantized chat weights (`7.9 MB`, `7.05×` smaller, response loss `1.4765`) |
| `brain_dpo_int8.pt` | 8-bit Stage 3 taste brain (`14.0 MB`, response loss `1.4667`) |
| `brain_dpo_int4.pt` | 4-bit Stage 3 taste brain (`7.9 MB`, response loss `1.4897`) |
| `brain_baby.pt` | Distilled 3.4M Baby Hedwig weights (`6.8 MB` in `bfloat16`, loss `2.521`) |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB) |
| `how-hedwig-learns.html` | Interactive visual field guide updated with all Phase 1–5, Stage 1–3, Mac benchmarks, and RunPod results |
| `docs/LEARNINGS.md` | Concepts and 32 sections of measured results |
| `docs/ROADMAP.md` | Complete project plan & stretch checklist |
| `docs/CLOUD_GPU_100.md` | Cloud GPU guide, RunPod smoke test results, & safety checklist |
| `docs/VIDEO_SCRIPT.md` | Full narrative video / case-study storyboard & B-roll checklist |
| `docs/images/hedwig-next-steps-board.png` | Visual roadmap board: where we are, why scale up, budget, uses & open-source plan |
| `logs/DEVLOG.md` | Automatic daily dev log (updated on push) |
| `AGENTS.md` | Context for AI coding assistants |

## Next Steps & Plan
- **All core $0 MacBook Air milestones are complete ✅!**
- **Optional $0 Mac experiments (anytime before Dec 2026):**
  1. ~~**Layer-by-layer SSD streaming (`ssd_runner.py`):**~~ Done. `52.9 MB → 18.8 MB` of weights in memory, `108 → 33 tok/s`, same story. Process RAM only `403 → 372 MB`.
  2. ~~**Apple MLX vs PyTorch MPS benchmark (`mlx_bench.py`):**~~ Done. Practice step `596 ms → 99 ms` (16-bit MLX). Writing `635 → 832` word-pieces/sec. The greedy story matches.
  3. ~~**Second-language Hedwig (`lang_hedwig.py`):**~~ Done. A 2.56M French brain on public-domain Dumas and Verne, val `7.776 → 3.541`. The English brain is unchanged.
  4. ~~**iPhone brain (`export_iphone.py`):**~~ Core ML matches PyTorch 30/30 pieces. Xcode clicks are at the top of `ios/HedwigApp.swift`. Press Play there when a phone or simulator is open.
- **Next Month (Dec 2026 — ~$15–100 Cloud GPU Sprint):**
  1. ~~Local rehearsal of `cloud_train.sh --dry-run`~~ Done (92 s, checkpoints resume, `--124m` counts as 91,286,272 knobs).
  2. ~~Rent 1× GPU and run `bash cloud_train.sh`~~ Done 20 Nov 2026. RTX A5000, reading score `2.100` in 135 s, about `$0.10`. See [docs/CLOUD_GPU_100.md](docs/CLOUD_GPU_100.md).
  3. Run `bash cloud_train.sh --124m` for the 91.3M brain: 20,000 steps, one read of 500 MB (about 2 word-pieces per knob, same as the Mac brain). Copy the files home. See [docs/ROADMAP.md](docs/ROADMAP.md) and the door-key steps in [docs/CLOUD_GPU_100.md](docs/CLOUD_GPU_100.md).
