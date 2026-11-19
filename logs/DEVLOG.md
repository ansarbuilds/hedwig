# Hedwig dev log

A diary of the project, one heading per day.

Every time you `git push`, a GitHub Action adds your commit messages here
under that day's date. Run `git pull` afterwards to get the updated log on your Mac.

Tip: write clear commit messages. They become this diary.
You can also add your own notes under any day (pull first, then edit, commit, push).

## 2026-11-16
- **21:32** Add checkpoint save/resume (Ctrl+C at 750, resumed 1.622, final 1.481) and automatic dev log (`9590f26`)
- **21:55** Add BPE tokenizer from scratch: 512 vocab, 1.96x shorter, round trip OK (`04eaa40`)
- **23:06** Refresh docs for checkpointing and BPE tokenizer (`b0b4889`)
- **23:53** Wire 512-vocab BPE into tinygpt: best val 3.007 at step 750, overfit to 3.429 (`dac40fe`)

## 2026-11-18
- **20:25** Add best-val brain.pt saving and RoPE (3.013 at step 500 vs 3.007 at step 750) (`b20f173`)
- **21:15** Complete Phase 1: add RMSNorm, SwiGLU, bf16 autocast, and SDPA (`14cbff2`)
- **22:04** Complete Phase 2: TinyStories + Gutenberg pipeline, 4096-vocab BPE, 13.8M storyteller (val 2.276 at step 1500) (`4bb7820`)
- **22:36** Finish 3000-step 13.8M storyteller run (val 2.080) and save brain.pt in 27.8 MB bfloat16 (`b70b807`)
- **22:54** Add Stage 2 SFT (sft.py) and 13.8M chat storyteller weights (brain_chat.pt, loss 1.766) (`c6232e6`)
- **22:59** Add Phase 5 8-bit/4-bit quantization (14 MB / 7.9 MB) and offline Mac Chat UI (app.py) (`8b0c1cd`)
- **23:00** Add sys import to app.py (`2bf602d`)
- **23:04** Document full progress (v1 -> v3 SFT/int8/int4/UI), zsh Mac next steps, and Dec 2026 cloud GPU plan (`b6b6f39`)
- **23:22** Complete Stage 3 DPO (dpo.py, brain_dpo.pt, 91.7% preference win-rate) and add to Mac Chat UI (`a5b12bc`)

## 2026-11-19
- **00:17** Refresh how-hedwig-learns.html, add 1-command cloud_train.sh, and add export_iphone.py (`118ab51`)
- **00:25** Add KV cache (89 tok/s, 1.89x faster), top-k + repetition penalty, and word-by-word streaming in app.py (`79e2742`)
- **01:13** Add Mixture-of-Experts (moe.py, 26.2M/11.6M), Distilled Baby Hedwig (distill.py, 3.4M/6.8MB), and Live Brain X-Ray in app.py (`cb4bab4`)
- **01:21** Update README and AGENTS with full v1-v4 progress, optional Mac experiments, and Dec 2026 Cloud GPU plan (`4a06f07`)
  > Completed all core zsh Mac milestones (Pretrain -> SFT -> DPO -> Quantize -> MoE -> Distill -> KV-Cache Streaming UI + Brain X-Ray + iPhone export + cloud_train.sh). Next: optional SSD/MLX/multilingual experiments on Mac, then Dec 2026 cloud GPU scale-up (--124m + nanochat).
- **01:29** Add layer-by-layer SSD loading so we can measure RAM against speed. (`e6173fb`)
  > Streaming one thinking layer at a time cuts weights in memory from 53 MB to 19 MB and slows generation from 108 to 33 tok/s, with the same story.
  > Co-authored-by: Cursor <cursoragent@cursor.com>
- **11:08** Make the cloud script resumable and prove it with a tiny local rehearsal. (`1e37ee9`)
  > A crash no longer deletes checkpoints, the scale-up switch must really yield 91.3M knobs, and --dry-run finishes every stage in a sandbox without touching the real brains.
  > Co-authored-by: Cursor <cursoragent@cursor.com>
- **11:21** Give the bigger cloud brain a full read, and pack the taste brain small. (`f40c010`)
  > 20,000 steps matches the Mac brain's two word-pieces per knob, and the Stage 3 brain now has 8-bit and 4-bit copies with almost the same story score.
  > Co-authored-by: Cursor <cursoragent@cursor.com>
- **11:29** Measure the same 13.8M brain in Apple MLX and in PyTorch. (`5265411`)
  > A practice step drops from 596 ms to 99 ms in 16-bit MLX, story writing is a little faster, and the most-likely story matches piece for piece.
  > Co-authored-by: Cursor <cursoragent@cursor.com>
- **11:42** Add a real Core ML iPhone brain and a small French Hedwig. (`10cb900`)
  > The phone model matches PyTorch on 30 greedy pieces, the Swift dictionary matches Python, and a 2.56M French brain trained on public-domain Dumas and Verne drops its val score from 7.78 to 3.54.
  > Co-authored-by: Cursor <cursoragent@cursor.com>
