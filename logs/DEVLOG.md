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
