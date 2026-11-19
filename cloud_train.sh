#!/usr/bin/env bash
# ==============================================================================
# Hedwig 1-Command Cloud GPU Pipeline (for the Dec 2026 rented GPU)
#
# Runs the whole storytelling pipeline:
#   1. Creates .venv and checks for an NVIDIA GPU
#   2. Downloads TinyStories + 10 Project Gutenberg novels, writes uint16 .bin shards
#   3. Stage 1 pretrain  (tinygpt.py  -> brain.pt)
#   4. Stage 2 SFT       (sft.py      -> brain_chat.pt)
#   5. Stage 3 DPO       (dpo.py      -> brain_dpo.pt)
#   6. 8-bit and 4-bit   (quantize.py -> brain_chat_int8.pt, brain_chat_int4.pt)
#
# A crash no longer deletes the save. Run the same command again and training
# continues from ckpt.pt / ckpt_sft.pt / ckpt_dpo.pt. Data prep is skipped when
# the same download size was already finished.
#
# Usage:
#   bash cloud_train.sh --dry-run   # Mac rehearsal: 2 steps in ./dry_run, real brains untouched
#   bash cloud_train.sh             # rented GPU: 13.8M smoke test
#   bash cloud_train.sh --124m      # rented GPU: 12 layers, 768 wide, 500 MB of stories (~91.3M knobs)
#   bash cloud_train.sh --fresh     # throw away checkpoints and prepared data, then start over
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

FRESH=0
DRY=0
SCALE=0
for arg in "$@"; do
  case "$arg" in
    --fresh)   FRESH=1 ;;
    --dry-run) DRY=1 ;;
    --124m)    SCALE=1 ;;
    *) echo "Unknown option: $arg" >&2; echo "Use: --dry-run, --124m, --fresh" >&2; exit 1 ;;
  esac
done

# The Mac holds the real brains. Training belongs on a rented GPU, or inside
# the sandbox started by --dry-run.
if [[ "$DRY" -eq 0 ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU found. Refusing to train in this folder."
  echo "Your finished brains (brain.pt, brain_chat.pt, brain_dpo.pt) live here."
  echo "Rehearse the pipeline with:  bash cloud_train.sh --dry-run"
  exit 1
fi

echo "🦉 Hedwig Cloud GPU Pipeline starting..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install --upgrade pip
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
fi
PY="$ROOT/.venv/bin/python"

# Replace one exact setting. Stop if the line is missing, so a stale script
# cannot pretend it scaled the model up and then train the small one.
apply_patch() {
  local path="$1" old="$2" new="$3"
  "$PY" - "$path" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1:]
file = open(path).read()
if old not in file:
    raise SystemExit(
        f"Refusing to start. {path} does not contain this line:\n  {old}\n"
        "A silent skip here would train the wrong model."
    )
open(path, "w").write(file.replace(old, new, 1))
print(f"  patched {path}: {old.strip()}  ->  {new.strip()}")
PY
}

# Prove the scale-up edits on a throwaway copy, then print the real knob count.
# 91,286,272 is width 768, 12 heads, 12 layers, vocab 4,096, SwiGLU hidden 2048.
check_124m_on_copy() {
  local tmp
  tmp="$(mktemp -d)"
  cp "$ROOT/tinygpt.py" "$ROOT/prepare_data.py" "$ROOT/tokenizer.py" "$ROOT/tokenizer.json" "$tmp/"
  apply_patch "$tmp/tinygpt.py"      "n_embd     = 384" "n_embd     = 768"
  apply_patch "$tmp/tinygpt.py"      "n_head     = 6"   "n_head     = 12"
  apply_patch "$tmp/tinygpt.py"      "n_layer    = 6"   "n_layer    = 12"
  apply_patch "$tmp/tinygpt.py"      "steps      = 3000" "steps      = 6000"
  apply_patch "$tmp/prepare_data.py" "tinystories_train_mb = 50" "tinystories_train_mb = 500"
  (cd "$tmp" && "$PY" - <<'PY'
import tinygpt
n = sum(p.numel() for p in tinygpt.TinyGPT().parameters())
print(f"Scale-up check: n_embd={tinygpt.n_embd} n_head={tinygpt.n_head} n_layer={tinygpt.n_layer} steps={tinygpt.steps} knobs={n:,}")
assert (tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.steps) == (768, 12, 12, 6000)
assert n == 91_286_272, n
PY
  )
  rm -rf "$tmp"
}

if [[ "$SCALE" -eq 1 && "$DRY" -eq 1 ]]; then
  echo "Dry-run ignores --124m for the tiny rehearsal, and still checks that the scale-up edits work."
  SCALE=0
  check_124m_on_copy
elif [[ "$DRY" -eq 1 ]]; then
  check_124m_on_copy
fi

if [[ "$DRY" -eq 1 ]]; then
  SANDBOX="$ROOT/dry_run"
  rm -rf "$SANDBOX"
  mkdir -p "$SANDBOX"
  cp "$ROOT/prepare_data.py" "$ROOT/tinygpt.py" "$ROOT/sft.py" "$ROOT/dpo.py" \
     "$ROOT/quantize.py" "$ROOT/tokenizer.py" "$ROOT/tokenizer.json" "$SANDBOX/"
  cd "$SANDBOX"
  echo "Rehearsing inside $SANDBOX (2 steps, a few MB of stories). Real brains are not touched."
  apply_patch prepare_data.py "tinystories_train_mb = 50" "tinystories_train_mb = 4"
  apply_patch prepare_data.py "tinystories_val_mb   = 10" "tinystories_val_mb   = 2"
  apply_patch prepare_data.py "bpe_sample_mb        = 4"  "bpe_sample_mb        = 2"
  apply_patch tinygpt.py "steps      = 3000" "steps      = 2"
  apply_patch tinygpt.py "ckpt_every = 250"  "ckpt_every = 1"
  apply_patch sft.py "steps        = 750" "steps        = 2"
  apply_patch sft.py "eval_every   = 150" "eval_every   = 1"
  apply_patch sft.py "train_mb     = 25"  "train_mb     = 4"
  apply_patch sft.py "val_mb       = 4"   "val_mb       = 2"
  apply_patch dpo.py "steps      = 150" "steps      = 2"
  apply_patch dpo.py "eval_every = 50"  "eval_every = 1"
  apply_patch quantize.py "n_batches=30" "n_batches=2"
  apply_patch quantize.py "sft.reply(model, prompt)" "sft.reply(model, prompt, max_new_tokens=20)"
fi

if [[ "$SCALE" -eq 1 ]]; then
  echo "Scaling config to 12 layers, 768 wide, 6,000 pretrain steps, 500 MB of TinyStories."
  apply_patch tinygpt.py      "n_embd     = 384" "n_embd     = 768"
  apply_patch tinygpt.py      "n_head     = 6"   "n_head     = 12"
  apply_patch tinygpt.py      "n_layer    = 6"   "n_layer    = 12"
  apply_patch tinygpt.py      "steps      = 3000" "steps      = 6000"
  apply_patch prepare_data.py "tinystories_train_mb = 50" "tinystories_train_mb = 500"
  "$PY" - <<'PY'
import tinygpt
n = sum(p.numel() for p in tinygpt.TinyGPT().parameters())
print(f"Scale-up is in effect: {n:,} knobs")
assert (tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.steps) == (768, 12, 12, 6000)
assert n == 91_286_272, n
PY
fi

if [[ "$FRESH" -eq 1 ]]; then
  echo "Fresh start: removing checkpoints and prepared data in $(pwd)"
  rm -f ckpt.pt ckpt.pt.tmp ckpt_sft.pt ckpt_sft.pt.tmp ckpt_dpo.pt ckpt_dpo.pt.tmp
  rm -f data/train.bin data/val.bin data/prepare_stamp.txt
  rm -f data/tinystories_train.txt data/tinystories_val.txt data/tinystories_train.txt.mb data/tinystories_val.txt.mb
  rm -f data/sft_cache.pt data/dpo_cache.pt data/sft_train.txt data/sft_val.txt
fi

# The stamp is the download size written into prepare_data.py. Same size on a
# second launch means the .bin files are already done, so we do not rebuild them.
prepare_if_needed() {
  local want
  want="$("$PY" -c 'import prepare_data; print(prepare_data.tinystories_train_mb)')"
  if [[ -f data/train.bin && -f data/val.bin && -f data/prepare_stamp.txt ]] \
     && [[ "$(cat data/prepare_stamp.txt)" == "$want" ]]; then
    echo "Stories already prepared (TinyStories train slice = ${want} MB). Skipping download. Use --fresh to redo."
    return
  fi
  echo "Preparing stories (TinyStories train slice = ${want} MB)..."
  rm -f data/prepare_stamp.txt data/train.bin data/val.bin
  rm -f data/tinystories_train.txt data/tinystories_val.txt
  "$PY" -u prepare_data.py
  echo "$want" > data/prepare_stamp.txt
}

prepare_if_needed
if [[ "$DRY" -eq 1 ]]; then
  skip_log="$(prepare_if_needed)"
  echo "$skip_log"
  echo "$skip_log" | grep -q "Stories already prepared" || { echo "Second prepare should have been skipped."; exit 1; }
fi

echo "Checkpoints: $(ls ckpt.pt ckpt_sft.pt ckpt_dpo.pt 2>/dev/null || echo none yet). They are kept across restarts."

"$PY" -u tinygpt.py train
"$PY" -u sft.py train
"$PY" -u dpo.py train
"$PY" -u quantize.py

if [[ "$DRY" -eq 1 ]]; then
  for f in brain.pt brain_chat.pt brain_dpo.pt brain_chat_int8.pt brain_chat_int4.pt ckpt.pt ckpt_sft.pt ckpt_dpo.pt; do
    [[ -f "$f" ]] || { echo "Dry-run failed: $f was not written."; exit 1; }
  done
  echo "--- resume check (the saves must still be here) ---"
  for pair in "tinygpt.py train|Resuming from step" "sft.py train|Resuming SFT from step" "dpo.py train|Resuming DPO from step"; do
    cmd="${pair%%|*}"
    expect="${pair##*|}"
    log="$(mktemp)"
    # shellcheck disable=SC2086
    "$PY" -u $cmd >"$log" 2>&1 || { cat "$log"; exit 1; }
    cat "$log"
    grep -q "$expect" "$log" || { echo "Resume failed: did not see '$expect'."; exit 1; }
    rm -f "$log"
  done
  echo "Dry-run passed. Sandbox is $SANDBOX. The real brains in $ROOT were not changed."
fi

echo "✅ All done. Copy the brains back to the Mac BEFORE stopping the machine:"
echo "   scp -P <port> root@<host>:~/hedwig/brain*.pt ./"
echo "A 12-layer brain is about 180 MB in 16-bit. Do not git commit it (GitHub's limit is 100 MB)."
