# Spending $100 on cloud GPUs to learn

Prices are rough (late 2026) and change often. Always check the provider's current price.

## How renting works
1. Make an account on a provider and add a card (set a spending limit if offered).
2. Pick a machine (e.g. "1× H100" or "8× H100") and click Start.
3. Connect from your Mac: SSH from Terminal, or a browser notebook/terminal.
4. `git clone` this repo, install deps, train.
5. **Copy the results back before stopping** (`scp` / `rsync`, or push weights to Hugging Face).
6. **Stop / Terminate the machine.** Billing continues until you do. Delete leftover volumes too.

## Providers
| Provider | Notes |
|---|---|
| Google Colab | Free tier GPU in the browser; great for first experiments; sessions time out |
| RunPod | Simple per-hour pods; community cloud is cheaper |
| Lambda | Clean H100 / 8×H100 instances; often what nanochat docs assume |
| Vast.ai | Marketplace; cheapest, more variable reliability |

Rough prices: RTX 4090/5090 ~$0.30–0.70/hr · A100 ~$1–2/hr · H100 ~$2–4/hr · 8×H100 ~$16–30/hr.

## Suggested $100 plan
| Step | Machine | Time | Rough cost | Goal |
|---|---|---|---|---|
| 1. Smoke test | 1× cheap GPU (4090) | 1 hr | ~$1 | Learn SSH, clone, install, run `tinygpt.py` |
| 2. Scale Hedwig up | 1× H100 | 1–2 hr | ~$5 | 124M BPE Hedwig on TinyStories |
| 3. nanochat speedrun | 8× H100 | ~2–3 hr | ~$40–75 | Full pretrain → SFT → chat pipeline |
| Reserve | — | — | ~$20 | Mistakes, re-runs, forgotten-to-stop insurance |

Tip: debug everything on the Mac or a cheap GPU first. Only rent 8×H100 when the script already runs end to end.

## Safety checklist (read before every run)
- [ ] Code already tested locally on a tiny config
- [ ] Checkpoints save regularly
- [ ] I know how to copy results back
- [ ] Timer/alarm set to stop the machine
- [ ] Machine **terminated**, storage deleted, dashboard shows $0/hr

## Useful commands
```bash
# on the rented machine
git clone https://github.com/ansarbuilds/hedwig.git && cd hedwig
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
nvidia-smi                     # check the GPU
.venv/bin/python tinygpt.py train

# from the Mac: copy a file back
scp -P <port> root@<host>:~/hedwig/brain.pt ./brain-cloud.pt
```
The repo is private, so cloning on a rented machine needs a GitHub personal access token or SSH deploy key. Use a token limited to this repo and revoke it afterwards.
