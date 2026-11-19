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
| 2. Scale Hedwig up | 1× H100 | a few hours; stop early if you want | ~$5–15 | 91.3M knobs, 20,000 steps, one read of 500 MB |
| 3. nanochat speedrun | 8× H100 | ~2–3 hr | ~$40–75 | Full pretrain → SFT → chat pipeline |
| Reserve | — | — | ~$20 | Mistakes, re-runs, forgotten-to-stop insurance |

Tip: debug everything on the Mac or a cheap GPU first. Only rent 8×H100 when the script already runs end to end.

## Safety checklist (read before every run)
- [x] Code already tested locally on a tiny config (`bash cloud_train.sh --dry-run`, 92 s, real brains untouched)
- [x] Checkpoints save regularly, and a second launch resumes instead of deleting them (`--fresh` is the only way to wipe saves)
- [ ] I know how to copy results back
- [ ] Timer/alarm set to stop the machine
- [ ] Machine **terminated**, storage deleted, dashboard shows $0/hr

## Useful commands
```bash
# on the Mac, before spending anything (writes only inside dry_run/)
bash cloud_train.sh --dry-run

# on the rented machine
git clone https://github.com/ansarbuilds/hedwig.git && cd hedwig
bash cloud_train.sh           # 13.8M smoke test
bash cloud_train.sh --124m    # 91.3M knobs, 20,000 steps (one read of the 500 MB pile)
bash cloud_train.sh --fresh   # delete checkpoints and prepared data, then start over

# from the Mac: copy the trained brains back BEFORE terminating the pod!
scp -P <port> root@<host>:~/hedwig/brain*.pt ./
```
## The door key (only you can click this)

The project is a private bookshelf. The rented computer is a stranger, so GitHub will not hand it the books until you lend it a temporary library card.

1. On your Mac, open GitHub → Settings → Developer settings → Fine-grained personal access tokens → Generate new token.
2. Name it `hedwig-cloud`. Set the expiration to 7 days.
3. Repository access: **Only select repositories** → `hedwig`.
4. Permissions: **Contents: Read-only**. Nothing else.
5. Generate, and copy the token once. It looks like a long password. Do not put it in a file in this project, and do not commit it.
6. On the rented machine, run `git clone https://github.com/ansarbuilds/hedwig.git`. When it asks for a username, type `x-access-token`. When it asks for a password, paste the token. The password prompt does not show what you paste. That is normal.
7. After the brains are copied home and the machine is terminated, open the token page again and **delete** `hedwig-cloud`.

Your Mac is already logged in as `kmsmohamedansar`, so you do not need this card on the laptop. It is only for the rented computer.
