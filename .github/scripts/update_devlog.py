"""
Adds the commits from one `git push` to logs/DEVLOG.md, grouped by day.

GitHub runs this automatically (see .github/workflows/devlog.yml).
It reads the pushed commits from the COMMITS environment variable (JSON).
"""
import json, os

LOG = "logs/DEVLOG.md"

commits = json.loads(os.environ.get("COMMITS") or "[]")
text = open(LOG).read() if os.path.exists(LOG) else "# Hedwig dev log\n"

for c in commits:
    if c["author"]["name"] == "github-actions[bot]":
        continue                                   # don't log the bot's own commits
    day  = c["timestamp"][:10]                     # "2026-11-16" (your local date)
    hour = c["timestamp"][11:16]                   # "14:32"
    title, _, body = c["message"].partition("\n")
    files = c.get("added", []) + c.get("modified", []) + c.get("removed", [])

    heading = f"## {day}"
    if heading not in text:
        text = text.rstrip() + f"\n\n{heading}\n"

    entry = f"- **{hour}** {title} (`{c['id'][:7]}`)\n"
    body = body.strip()
    if body:
        entry += "".join(f"  > {line}\n" for line in body.splitlines() if line.strip()
                         and not line.startswith("Co-Authored-By"))
    if files:
        entry += f"  - files: {', '.join(files)}\n"

    # put the entry at the end of that day's section
    start = text.index(heading)
    nxt = text.find("\n## ", start + len(heading))
    end = len(text) if nxt == -1 else nxt
    section = text[start:end].rstrip() + "\n" + entry
    text = text[:start] + section + ("\n" + text[end:].lstrip("\n") if nxt != -1 else "")

open(LOG, "w").write(text.rstrip() + "\n")
print(f"Logged {len(commits)} commit(s)")
