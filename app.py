"""
Phase 5 for Hedwig: Simple offline Mac Chat UI (zero external dependencies).

Starts a local web server on http://127.0.0.1:8000 where you can chat with Hedwig
and switch live between the 16-bit (27.8 MB), 8-bit (13.9 MB), and 4-bit (7.8 MB)
quantized story brains!

Run:   .venv/bin/python app.py
"""
import json, os, sys, time, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch, torch.nn.functional as F
import tinygpt, sft, quantize

PORT = 8000
MODELS = {}

BRAIN_SPECS = {
    "chat_16": ("16-bit Chat (27.8 MB)", "brain_chat.pt",      "chat", 16),
    "chat_8":  ("8-bit Chat (13.9 MB)",  "brain_chat_int8.pt", "chat", 8),
    "chat_4":  ("4-bit Chat (7.8 MB)",   "brain_chat_int4.pt", "chat", 4),
    "base_16": ("Stage 1 Base (27.8 MB)","brain.pt",           "base", 16),
}

def get_model(key):
    """Lazy-load and cache the requested brain on the M4 GPU."""
    if key not in BRAIN_SPECS:
        key = "chat_16"
    if key not in MODELS:
        label, path, mode, bits = BRAIN_SPECS[key]
        if not os.path.exists(path):
            path = "brain_chat.pt" if os.path.exists("brain_chat.pt") else "brain.pt"
            bits = 16
        m = tinygpt.TinyGPT().to(tinygpt.device)
        sd = torch.load(path, map_location="cpu")
        if bits in (4, 8):
            sd = quantize.dequantize(sd)
        m.load_state_dict(sd)
        m.eval()
        MODELS[key] = (m, label, path, mode, os.path.getsize(path) / 1e6)
    return MODELS[key]

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>🦉 Hedwig — Offline Mac Storyteller</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#0e1226; --surface:#171c36; --surface2:#1f2547; --ink:#e8ebfa; --muted:#9aa1c4;
  --line:#2b3255; --accent:#8b9bff; --amber:#ffbe4d; --good:#4fd39a;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;background:var(--surface);border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:1.15rem}
.badge{font-family:var(--mono);font-size:.75rem;padding:4px 9px;border-radius:99px;background:var(--surface2);color:var(--amber);border:1px solid var(--line)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;font-size:.88rem;color:var(--muted)}
select,input[type=text],button{font:inherit;border-radius:10px;border:1px solid var(--line);background:var(--surface2);color:var(--ink);padding:8px 12px}
button{background:var(--accent);color:#0e1226;font-weight:700;border:0;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.chipbar{padding:10px 20px;display:flex;flex-wrap:wrap;gap:8px;border-bottom:1px solid var(--line);background:rgba(23,28,54,.5)}
.chip{font-size:.82rem;padding:6px 11px;border-radius:99px;background:var(--surface2);color:var(--ink);border:1px solid var(--line);cursor:pointer}
.chip:hover{border-color:var(--accent)}
#chat{flex:1;overflow-y:auto;padding:22px 20px;display:flex;flex-direction:column;gap:16px;max-width:820px;width:100%;margin:0 auto}
.msg{padding:14px 18px;border-radius:14px;line-height:1.6;white-space:pre-wrap;max-width:88%}
.msg.user{align-self:flex-end;background:var(--accent);color:#0e1226;font-weight:600}
.msg.bot{align-self:flex-start;background:var(--surface);border:1px solid var(--line)}
.meta{margin-top:8px;font-family:var(--mono);font-size:.74rem;color:var(--muted)}
footer{padding:14px 20px;background:var(--surface);border-top:1px solid var(--line)}
form{max-width:820px;margin:0 auto;display:flex;gap:10px}
form input{flex:1;padding:12px 14px;font-size:1rem}
</style>
</head>
<body>
<header>
  <div class="brand">🦉 Hedwig <span class="badge">13.8M knobs · 100% Offline on Mac M4</span></div>
  <div class="controls">
    <label>Brain:
      <select id="model">
        <option value="chat_16">16-bit Chat (27.8 MB · score 1.766)</option>
        <option value="chat_8">8-bit Quantized Chat (13.9 MB · 4× smaller)</option>
        <option value="chat_4">4-bit Quantized Chat (7.8 MB · 7× smaller)</option>
        <option value="base_16">Stage 1 Autocomplete Base (27.8 MB)</option>
      </select>
    </label>
    <label>Creativity: <span id="tval" style="font-family:var(--mono)">0.75</span>
      <input id="temp" type="range" min="0.3" max="1.1" step="0.05" value="0.75" style="width:90px;vertical-align:middle" oninput="tval.textContent=this.value">
    </label>
  </div>
</header>
<div class="chipbar">
  <span class="chip" onclick="useChip('Tell me a story where a little girl named Lily finds a shiny key in the garden.')">🔑 Lily & the shiny key</span>
  <span class="chip" onclick="useChip('Write a story: A little bird named Pip learns to fly high above the forest with his mom.')">🐦 Pip learns to fly</span>
  <span class="chip" onclick="useChip('Write a short story about dog and ball.')">🐶 Dog & ball</span>
  <span class="chip" onclick="useChip('Once upon a time, a little owl ')">🦉 Once upon a time, a little owl...</span>
</div>
<div id="chat">
  <div class="msg bot">Hello! I'm <b>Hedwig</b>, your 13.8-million-knob storytelling model running completely offline on your MacBook Air M4. Pick a 16-bit, 8-bit, or 4-bit brain above and ask me for a story!</div>
</div>
<footer>
  <form onsubmit="send(event)">
    <input id="prompt" type="text" placeholder="Ask for a story (e.g. Tell me a story where a brave little owl finds a treasure)..." autofocus>
    <button id="btn" type="submit">Write Story</button>
  </form>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('prompt');
const btn = document.getElementById('btn');
function useChip(text){ input.value = text; input.focus(); }
function addMsg(cls, text, meta=''){
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  if(meta){
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    d.appendChild(m);
  }
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}
async function send(e){
  e.preventDefault();
  const p = input.value.trim();
  if(!p) return;
  input.value = '';
  addMsg('user', p);
  btn.disabled = true;
  const thinking = addMsg('bot', '🦉 Hedwig is writing...');
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: p,
        model: document.getElementById('model').value,
        temperature: parseFloat(document.getElementById('temp').value)
      })
    });
    const data = await res.json();
    thinking.textContent = data.story;
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = `${data.label} (${data.mb.toFixed(1)} MB) · ${data.seconds.toFixed(2)}s`;
    thinking.appendChild(m);
  } catch(err) {
    thinking.textContent = 'Error: ' + err;
  } finally {
    btn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def do_GET(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = req.get("prompt", "Write a short story about a little owl.")
        key = req.get("model", "chat_16")
        temp = max(0.1, min(1.5, float(req.get("temperature", 0.75))))
        model, label, _, mode, mb = get_model(key)
        t0 = time.time()
        if mode == "base":
            idx = torch.tensor([tinygpt.encode(prompt)], device=tinygpt.device)
            story = tinygpt.decode(model.generate(idx, 180, temperature=temp)[0].tolist())
        else:
            story = sft.reply(model, prompt, max_new_tokens=220, temperature=temp)
        out = json.dumps({"story": story, "label": label, "mb": mb, "seconds": time.time() - t0}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

if __name__ == "__main__":
    get_model("chat_16")
    url = f"http://127.0.0.1:{PORT}"
    print(f"🦉 Hedwig Mac Chat UI running at {url} (Ctrl+C to stop)", flush=True)
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
