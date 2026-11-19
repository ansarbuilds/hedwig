"""
Phase 5 for Hedwig: Offline Mac Chat UI with Live Word-by-Word Streaming & KV Cache.

Starts a local web server on http://127.0.0.1:8000 where you can chat with Hedwig,
watch stories stream out word-by-word at ~90 tokens/sec, and switch live between
Stage 3 DPO (27.8 MB), Stage 2 SFT (27.8 MB), 8-bit (14.0 MB), and 4-bit (7.9 MB) brains!

Run:   .venv/bin/python app.py
"""
import json, os, sys, time, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import tinygpt, sft, quantize

PORT = 8000
MODELS = {}

BRAIN_SPECS = {
    "dpo_16":  ("Stage 3 DPO Chat (27.8 MB)", "brain_dpo.pt",       "chat", 16),
    "chat_16": ("Stage 2 SFT Chat (27.8 MB)", "brain_chat.pt",      "chat", 16),
    "chat_8":  ("8-bit Quantized (14.0 MB)",  "brain_chat_int8.pt", "chat", 8),
    "chat_4":  ("4-bit Quantized (7.9 MB)",   "brain_chat_int4.pt", "chat", 4),
    "base_16": ("Stage 1 Base (27.8 MB)",     "brain.pt",           "base", 16),
}

def get_model(key):
    """Lazy-load and cache the requested brain on the M4 GPU."""
    if key not in BRAIN_SPECS:
        key = "dpo_16"
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
  <div class="brand">🦉 Hedwig <span class="badge">13.8M · KV Cache (~90 tok/s) · Offline M4</span></div>
  <div class="controls">
    <label>Brain:
      <select id="model">
        <option value="dpo_16">Stage 3 DPO Chat (27.8 MB · 91.7% preference win-rate)</option>
        <option value="chat_16">Stage 2 SFT Chat (27.8 MB · score 1.754)</option>
        <option value="chat_8">8-bit Quantized Chat (14.0 MB · 4× smaller)</option>
        <option value="chat_4">4-bit Quantized Chat (7.9 MB · 7× smaller)</option>
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
  <span class="chip" onclick="useChip('Write a short story about dragon and castle.')">🐉 Dragon & castle</span>
  <span class="chip" onclick="useChip('Write a short story about owl and tree.')">🦉 Owl & tree</span>
  <span class="chip" onclick="useChip('Once upon a time, a little owl ')">📖 Autocomplete: Once upon a time...</span>
</div>
<div id="chat">
  <div class="msg bot">Hello! I'm <b>Hedwig</b>, running completely offline on your MacBook Air M4 with <b>KV Cache (~90 tokens/sec)</b>, <b>Top-k=40 filtering</b>, and <b>Repetition Penalty (1.15)</b>. Pick a brain above and watch me stream stories word-by-word!</div>
</div>
<footer>
  <form onsubmit="send(event)">
    <input id="prompt" type="text" placeholder="Ask for a story (e.g. Write a short story about dragon and castle)..." autofocus>
    <button id="btn" type="submit">Write Story</button>
  </form>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('prompt');
const btn = document.getElementById('btn');
function useChip(text){ input.value = text; input.focus(); }
function addMsg(cls, text){
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  const body = document.createElement('div');
  body.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'meta';
  d.appendChild(body);
  d.appendChild(meta);
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return {box: d, body, meta};
}
async function send(e){
  e.preventDefault();
  const p = input.value.trim();
  if(!p) return;
  input.value = '';
  addMsg('user', p);
  btn.disabled = true;
  const bot = addMsg('bot', '🦉 ...');
  try {
    const res = await fetch('/api/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: p,
        model: document.getElementById('model').value,
        temperature: parseFloat(document.getElementById('temp').value)
      })
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for(const line of lines){
        if(!line.trim()) continue;
        const evt = JSON.parse(line);
        bot.body.textContent = evt.text;
        bot.meta.textContent = `${evt.label} (${evt.mb.toFixed(1)} MB) · ${evt.tokens} tokens · ${evt.tps.toFixed(0)} tok/s · ${evt.seconds.toFixed(2)}s`;
        chat.scrollTop = chat.scrollHeight;
      }
    }
  } catch(err) {
    bot.body.textContent = 'Error: ' + err;
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
        key = req.get("model", "dpo_16")
        temp = max(0.1, min(1.5, float(req.get("temperature", 0.75))))
        model, label, _, mode, mb = get_model(key)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        t0 = time.time()
        n_tok = 0
        if mode == "base":
            idx = torch.tensor([tinygpt.encode(prompt)], device=tinygpt.device)
            kv_cache = [[None, None] for _ in range(tinygpt.n_layer)]
            T0 = idx.size(1)
            for step in range(180):
                pos = T0 + step - 1
                with torch.no_grad():
                    if step == 0:
                        logits, _ = model(idx[:, -tinygpt.block_size:], kv_cache=kv_cache, start_pos=0)
                    elif pos < tinygpt.block_size:
                        logits, _ = model(idx[:, -1:], kv_cache=kv_cache, start_pos=pos)
                    else:
                        logits, _ = model(idx[:, -tinygpt.block_size:])
                nxt = tinygpt.sample_next(logits, idx[0].tolist(), temperature=temp)
                idx = torch.cat([idx, nxt], dim=1)
                n_tok += 1
                dt = max(1e-4, time.time() - t0)
                text = tinygpt.decode(idx[0].tolist())
                pkt = json.dumps({"text": text, "label": label, "mb": mb, "tokens": n_tok, "seconds": dt, "tps": n_tok / dt}) + "\n"
                self.wfile.write(pkt.encode("utf-8"))
                self.wfile.flush()
        else:
            for partial in sft.reply_stream(model, prompt, max_new_tokens=220, temperature=temp):
                n_tok += 1
                dt = max(1e-4, time.time() - t0)
                pkt = json.dumps({"text": partial, "label": label, "mb": mb, "tokens": n_tok, "seconds": dt, "tps": n_tok / dt}) + "\n"
                self.wfile.write(pkt.encode("utf-8"))
                self.wfile.flush()

if __name__ == "__main__":
    get_model("dpo_16")
    url = f"http://127.0.0.1:{PORT}"
    print(f"🦉 Hedwig Mac Chat UI (Streaming + KV Cache) running at {url} (Ctrl+C to stop)", flush=True)
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
