"""
Phase 5 + Stretch Goals for Hedwig: Offline Mac Chat UI with:
  - Live Word-by-Word Streaming & KV Cache (~90–180 tok/s)
  - 7 Switchable Brains: Stage 3 DPO, 26.2M Mixture-of-Experts (8 Experts Top-2),
    3.1M Distilled Baby Hedwig, Stage 2 SFT, 8-bit (14 MB), 4-bit (7.9 MB), Stage 1 Base
  - Interactive "🔬 Brain X-Ray": click ANY word in Hedwig's story to inspect the Top-5
    candidate words (with % probability bars) and which 2 MoE Experts woke up for that word!

Run:   .venv/bin/python app.py
"""
import json, os, sys, time, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import tinygpt, sft, quantize, moe, distill

PORT = 8000
MODELS = {}

BRAIN_SPECS = {
    "dpo_16":  ("Stage 3 DPO Chat (13.8M · 27.8 MB)",              "brain_dpo.pt",       "chat", 16, "dense"),
    "moe_16":  ("MoE 8-Experts Top-2 (26.2M total / 11.6M active)","brain_moe.pt",       "chat", 16, "moe"),
    "baby_16": ("🐣 Baby Hedwig Distilled (3.1M · 6.2 MB)",        "brain_baby.pt",      "chat", 16, "baby"),
    "chat_16": ("Stage 2 SFT Chat (13.8M · 27.8 MB)",              "brain_chat.pt",      "chat", 16, "dense"),
    "chat_8":  ("8-bit Quantized Chat (13.8M · 14.0 MB)",          "brain_chat_int8.pt", "chat", 8,  "dense"),
    "chat_4":  ("4-bit Quantized Chat (13.8M · 7.9 MB)",           "brain_chat_int4.pt", "chat", 4,  "dense"),
    "base_16": ("Stage 1 Autocomplete Base (13.8M · 27.8 MB)",     "brain.pt",           "base", 16, "dense"),
}

def get_model(key):
    """Lazy-load and cache the requested brain (dense, MoE, distilled Baby, or quantized) on M4 GPU."""
    if key not in BRAIN_SPECS:
        key = "dpo_16"
    label, path, mode, bits, arch = BRAIN_SPECS[key]
    if not os.path.exists(path):
        key = "dpo_16" if os.path.exists("brain_dpo.pt") else "chat_16"
        label, path, mode, bits, arch = BRAIN_SPECS[key]
    if key not in MODELS:
        if arch == "moe":
            m = moe.MoEGPT().to(tinygpt.device)
        elif arch == "baby":
            m = distill.BabyGPT().to(tinygpt.device)
        else:
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
<title>🦉 Hedwig — Offline Mac Storyteller & Brain X-Ray</title>
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
#chat{flex:1;overflow-y:auto;padding:22px 20px;display:flex;flex-direction:column;gap:16px;max-width:860px;width:100%;margin:0 auto}
.msg{padding:14px 18px;border-radius:14px;line-height:1.65;white-space:pre-wrap;max-width:92%}
.msg.user{align-self:flex-end;background:var(--accent);color:#0e1226;font-weight:600}
.msg.bot{align-self:flex-start;background:var(--surface);border:1px solid var(--line)}
.meta{margin-top:8px;font-family:var(--mono);font-size:.74rem;color:var(--muted)}
.tok{border-radius:4px;padding:1px 0;cursor:pointer;transition:background .15s}
.tok:hover,.tok.sel{background:rgba(139,155,255,.28);outline:1px solid var(--accent)}
.xray{margin-top:12px;padding:12px 14px;border-radius:10px;background:var(--bg);border:1px solid var(--line);font-size:.84rem;display:none}
.xbar{display:grid;grid-template-columns:95px 1fr 48px;gap:8px;align-items:center;font-family:var(--mono);margin:4px 0}
.xtrack{height:12px;background:var(--surface2);border-radius:6px;overflow:hidden}
.xfill{height:100%;background:var(--accent)}
.xfill.win{background:var(--good)}
.expbadge{display:inline-block;font-family:var(--mono);font-size:.76rem;padding:3px 8px;border-radius:6px;background:var(--surface2);color:var(--amber);border:1px solid var(--line);margin-right:6px}
footer{padding:14px 20px;background:var(--surface);border-top:1px solid var(--line)}
form{max-width:860px;margin:0 auto;display:flex;gap:10px}
form input{flex:1;padding:12px 14px;font-size:1rem}
</style>
</head>
<body>
<header>
  <div class="brand">🦉 Hedwig <span class="badge">7 Brains · KV Cache · 🔬 Live Brain X-Ray</span></div>
  <div class="controls">
    <label>Brain:
      <select id="model">
        <option value="dpo_16">Stage 3 DPO Chat (13.8M · 27.8 MB · 91.7% win-rate)</option>
        <option value="moe_16">Mixture-of-Experts (26.2M / 11.6M active · 8 Experts Top-2)</option>
        <option value="baby_16">🐣 Baby Hedwig Distilled (3.1M knobs · 6.2 MB · 4.4× smaller!)</option>
        <option value="chat_16">Stage 2 SFT Chat (13.8M · 27.8 MB · score 1.754)</option>
        <option value="chat_8">8-bit Quantized Chat (13.8M · 14.0 MB · 4× smaller)</option>
        <option value="chat_4">4-bit Quantized Chat (13.8M · 7.9 MB · 7× smaller)</option>
        <option value="base_16">Stage 1 Autocomplete Base (13.8M · 27.8 MB)</option>
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
  <div class="msg bot">Hello! I'm <b>Hedwig</b>, running 100% offline on your MacBook Air M4.<br>• Switch between all <b>7 brains</b> above (including <b>26.2M Mixture-of-Experts</b> and <b>3.1M Distilled Baby Hedwig</b>).<br>• <b>🔬 Brain X-Ray:</b> After I write a story, <b>click any word</b> to inspect the Top-5 words I was debating between (and which 2 MoE Experts woke up for that word)!</div>
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
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function addMsg(cls, text){
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  const body = document.createElement('div');
  body.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'meta';
  const xray = document.createElement('div');
  xray.className = 'xray';
  d.appendChild(body);
  d.appendChild(meta);
  d.appendChild(xray);
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return {box: d, body, meta, xray};
}
function showXray(bot, steps, idx){
  const s = steps[idx];
  if(!s) return;
  bot.body.querySelectorAll('.tok').forEach((el,i)=>el.classList.toggle('sel', i===idx));
  let html = `<div style="margin-bottom:6px"><b>🔬 Brain X-Ray · Word #${idx+1}:</b> picked <span class="expbadge" style="color:var(--good)">${esc(JSON.stringify(s.piece))}</span></div>`;
  if(s.experts && s.experts.length){
    html += `<div style="margin-bottom:8px"><b>🚦 MoE Traffic Cop (Top-2 of 8 Experts awake):</b> ` +
      s.experts.map(e=>`<span class="expbadge">Expert #${e[0]} (${e[1]}%)</span>`).join('') +
      `<span style="color:var(--muted);font-size:.78rem">(6 other experts slept)</span></div>`;
  }
  html += `<div style="color:var(--muted);margin-bottom:4px">Top 5 candidate word-pieces inside the brain:</div>`;
  for(const [cand, pct] of s.top5){
    const win = cand === s.piece;
    html += `<div class="xbar"><span>${esc(JSON.stringify(cand))}</span><div class="xtrack"><div class="xfill ${win?'win':''}" style="width:${Math.min(100,pct)}%"></div></div><span>${pct.toFixed(1)}%</span></div>`;
  }
  bot.xray.innerHTML = html;
  bot.xray.style.display = 'block';
}
function attachInteractiveTokens(bot, finalText, steps){
  if(!steps || !steps.length) return;
  bot.body.innerHTML = '';
  let built = '';
  steps.forEach((s, idx)=>{
    if(built.length >= finalText.length) return;
    const span = document.createElement('span');
    span.className = 'tok';
    span.textContent = s.piece;
    span.title = 'Click for 🔬 Brain X-Ray';
    span.onclick = ()=>showXray(bot, steps, idx);
    bot.body.appendChild(span);
    built += s.piece;
  });
}
async function send(e){
  e.preventDefault();
  const p = input.value.trim();
  if(!p) return;
  input.value = '';
  addMsg('user', p);
  btn.disabled = true;
  const bot = addMsg('bot', '🦉 ...');
  let lastEvt = null;
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
        lastEvt = evt;
        bot.body.textContent = evt.text;
        bot.meta.textContent = `${evt.label} (${evt.mb.toFixed(1)} MB) · ${evt.tokens} tokens · ${evt.tps.toFixed(0)} tok/s · ${evt.seconds.toFixed(2)}s · 🔬 Click any word for Brain X-Ray`;
        chat.scrollTop = chat.scrollHeight;
      }
    }
    if(lastEvt && lastEvt.xray){
      attachInteractiveTokens(bot, lastEvt.text, lastEvt.xray);
      showXray(bot, lastEvt.xray, Math.min(3, lastEvt.xray.length - 1));
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
            kv_cache = [[None, None] for _ in range(len(model.blocks))]
            T0 = idx.size(1)
            xray = []
            for step in range(180):
                pos = T0 + step - 1
                with torch.no_grad():
                    if step == 0:
                        logits, _ = model(idx[:, -tinygpt.block_size:], kv_cache=kv_cache, start_pos=0)
                    elif pos < tinygpt.block_size:
                        logits, _ = model(idx[:, -1:], kv_cache=kv_cache, start_pos=pos)
                    else:
                        logits, _ = model(idx[:, -tinygpt.block_size:])
                nxt, top5 = tinygpt.sample_next(logits, idx[0].tolist(), temperature=temp, return_top5=True)
                xray.append({"piece": tinygpt.decode([int(nxt[0, 0])]), "top5": top5})
                idx = torch.cat([idx, nxt], dim=1)
                n_tok += 1
                dt = max(1e-4, time.time() - t0)
                text = tinygpt.decode(idx[0].tolist())
                pkt = json.dumps({"text": text, "label": label, "mb": mb, "tokens": n_tok, "seconds": dt, "tps": n_tok / dt, "xray": xray if step == 179 else None}) + "\n"
                self.wfile.write(pkt.encode("utf-8"))
                self.wfile.flush()
        else:
            last_text, last_xray = "", []
            for partial, xray in sft.reply_stream(model, prompt, max_new_tokens=220, temperature=temp, with_xray=True):
                n_tok += 1
                last_text, last_xray = partial, xray
                dt = max(1e-4, time.time() - t0)
                pkt = json.dumps({"text": partial, "label": label, "mb": mb, "tokens": n_tok, "seconds": dt, "tps": n_tok / dt}) + "\n"
                self.wfile.write(pkt.encode("utf-8"))
                self.wfile.flush()
            dt = max(1e-4, time.time() - t0)
            final_pkt = json.dumps({"text": last_text, "label": label, "mb": mb, "tokens": n_tok, "seconds": dt, "tps": n_tok / dt, "xray": last_xray}) + "\n"
            self.wfile.write(final_pkt.encode("utf-8"))
            self.wfile.flush()

if __name__ == "__main__":
    get_model("dpo_16")
    url = f"http://127.0.0.1:{PORT}"
    print(f"🦉 Hedwig Mac Chat UI (7 Brains + 🔬 Brain X-Ray) running at {url} (Ctrl+C to stop)", flush=True)
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
