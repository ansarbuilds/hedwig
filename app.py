"""
Phase 5 + Stretch Goals for Hedwig: Offline Mac Chat UI with:
  - Live Word-by-Word Streaming & KV Cache (~90–180 tok/s)
  - Switchable Brains: Stage 3 DPO at 16-bit / 8-bit / 4-bit, 26.2M Mixture-of-Experts (8 Experts Top-2),
    3.1M Distilled Baby Hedwig, Stage 2 SFT, 8-bit (14 MB), 4-bit (7.9 MB), Stage 1 Base
  - Interactive "🔬 Brain X-Ray": click ANY word in Hedwig's story to inspect the Top-5
    candidate words (with % probability bars) and which 2 MoE Experts woke up for that word!

Run:   .venv/bin/python app.py
"""
import json, os, sys, time, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import torch
import tokenizer, tinygpt, sft, quantize, moe, distill

HOST = os.environ.get("HEDWIG_HOST", "127.0.0.1")   # the Hugging Face Space sets 0.0.0.0 so visitors can reach it
PORT = int(os.environ.get("HEDWIG_PORT", "8000"))  # the Space uses 7860
MODELS = {}
TOKENIZERS = {}

BRAIN_SPECS = {
    "c300_dpo_16":  ("🏆 286.7M Flagship Hedwig — Stories + Grammar (DPO 16-bit · 575 MB)", "cloud_brains_300m/brain_dpo.pt",      "chat", 16, "cloud300"),
    "c300_dpo_4":   ("🏆 286.7M Flagship Hedwig — Stories + Grammar (DPO 4-bit · 162 MB)",  "cloud_brains_300m/brain_dpo_int4.pt", "chat", 4,  "cloud300"),
    "c300_chat_16": ("🏆 286.7M Flagship Hedwig — Stories + Grammar (SFT 16-bit · 575 MB)", "cloud_brains_300m/brain_chat.pt",     "chat", 16, "cloud300"),
    "c300_chat_4":  ("🏆 286.7M Flagship Hedwig — Stories + Grammar (SFT 4-bit · 162 MB)",  "cloud_brains_300m/brain_chat_int4.pt","chat", 4,  "cloud300"),
    "c97_dpo_16":   ("🌟 97.6M Cloud Hedwig — Stories + Grammar (DPO 16-bit · 196 MB)",     "cloud_brains_97m/brain_dpo.pt",       "chat", 16, "cloud97"),
    "c97_dpo_4":    ("🌟 97.6M Cloud Hedwig — Stories + Grammar (DPO 4-bit · 55 MB)",       "cloud_brains_97m/brain_dpo_int4.pt",  "chat", 4,  "cloud97"),
    "c97_chat_16":  ("🌟 97.6M Cloud Hedwig — Stories + Grammar (SFT 16-bit · 196 MB)",     "cloud_brains_97m/brain_chat.pt",      "chat", 16, "cloud97"),
    "c97_chat_4":   ("🌟 97.6M Cloud Hedwig — Stories + Grammar (SFT 4-bit · 55 MB)",       "cloud_brains_97m/brain_chat_int4.pt", "chat", 4,  "cloud97"),
    "dpo_16":       ("Stage 3 DPO Chat (13.8M Mac · 27.8 MB · score 1.297)",                "brain_dpo.pt",                        "chat", 16, "dense"),
    "dpo_8":        ("Stage 3 DPO 8-bit (13.8M Mac · 14.0 MB)",                             "brain_dpo_int8.pt",                   "chat", 8,  "dense"),
    "dpo_4":        ("Stage 3 DPO 4-bit (13.8M Mac · 7.9 MB)",                              "brain_dpo_int4.pt",                   "chat", 4,  "dense"),
    "moe_16":       ("MoE 8-Experts Top-2 (26.2M total / 11.6M active)",                    "brain_moe.pt",                        "chat", 16, "moe"),
    "baby_16":      ("🐣 Baby Hedwig Distilled (3.4M · 6.8 MB)",                             "brain_baby.pt",                       "chat", 16, "baby"),
    "chat_16":      ("Stage 2 SFT Chat (13.8M Mac · 27.8 MB · score 1.438)",                "brain_chat.pt",                       "chat", 16, "dense"),
    "chat_8":       ("8-bit Quantized Chat (13.8M Mac · 14.0 MB)",                          "brain_chat_int8.pt",                  "chat", 8,  "dense"),
    "chat_4":       ("4-bit Quantized Chat (13.8M Mac · 7.9 MB)",                           "brain_chat_int4.pt",                  "chat", 4,  "dense"),
    "base_16":      ("Stage 1 Autocomplete Base (13.8M Mac · 27.8 MB · score 1.795)",       "brain.pt",                            "base", 16, "dense"),
}

def get_tokenizer(tok_path):
    """Load and cache a BPE tokenizer (4k for 14M Mac, 8k for 97.6M Cloud, 16k for 286.7M Flagship)."""
    if tok_path not in TOKENIZERS:
        TOKENIZERS[tok_path] = tokenizer.load(tok_path)
    return TOKENIZERS[tok_path]

def use_tokenizer(tok_path, ctx_len=256):
    """Point tinygpt's active encode/decode helpers and context window at the selected brain."""
    merges, vocab = get_tokenizer(tok_path)
    tinygpt.merges, tinygpt.vocab, tinygpt.vocab_size = merges, vocab, len(vocab)
    tinygpt.block_size = ctx_len
    tinygpt.encode = lambda s, m=merges: tokenizer.encode(s, m)
    tinygpt.decode = lambda l, v=vocab: tokenizer.decode(l, v)

def default_brain_key():
    wanted = os.environ.get("HEDWIG_DEFAULT")  # e.g. the Space picks the fast 97.6M 4-bit brain
    if wanted in BRAIN_SPECS and os.path.exists(BRAIN_SPECS[wanted][1]):
        return wanted
    for k in ("c300_dpo_16", "c97_dpo_16", "c300_dpo_4", "c97_dpo_4", "dpo_16", "chat_16"):
        if os.path.exists(BRAIN_SPECS[k][1]):
            return k
    return "dpo_16"

def get_model(key):
    """Lazy-load and cache the requested brain (286.7M Flagship, 97.6M Cloud, 14M Mac, MoE, Baby, or quantized) on M4 GPU."""
    if key not in BRAIN_SPECS:
        key = default_brain_key()
    label, path, mode, bits, arch = BRAIN_SPECS[key]
    if not os.path.exists(path):
        key = default_brain_key()
        label, path, mode, bits, arch = BRAIN_SPECS[key]
    tok_path = (
        "cloud_brains_300m/tokenizer.json" if arch == "cloud300" else
        "cloud_brains_97m/tokenizer.json"  if arch == "cloud97"  else
        "tokenizer.json"
    )
    ctx_len = 512 if arch == "cloud300" else 256
    if key not in MODELS:
        use_tokenizer(tok_path, ctx_len)
        if arch == "cloud300":
            saved = (tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size)
            tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size = 1024, 16, 20, 512
            m = tinygpt.TinyGPT().to(tinygpt.device)
            tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size = saved
        elif arch == "cloud97":
            saved = (tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size)
            tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size = 768, 12, 12, 256
            m = tinygpt.TinyGPT().to(tinygpt.device)
            tinygpt.n_embd, tinygpt.n_head, tinygpt.n_layer, tinygpt.block_size = saved
        elif arch == "moe":
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
        MODELS[key] = (m, label, path, mode, os.path.getsize(path) / 1e6, tok_path, ctx_len)
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
.brand svg{width:38px;height:38px;animation:bob 5s ease-in-out infinite}
@keyframes bob{50%{transform:translateY(-3px)}}
.brand .lid{transform-origin:center;transform-box:fill-box;animation:blink 6s infinite}
@keyframes blink{0%,94%,100%{transform:scaleY(0)}96%{transform:scaleY(1)}}
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
  <div class="brand"><svg viewBox="-100 -100 200 200" aria-hidden="true"><path d="M-38 78 Q-10 66 26 70 Q54 73 74 91 L66 97 Q28 84 -8 86 Q-28 89 -42 95 Z" fill="#1b243c" stroke="#314169" stroke-width="1.4"/><path d="M-14 -50 C-2 -70 30 -70 42 -48 C48 -36 47 -22 46 -6 C48 24 34 56 -6 66 C-34 72 -52 46 -40 14 C-32 -8 -24 -34 -14 -50 Z" fill="#f7f4ec"/><path d="M8 -14 C26 4 12 36 -18 62 C-46 86 -72 98 -82 96 C-80 76 -64 42 -42 16 C-22 -8 -4 -22 8 -14 Z" transform="translate(24 -10)" fill="#ece6da" stroke="#cfc6b4" stroke-width="1.2"/><g transform="translate(24 -10)" fill="#4e341b" opacity=".82"><path d="M-6 12 l5 1 l-3 4 Z"/><path d="M-18 22 l5 1 l-3 4 Z"/><path d="M-28 32 l5 2 l-3 4 Z"/><path d="M-36 42 l6 2 l-4 4 Z"/><path d="M-44 52 l6 2 l-4 4 Z"/><path d="M-50 62 l5 2 l-3 4 Z"/></g><g transform="rotate(10 2 -40)"><ellipse cx="2" cy="-40" rx="7.6" ry="6.8" fill="#e59200" stroke="#241b11" stroke-width="1.4"/><circle cx="2" cy="-40" r="3.4" fill="#0c0905"/><circle cx=".5" cy="-41.8" r="1.4" fill="#fff"/><ellipse class="lid" cx="2" cy="-40" rx="8.2" ry="7.4" fill="#efe8d8"/></g><g transform="rotate(-10 24 -40)"><ellipse cx="24" cy="-40" rx="7.6" ry="6.8" fill="#e59200" stroke="#241b11" stroke-width="1.4"/><circle cx="24" cy="-40" r="3.4" fill="#0c0905"/><circle cx="22.5" cy="-41.8" r="1.4" fill="#fff"/><ellipse class="lid" cx="24" cy="-40" rx="8.2" ry="7.4" fill="#efe8d8"/></g><path d="M10 -34 Q13 -37 16 -34 L14 -24 Q13 -22 11 -24 Z" fill="#23252b"/></svg> Hedwig <span class="badge">286.7M Flagship + 97.6M + 14M · 🔬 Brain X-Ray</span></div>
  <div class="controls">
    <label>Brain:
      <select id="model">
        <option value="c300_dpo_16">🏆 286.7M Flagship Hedwig — Stories + Grammar (DPO 16-bit · 575 MB)</option>
        <option value="c300_dpo_4">🏆 286.7M Flagship Hedwig — Stories + Grammar (DPO 4-bit · 162 MB)</option>
        <option value="c300_chat_16">🏆 286.7M Flagship Hedwig — Stories + Grammar (SFT 16-bit · 575 MB)</option>
        <option value="c300_chat_4">🏆 286.7M Flagship Hedwig — Stories + Grammar (SFT 4-bit · 162 MB)</option>
        <option value="c97_dpo_16">🌟 97.6M Cloud Hedwig — Stories + Grammar (DPO 16-bit · 196 MB)</option>
        <option value="c97_dpo_4">🌟 97.6M Cloud Hedwig — Stories + Grammar (DPO 4-bit · 55 MB)</option>
        <option value="c97_chat_16">🌟 97.6M Cloud Hedwig — Stories + Grammar (SFT 16-bit · 196 MB)</option>
        <option value="c97_chat_4">🌟 97.6M Cloud Hedwig — Stories + Grammar (SFT 4-bit · 55 MB)</option>
        <option value="dpo_16">Stage 3 DPO Chat (13.8M Mac · 27.8 MB · score 1.297)</option>
        <option value="dpo_8">Stage 3 DPO 8-bit (13.8M Mac · 14.0 MB)</option>
        <option value="dpo_4">Stage 3 DPO 4-bit (13.8M Mac · 7.9 MB)</option>
        <option value="moe_16">Mixture-of-Experts (26.2M / 11.6M active · 8 Experts Top-2)</option>
        <option value="baby_16">🐣 Baby Hedwig Distilled (3.4M knobs · 6.8 MB)</option>
        <option value="chat_16">Stage 2 SFT Chat (13.8M Mac · 27.8 MB · score 1.438)</option>
        <option value="chat_8">8-bit Quantized Chat (13.8M Mac · 14.0 MB)</option>
        <option value="chat_4">4-bit Quantized Chat (13.8M Mac · 7.9 MB)</option>
        <option value="base_16">Stage 1 Autocomplete Base (13.8M Mac · 27.8 MB)</option>
      </select>
    </label>
    <label>Creativity: <span id="tval" style="font-family:var(--mono)">0.75</span>
      <input id="temp" type="range" min="0.3" max="1.1" step="0.05" value="0.75" style="width:90px;vertical-align:middle" oninput="tval.textContent=this.value">
    </label>
  </div>
</header>
<div class="chipbar">
  <span class="chip" onclick="useChip('Fix grammar: hey can we move the call tomorrow im stuck with the update and need 2 hrs')">✍️ Fix grammar (Teams)</span>
  <span class="chip" onclick="useChip('Improve the grammar of this text: I goes to the office yesterday and send the report to my manager.')">📝 Polish sentence</span>
  <span class="chip" onclick="useChip('Tell me a story where a little girl named Lily finds a shiny key in the garden.')">🔑 Lily &amp; the shiny key</span>
  <span class="chip" onclick="useChip('Write a short story about a brave little owl named Pip and a golden key.')">🦉 Brave owl Pip</span>
  <span class="chip" onclick="useChip('Write a story about a boy named Harry Potter.')">⚡ Harry Potter test</span>
</div>
<div id="chat">
  <div class="msg bot">Hello! I'm <b>Hedwig</b>, running 100% offline on your MacBook Air M4.<br>• Switch between the brains above (including <b>Stage 3 at 16-bit, 8-bit, and 4-bit</b>, <b>26.2M Mixture-of-Experts</b>, and <b>3.1M Distilled Baby Hedwig</b>).<br>• <b>🔬 Brain X-Ray:</b> After I write a story, <b>click any word</b> to inspect the Top-5 words I was debating between (and which 2 MoE Experts woke up for that word)!</div>
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
        # Only show brains that are actually on disk (the Space only downloads a few),
        # and pre-select the default one.
        default = default_brain_key()
        lines = []
        for line in HTML.split("\n"):
            if '<option value="' in line:
                key = line.split('<option value="')[1].split('"')[0]
                if key in BRAIN_SPECS and not os.path.exists(BRAIN_SPECS[key][1]):
                    continue
                if key == default:
                    line = line.replace(f'value="{key}"', f'value="{key}" selected', 1)
            lines.append(line)
        body = "\n".join(lines).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = req.get("prompt", "Write a short story about a little owl.")
        key = req.get("model", default_brain_key())
        temp = max(0.1, min(1.5, float(req.get("temperature", 0.75))))
        model, label, _, mode, mb, tok_path, ctx_len = get_model(key)
        use_tokenizer(tok_path, ctx_len)

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
    get_model(default_brain_key())
    url = f"http://{HOST}:{PORT}"
    print(f"🦉 Hedwig Mac Chat UI + 🔬 Brain X-Ray running at {url} (Ctrl+C to stop)", flush=True)
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    HTTPServer((HOST, PORT), Handler).serve_forever()
