// Hedwig in plain JavaScript: the same tokenizer, brain and word-picking as tinygpt.py / sft.py,
// rewritten by hand so it runs inside a web browser with no server and no libraries.
// Each section names the Python function it copies.

// ---------------- Tokenizer (tokenizer.py) ----------------
// Same pre-split pattern as Python's PAT. Python's \w means "letter, digit or _" in any language,
// so we spell that out with \p{L}\p{N}_ (JavaScript's own \w only knows English letters).
const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\p{L}\p{N}_]+|\s+(?!\S)|\s+/gu;

export class Tokenizer {
  constructor(mergeList) {                     // mergeList = [[a, b, new], ...] from tokenizer.json
    this.rank = new Map();                     // "a,b" -> new piece number (lower = learned earlier)
    this.vocab = [];                           // piece number -> its bytes
    for (let i = 0; i < 256; i++) this.vocab[i] = new Uint8Array([i]);
    for (const [a, b, n] of mergeList) {
      this.rank.set(a * 65536 + b, n);
      const va = this.vocab[a], vb = this.vocab[b], v = new Uint8Array(va.length + vb.length);
      v.set(va); v.set(vb, va.length);
      this.vocab[n] = v;
    }
    this.cache = new Map();
    this.enc = new TextEncoder();
    this.dec = new TextDecoder("utf-8");       // replaces broken bytes with � like Python's errors="replace"
  }
  encodeChunk(bytes) {                         // _encode_chunk: keep gluing the earliest-learned pair
    let ids = Array.from(bytes);
    while (ids.length >= 2) {
      let best = Infinity, bestI = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.rank.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && r < best) { best = r; bestI = i; }
      }
      if (bestI < 0) break;
      const a = ids[bestI], b = ids[bestI + 1], out = [];
      for (let i = 0; i < ids.length; i++) {   // merge(): replace every copy of the pair
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { out.push(best); i++; }
        else out.push(ids[i]);
      }
      ids = out;
    }
    return ids;
  }
  encode(text) {
    const out = [];
    for (const m of text.matchAll(PAT)) {
      let ids = this.cache.get(m[0]);
      if (!ids) { ids = this.encodeChunk(this.enc.encode(m[0])); this.cache.set(m[0], ids); }
      out.push(...ids);
    }
    return out;
  }
  decode(ids) {
    let n = 0;
    for (const i of ids) n += this.vocab[i].length;
    const buf = new Uint8Array(n);
    let o = 0;
    for (const i of ids) { buf.set(this.vocab[i], o); o += this.vocab[i].length; }
    return this.dec.decode(buf);
  }
}

// ---------------- Loading the weights (export_web.py wrote them) ----------------
export function loadWeights(meta, buffer) {
  const W = {};
  for (const t of meta.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.kind === "f32") {
      W[t.name] = new Float32Array(buffer, t.data, size).slice();
    } else {                                   // 8-bit: whole number × that row's scale
      const [rows, cols] = t.shape;
      const scales = new Float32Array(buffer, t.scales, rows);
      const q = new Int8Array(buffer, t.data, size);
      const f = new Float32Array(size);
      for (let r = 0; r < rows; r++) {
        const s = scales[r], base = r * cols;
        for (let c = 0; c < cols; c++) f[base + c] = q[base + c] * s;
      }
      W[t.name] = f;
    }
  }
  return W;
}

// ---------------- The brain (TinyGPT in tinygpt.py) ----------------
// y = W·x + b, where W is stored row by row like PyTorch's nn.Linear ([out, in]).
function linear(out, x, W, b, nOut, nIn) {
  for (let o = 0; o < nOut; o++) {
    let s = b ? b[o] : 0, base = o * nIn;
    for (let i = 0; i < nIn; i++) s += W[base + i] * x[i];
    out[o] = s;
  }
}
function rmsnorm(out, x, w) {                  // nn.RMSNorm: divide by the average size, then scale
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + 1.1920928955078125e-7); // float32 eps, like PyTorch
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
}

export class Brain {
  constructor(config, W) {
    Object.assign(this, config);
    this.W = W;
    this.hd = this.n_embd / this.n_head;       // numbers per attention head (64)
    const half = this.hd / 2;
    // RoPE angles, same as the cos/sin buffers in CausalSelfAttention
    this.cos = new Float32Array(this.block_size * half);
    this.sin = new Float32Array(this.block_size * half);
    for (let p = 0; p < this.block_size; p++)
      for (let i = 0; i < half; i++) {
        const invFreq = 1 / Math.pow(10000, (2 * i) / this.hd);
        const a = Math.fround(p * Math.fround(invFreq));
        this.cos[p * half + i] = Math.cos(a); this.sin[p * half + i] = Math.sin(a);
      }
    const C = this.n_embd, H = this.hidden;
    this.x = new Float32Array(C); this.xn = new Float32Array(C);
    this.qkv = new Float32Array(3 * C); this.att = new Float32Array(C); this.tmp = new Float32Array(C);
    this.g = new Float32Array(H); this.u = new Float32Array(H);
    this.scores = new Float32Array(this.block_size);
    this.logits = new Float32Array(this.vocab_size);
    this.reset();
  }
  reset() {                                    // empty KV cache: forget the story so far
    const n = this.block_size * this.n_embd;
    this.kCache = Array.from({ length: this.n_layer }, () => new Float32Array(n));
    this.vCache = Array.from({ length: this.n_layer }, () => new Float32Array(n));
    this.pos = 0;
  }
  rope(v, off, p) {                            // apply_rope: rotate the two halves of each head
    const half = this.hd / 2;
    for (let i = 0; i < half; i++) {
      const c = this.cos[p * half + i], s = this.sin[p * half + i];
      const a = v[off + i], b = v[off + half + i];
      v[off + i] = a * c - b * s; v[off + half + i] = a * s + b * c;
    }
  }
  // Feed ONE piece in, get scores for every possible next piece out. The KV cache remembers
  // earlier pieces, so each new piece only costs one pass (like start_pos in tinygpt.py).
  step(token) {
    const { W, n_embd: C, n_head: nh, hd, hidden: H, x, xn, qkv, att, tmp, g, u, scores } = this;
    const p = this.pos;
    if (p >= this.block_size) throw new Error("story is longer than Hedwig's memory");
    x.set(W["tok.weight"].subarray(token * C, (token + 1) * C));
    const scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < this.n_layer; l++) {
      const pre = `blocks.${l}.`;
      // --- attention: which earlier pieces matter for this one? ---
      rmsnorm(xn, x, W[pre + "ln1.weight"]);
      linear(qkv, xn, W[pre + "attn.qkv.weight"], W[pre + "attn.qkv.bias"], 3 * C, C);
      for (let h = 0; h < nh; h++) { this.rope(qkv, h * hd, p); this.rope(qkv, C + h * hd, p); }
      const K = this.kCache[l], V = this.vCache[l];
      K.set(qkv.subarray(C, 2 * C), p * C); V.set(qkv.subarray(2 * C, 3 * C), p * C);
      for (let h = 0; h < nh; h++) {
        const qo = h * hd;
        let max = -Infinity;
        for (let t = 0; t <= p; t++) {
          let s = 0; const ko = t * C + qo;
          for (let i = 0; i < hd; i++) s += qkv[qo + i] * K[ko + i];
          s *= scale; scores[t] = s; if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t <= p; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
        for (let i = 0; i < hd; i++) att[qo + i] = 0;
        for (let t = 0; t <= p; t++) {
          const w = scores[t] / sum, vo = t * C + qo;
          for (let i = 0; i < hd; i++) att[qo + i] += w * V[vo + i];
        }
      }
      linear(tmp, att, W[pre + "attn.proj.weight"], W[pre + "attn.proj.bias"], C, C);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
      // --- SwiGLU: down( silu(gate·x) × up·x ) ---
      rmsnorm(xn, x, W[pre + "ln2.weight"]);
      linear(g, xn, W[pre + "mlp.gate.weight"], null, H, C);
      linear(u, xn, W[pre + "mlp.up.weight"], null, H, C);
      for (let i = 0; i < H; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
      linear(tmp, g, W[pre + "mlp.down.weight"], null, C, H);
      for (let i = 0; i < C; i++) x[i] += tmp[i];
    }
    rmsnorm(xn, x, W["ln.weight"]);
    linear(this.logits, xn, W["head.weight"], W["head.bias"], this.vocab_size, C);
    this.pos++;
    return this.logits;
  }
}

// ---------------- Picking the next piece (sample_next in tinygpt.py) ----------------
export function sampleNext(logitsIn, recentIds, { temperature = 0.75, topK = 40, repPenalty = 1.15, rand = Math.random } = {}) {
  const logits = Float32Array.from(logitsIn);
  if (recentIds.length && repPenalty > 1) {    // repetition penalty: make recently used pieces less likely
    for (const id of new Set(recentIds.slice(-64)))
      logits[id] = logits[id] > 0 ? logits[id] / repPenalty : logits[id] * repPenalty;
  }
  const t = Math.max(temperature, 1e-4);
  for (let i = 0; i < logits.length; i++) logits[i] /= t;
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
  const kth = logits[order[Math.min(topK, order.length) - 1]];
  const max = logits[order[0]];
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) if (logits[i] >= kth) { probs[i] = Math.exp(logits[i] - max); sum += probs[i]; }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  let r = rand(), pick = order[0];
  for (const i of order) { if (probs[i] === 0) break; r -= probs[i]; if (r <= 0) { pick = i; break; } }
  const top5 = order.slice(0, 5).map(i => [i, Math.round(probs[i] * 1000) / 10]);
  return { id: pick, top5 };
}

// ---------------- Chatting (reply_stream in sft.py) ----------------
const STOPS = ["\n\n", "Summary:", "Features:", "Random sentence:", "User:"];
export function* replyStream(brain, tok, prompt, opts = {}) {
  const maxNew = opts.maxNewTokens ?? 220;
  const prefixIds = tok.encode(`User: ${prompt.trim()}\nHedwig: `);
  brain.reset();
  let logits;
  for (const id of prefixIds.slice(-brain.block_size)) logits = brain.step(id);
  const out = [], xray = [];
  for (let s = 0; s < maxNew; s++) {
    const { id, top5 } = sampleNext(logits, out, opts);
    out.push(id);
    xray.push({ piece: tok.decode([id]), top5: top5.map(([i, p]) => [tok.decode([i]), p]) });
    const text = tok.decode(out);
    const stop = STOPS.find(st => text.includes(st));
    if (stop) { yield { text: text.split(stop)[0].trim(), xray, done: true }; return; }
    const full = brain.pos >= brain.block_size;   // memory is full: stop here (Python slides the window instead)
    yield { text: text.trim(), xray, done: full || s === maxNew - 1 };
    if (full) return;
    logits = brain.step(id);
  }
}
