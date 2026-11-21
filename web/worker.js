// Runs Hedwig in a background thread so the page never freezes while it thinks.
import { Tokenizer, loadWeights, Brain, replyStream } from "./hedwig.js";
let loaded = {};   // brain name -> { brain, tok }

async function load(name) {
  if (loaded[name]) return loaded[name];
  const base = `brains/${name}/`;
  const [meta, merges] = await Promise.all([fetch(base + "model.json").then(r => r.json()),
                                            fetch(base + "tokenizer.json").then(r => r.json())]);
  const res = await fetch(base + "model.bin");
  const total = +res.headers.get("Content-Length") || 0, reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {                                   // download with a progress bar
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    postMessage({ type: "progress", got, total });
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  loaded = { [name]: { brain: new Brain(meta.config, loadWeights(meta, buf.buffer)), tok: new Tokenizer(merges) } }; // keep one brain in memory
  return loaded[name];
}

onmessage = async ({ data }) => {
  try {
    postMessage({ type: "status", text: "Loading brain…" });
    const { brain, tok } = await load(data.brain);
    postMessage({ type: "status", text: "Writing…" });
    const t0 = performance.now(); let n = 0, last;
    for (const part of replyStream(brain, tok, data.prompt, { temperature: data.temperature })) {
      n++; last = part;
      postMessage({ type: "text", text: part.text, tps: n / ((performance.now() - t0) / 1000), tokens: n });
      if (n % 4 === 0) await new Promise(r => setTimeout(r, 0));   // let messages flow to the page
    }
    postMessage({ type: "done", text: last.text, xray: last.xray });
  } catch (e) { postMessage({ type: "error", text: String(e) }); }
};
