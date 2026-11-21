// Proves the JavaScript Hedwig matches PyTorch: same pieces from the tokenizer,
// same scores for the first word, and the same 40 greedy (always-pick-the-top) pieces.
// Run:  node web/test_web.mjs 14m     (needs: .venv/bin/python export_web.py first)
import fs from "node:fs";
import { Tokenizer, loadWeights, Brain } from "./hedwig.js";

const name = process.argv[2] || "14m", dir = new URL(`./brains/${name}/`, import.meta.url);
const meta = JSON.parse(fs.readFileSync(new URL("model.json", dir)));
const check = JSON.parse(fs.readFileSync(new URL("check.json", dir)));
const tok = new Tokenizer(JSON.parse(fs.readFileSync(new URL("tokenizer.json", dir))));
const bin = fs.readFileSync(new URL("model.bin", dir));
const brain = new Brain(meta.config, loadWeights(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length)));

const ids = tok.encode(check.prefix);
console.log("tokenizer matches:", JSON.stringify(ids) === JSON.stringify(check.prompt_ids));
let logits;
for (const id of ids) logits = brain.step(id);
const diff = Math.max(...check.first_logits.map((v, i) => Math.abs(v - logits[i])));
console.log("biggest first-score difference:", diff.toExponential(2));
const got = [];
const t0 = performance.now();
for (let s = 0; s < check.greedy_ids.length; s++) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  got.push(best);
  logits = brain.step(best);
}
const secs = (performance.now() - t0) / 1000;
const same = got.filter((v, i) => v === check.greedy_ids[i]).length;
console.log(`greedy pieces matching PyTorch: ${same}/${got.length}  (${(got.length / secs).toFixed(1)} pieces/s)`);
console.log("JS story:", JSON.stringify(tok.decode(got)));
