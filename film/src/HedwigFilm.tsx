// Redesigned 60-second Hedwig launch film. Same voiceover and timing as
// before, but every scene is now a place with depth, light and a character
// (Pip the owl) instead of floating cards.
import React from "react";
import { AbsoluteFill, Audio, interpolate, random, staticFile, useCurrentFrame } from "remotion";
import { AUDIO_PEAK, AUDIO_RMS } from "./audioData";
import { CameraRig, CinemaPost } from "./cinemaPost";
import { FilmSubtitles } from "./cinematic";
import { Bokeh, C, deepText, Embers, F, Fog, GodRays, glass, k, Owl, RevealText, Scene, Sky } from "./filmKit";

const center: React.CSSProperties = { alignItems: "center", justifyContent: "center" };
const fmt = (n: number) => Math.round(n).toLocaleString("en-US");

// =========================================================================
// 1 · Cold open (0–130): stars, a tilt down, two golden eyes open in the dark
// =========================================================================
const S1: React.FC<{ t: number }> = ({ t }) => {
  const tilt = k(t, 0, 110, -420, 0);
  const swap = k(t, 62, 74);
  const slam = k(t, 70, 82, 1.6, 1);
  const eyes = k(t, 92, 104);
  return (
    <AbsoluteFill>
      <Sky t={t} panY={-tilt} top="#010208" bottom="#0d1230" />
      <Fog t={t} y={72} />
      <AbsoluteFill style={{ ...center, opacity: eyes * 0.9 }}>
        <div style={{ transform: `translateY(-70px) scale(${2.6 + eyes * 0.2})`, filter: `blur(${(1 - eyes) * 6}px)` }}>
          <Owl size={360} blink={1 - eyes} glow={1.4} />
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ background: `radial-gradient(circle at 50% 42%, transparent 0%, rgba(1,2,8,${0.95 - eyes * 0.35}) 38%)` }} />
      <AbsoluteFill style={{ ...center, perspective: 1200 }}>
        <div style={{ position: "absolute", opacity: 1 - swap, transform: `translateZ(${-swap * 600}px)`, fontFamily: F.serif, fontSize: 150, color: C.ivory, textShadow: deepText(C.ivory, "#3a3f55", "rgba(160,180,255,0.35)") }}>
          <RevealText text="Everyone rents AI." t={t} at={14} />
        </div>
        <div style={{ position: "absolute", opacity: swap, transform: `scale(${slam})`, fontFamily: F.serif, fontStyle: "italic", fontSize: 210, color: C.goldHot, textShadow: deepText(C.gold, "#6b4300", "rgba(255,190,80,0.7)", 18), top: 690 - 105 }}>
          We built one.
        </div>
        {/* shockwave ring on the slam */}
        <div style={{ position: "absolute", top: 690, width: 40, height: 40, borderRadius: "50%", border: `3px solid ${C.goldHot}`, opacity: t > 70 ? 1 - k(t, 70, 100) : 0, transform: `scale(${k(t, 70, 100, 1, 60)})`, boxShadow: `0 0 40px ${C.gold}` }} />
      </AbsoluteFill>
      <Bokeh t={t} opacity={0.5} />
    </AbsoluteFill>
  );
};

// =========================================================================
// 2 · Zero knobs (130–238): a field of 3D dials lights up from the centre
// =========================================================================
const S2: React.FC<{ t: number }> = ({ t }) => {
  const cols = 15, rows = 9;
  const wave = k(t, 18, 96, 0, 1, (u) => u) * 12;
  const count = interpolate(k(t, 14, 98, 0, 1), [0, 1], [0, 286_663_680]);
  return (
    <AbsoluteFill style={{ background: "radial-gradient(ellipse at 50% 80%, #1a1440 0%, #04050c 70%)" }}>
      <AbsoluteFill style={{ perspective: 900, perspectiveOrigin: "50% 20%" }}>
        <div style={{ position: "absolute", left: "50%", top: "58%", transformStyle: "preserve-3d", transform: `translate(-50%,-50%) rotateX(64deg) rotateZ(${t * 0.12}deg) translateZ(${-120 + t}px)` }}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 110px)`, gap: 26 }}>
            {Array.from({ length: cols * rows }, (_, i) => {
              const cx = (i % cols) - (cols - 1) / 2, cy = Math.floor(i / cols) - (rows - 1) / 2;
              const on = k(wave - Math.hypot(cx, cy), 0, 1.2);
              const ang = -135 + on * (90 + random(`kn${i}`) * 180);
              return (
                <div key={i} style={{ width: 110, height: 110, borderRadius: "50%", background: `radial-gradient(circle at 35% 30%, #3a3f60, #10121f)`, border: `2px solid ${on > 0.5 ? C.gold : "#2a2f48"}`, boxShadow: on > 0.05 ? `0 0 ${30 * on}px ${C.gold}, inset 0 0 20px rgba(255,190,80,${on * 0.5})` : "inset 0 -6px 12px rgba(0,0,0,0.6)", position: "relative" }}>
                  <div style={{ position: "absolute", left: 52, top: 10, width: 6, height: 46, borderRadius: 3, background: on > 0.3 ? C.goldHot : "#555b78", transformOrigin: "3px 45px", transform: `rotate(${ang}deg)` }} />
                </div>
              );
            })}
          </div>
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "linear-gradient(180deg, rgba(4,5,12,0.95) 0%, rgba(4,5,12,0.2) 45%, rgba(4,5,12,0) 100%)" }} />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 150 }}>
        <div style={{ fontFamily: F.serif, fontSize: 170, color: C.ivory, textShadow: deepText(C.ivory, "#2f3450", "rgba(150,160,255,0.4)") }}>
          <RevealText text="From " t={t} at={4} />
          <RevealText text="zero" t={t} at={12} style={{ fontStyle: "italic", color: C.goldHot }} />
          <RevealText text=" knobs." t={t} at={22} />
        </div>
        <div style={{ marginTop: 18, fontFamily: F.mono, fontSize: 54, letterSpacing: "0.04em", color: C.goldHot, textShadow: `0 0 30px ${C.gold}`, opacity: k(t, 10, 20) }}>
          {fmt(count)}
        </div>
        <div style={{ fontFamily: F.sans, fontWeight: 600, fontSize: 22, letterSpacing: "0.3em", color: C.muted, opacity: k(t, 20, 30) }}>
          KNOBS TUNED BY HAND-WRITTEN CODE
        </div>
      </AbsoluteFill>
      <Bokeh t={t} />
    </AbsoluteFill>
  );
};

// =========================================================================
// 3 · Pure PyTorch (238–372): flying through a tunnel of Hedwig's own code
// =========================================================================
const CODE = [
  "q, k = apply_rope(q, k, cos, sin)",
  "y = F.scaled_dot_product_attention(q, k, v, is_causal=True)",
  "x = x + self.attn(self.norm1(x))",
  "x = x + self.ffn(self.norm2(x))",
  "return self.w2(F.silu(self.w1(x)) * self.w3(x))",
  "x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)",
  "loss = F.cross_entropy(logits.view(-1, V), y.view(-1))",
  "loss.backward()",
  "opt.step(); opt.zero_grad(set_to_none=True)",
  "with torch.autocast(device, dtype=torch.bfloat16):",
  "torch.save({'model': model.state_dict(), 'step': step}, path)",
  "logits = logits[:, -1, :] / temperature",
  "next_id = torch.multinomial(F.softmax(logits, -1), 1)",
];
const PIPELINE_BADGES = ["BPE", "RoPE + SwiGLU", "SFT", "DPO", "4-bit"];
const S3: React.FC<{ t: number }> = ({ t }) => {
  const n = 46;
  const a = k(t, 6, 40), b = k(t, 56, 90), strip = k(t, 44, 72);
  return (
    <AbsoluteFill style={{ background: "radial-gradient(circle at 50% 50%, #0c1a2a 0%, #020408 65%)" }}>
      <AbsoluteFill style={{ perspective: 700 }}>
        {Array.from({ length: n }, (_, i) => {
          const ang = (i / n) * Math.PI * 2 * 3.3;
          const z = ((random(`cz${i}`) * 3600 + t * 34) % 3600) - 3300;
          const r = 520 + random(`cr${i}`) * 260;
          const near = k(z, -3300, -200) * (1 - k(z, -200, 250));
          return (
            <div key={i} style={{ position: "absolute", left: "50%", top: "50%", whiteSpace: "nowrap", fontFamily: F.mono, fontSize: 30, color: i % 5 === 0 ? C.goldHot : C.cyan, opacity: near, textShadow: `0 0 18px ${i % 5 === 0 ? C.gold : C.cyan}`, transform: `translate3d(${Math.cos(ang) * r}px, ${Math.sin(ang) * r * 0.62}px, ${z}px) translate(-50%,-50%) rotateY(${Math.cos(ang) * -50}deg)`, filter: z > -400 ? `blur(${k(z, -400, 200, 0, 6)}px)` : undefined }}>
              {CODE[i % CODE.length]}
            </div>
          );
        })}
      </AbsoluteFill>
      <AbsoluteFill style={{ background: "radial-gradient(circle at 50% 50%, rgba(2,4,8,0.85) 0%, rgba(2,4,8,0) 32%)" }} />
      <AbsoluteFill style={{ ...center, flexDirection: "column" }}>
        <div style={{ fontFamily: F.serif, fontSize: 150, color: C.ivory, opacity: a * (1 - b * 0.55), transform: `translateY(${-b * 40}px) scale(${1 - b * 0.25})`, textShadow: deepText(C.ivory, "#203040", "rgba(94,231,255,0.35)") }}>
          No shortcuts.
        </div>
        <div style={{ fontFamily: F.sans, fontWeight: 800, fontSize: 120, letterSpacing: "-0.03em", color: C.cyan, opacity: b, transform: `scale(${k(t, 56, 74, 1.4, 1)})`, textShadow: deepText(C.cyan, "#06485a", "rgba(94,231,255,0.7)", 12) }}>
          Pure PyTorch.
        </div>
        <div style={{ marginTop: 34, display: "flex", alignItems: "center", gap: 12, opacity: strip }}>
          {PIPELINE_BADGES.map((label, idx) => {
            const pulse = k(t, 44 + idx * 10, 56 + idx * 10);
            const active = pulse > 0.4;
            return (
              <React.Fragment key={label}>
                <div style={{ padding: "10px 20px", borderRadius: 999, fontFamily: F.mono, fontSize: 21, fontWeight: 600, color: active ? "#04121a" : C.cyan, background: active ? `linear-gradient(90deg, ${C.gold}, ${C.goldHot})` : "rgba(2, 8, 18, 0.78)", border: `1px solid ${active ? C.goldHot : "rgba(94,231,255,0.45)"}`, boxShadow: active ? `0 0 28px rgba(245,184,65,0.6)` : "0 4px 18px rgba(0,0,0,0.6)", transform: `scale(${0.92 + pulse * 0.1})` }}>
                  {label}
                </div>
                {idx < PIPELINE_BADGES.length - 1 && (
                  <span style={{ fontFamily: F.mono, fontSize: 22, color: C.ivory, opacity: 0.7 }}>→</span>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// =========================================================================
// 4 · A billion characters (372–556): a galaxy of story letters falls into a
//     glowing core, then the learning curve is drawn as a light on the horizon
// =========================================================================
const STORY = "Once upon a time there was a little owl named Pip who loved to read stories under the moon with her friend Lily ";
const S4: React.FC<{ t: number }> = ({ t }) => {
  const core = k(t, 0, 120, 0.4, 1.3);
  const curve = k(t, 96, 160);
  const pts = Array.from({ length: 60 }, (_, i) => {
    const x = i / 59;
    const loss = 1.795 + (8.47 - 1.795) * Math.exp(-x * 6) + Math.sin(i * 1.7) * 0.05 * (1 - x);
    return [120 + x * 1680, 860 - ((loss - 1.5) / 7.2) * 330] as const;
  });
  const shown = Math.max(2, Math.floor(curve * pts.length));
  const d = pts.slice(0, shown).map((p, i) => `${i ? "L" : "M"}${p[0]} ${p[1]}`).join(" ");
  const head = pts[shown - 1];
  return (
    <AbsoluteFill style={{ background: "radial-gradient(circle at 50% 42%, #1c1240 0%, #03030a 70%)" }}>
      <Sky t={t} top="rgba(0,0,0,0)" bottom="rgba(0,0,0,0)" />
      <AbsoluteFill style={{ perspective: 1000 }}>
        <div style={{ position: "absolute", left: "50%", top: "42%", transformStyle: "preserve-3d", transform: "rotateX(62deg)" }}>
          {Array.from({ length: 220 }, (_, i) => {
            const base = random(`gr${i}`);
            const p = (base + t * 0.004) % 1;
            const r = 900 * (1 - p);
            const ang = base * 40 + t * 0.03 + p * 7;
            return (
              <div key={i} style={{ position: "absolute", fontFamily: F.serif, fontSize: 34 + base * 20, color: p > 0.85 ? C.goldHot : C.ivory, opacity: Math.min(1, p * 4) * (1 - k(p, 0.85, 1)), transform: `translate(${Math.cos(ang) * r}px, ${Math.sin(ang) * r}px) rotateX(-62deg)`, textShadow: `0 0 12px ${p > 0.7 ? C.gold : "rgba(180,190,255,0.6)"}` }}>
                {STORY[i % STORY.length]}
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
      <div style={{ position: "absolute", left: 960 - 220 * core, top: 453 - 220 * core, width: 440 * core, height: 440 * core, borderRadius: "50%", background: `radial-gradient(circle, #fff 0%, ${C.goldHot} 12%, rgba(245,184,65,0.5) 30%, rgba(129,140,248,0.15) 55%, transparent 70%)`, mixBlendMode: "screen" }} />
      <GodRays t={t} x={50} y={42} opacity={0.6} />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 120 }}>
        <div style={{ fontFamily: F.mono, fontSize: 92, color: C.ivory, textShadow: deepText(C.ivory, "#352a60", "rgba(200,170,255,0.5)", 10) }}>
          {fmt(interpolate(k(t, 6, 90), [0, 1], [0, 1_000_000_000]))}
        </div>
        <div style={{ fontFamily: F.serif, fontStyle: "italic", fontSize: 58, color: C.goldHot, opacity: k(t, 30, 46) }}>characters, learned from scratch</div>
      </AbsoluteFill>
      <svg width={1920} height={1080} style={{ position: "absolute", opacity: k(t, 92, 102) }}>
        <defs>
          <linearGradient id="lc" x1="0" x2="1">
            <stop offset="0" stopColor={C.rose} />
            <stop offset="1" stopColor={C.emerald} />
          </linearGradient>
        </defs>
        <path d={d} stroke="url(#lc)" strokeWidth={14} fill="none" opacity={0.35} style={{ filter: "blur(10px)" }} />
        <path d={d} stroke="url(#lc)" strokeWidth={4} fill="none" />
        <circle cx={head[0]} cy={head[1]} r={10} fill="#fff" style={{ filter: `drop-shadow(0 0 16px ${C.emerald})` }} />
        <text x={130} y={520} fill={C.rose} fontFamily="JetBrains Mono" fontSize={30}>loss 8.47</text>
        <text x={1790} y={840} fill={C.emerald} fontFamily="JetBrains Mono" fontSize={30} textAnchor="end" opacity={k(t, 150, 162)}>1.795</text>
      </svg>
      <Bokeh t={t} opacity={0.5} />
    </AbsoluteFill>
  );
};

// =========================================================================
// 5 · Aligned (556–762): two story pages; the bad one burns, the good one wins
// =========================================================================
const Page: React.FC<{ title: string; body: string; accent: string; style?: React.CSSProperties }> = ({ title, body, accent, style }) => (
  <div style={{ position: "absolute", width: 560, height: 700, padding: 54, borderRadius: 10, background: "linear-gradient(170deg, #fbf6e9, #e8dfc8)", boxShadow: `0 60px 140px rgba(0,0,0,0.65), 0 0 0 2px ${accent}`, fontFamily: F.serif, color: "#2a2418", ...style }}>
    <div style={{ fontFamily: F.sans, fontWeight: 800, fontSize: 20, letterSpacing: "0.25em", color: accent }}>{title}</div>
    <div style={{ marginTop: 28, fontSize: 44, lineHeight: 1.25 }}>{body}</div>
  </div>
);
const S5: React.FC<{ t: number }> = ({ t }) => {
  const enter = k(t, 0, 40);
  const burn = k(t, 70, 120, 0, 1, (u) => u);
  const win = k(t, 100, 140);
  const big = k(t, 120, 150);
  return (
    <AbsoluteFill style={{ background: "radial-gradient(ellipse at 50% 60%, #2a1a0c 0%, #070406 70%)" }}>
      <GodRays t={t} x={50 + win * 12} y={30} opacity={win} />
      <AbsoluteFill style={{ perspective: 1600 }}>
        <div style={{ position: "absolute", left: "50%", top: "50%", transformStyle: "preserve-3d", transform: `translate(-50%,-50%) rotateY(${(1 - enter) * 30 - win * 6}deg)` }}>
          <div>
            <Page title="REJECTED" accent={C.rose} body="the the key was key. Lily tree tree the went went and." style={{ left: -620 - enter * 40, top: -350, transform: `rotateY(18deg) translateZ(${-burn * 200}px)`, filter: `sepia(${burn}) brightness(${1 - burn * 0.4})`, WebkitMaskImage: `radial-gradient(circle at 50% 100%, transparent ${burn * 130}%, #000 ${burn * 130 + 6}%)`, maskImage: `radial-gradient(circle at 50% 100%, transparent ${burn * 130}%, #000 ${burn * 130 + 6}%)` }} />
          </div>
          <Page title="CHOSEN" accent={C.gold} body="Pip found the shiny key under the old oak tree, and she shared it with Lily." style={{ left: 60 + enter * 40 - win * 380, top: -350, transform: `rotateY(${-18 + win * 18}deg) translateZ(${win * 160}px) scale(${1 - big * 0.18})`, boxShadow: `0 60px 140px rgba(0,0,0,0.65), 0 0 0 3px ${C.gold}, 0 0 ${120 * win}px rgba(255,190,80,0.6)` }} />
        </div>
      </AbsoluteFill>
      <Embers t={t} n={90} x={30} w={34} opacity={burn * (1 - k(t, 150, 200))} />
      <AbsoluteFill style={{ alignItems: "flex-end", justifyContent: "center", paddingRight: 140, opacity: big }}>
        <div style={{ textAlign: "right", transform: `translateX(${(1 - big) * 80}px)` }}>
          <div style={{ fontFamily: F.serif, fontSize: 260, lineHeight: 0.9, color: C.goldHot, textShadow: deepText(C.gold, "#6b4300", "rgba(255,190,80,0.7)", 18) }}>
            {(98.3 * k(t, 120, 175)).toFixed(1)}%
          </div>
          <div style={{ fontFamily: F.sans, fontWeight: 600, fontSize: 28, letterSpacing: "0.28em", color: C.ivory }}>PREFERENCE WIN RATE</div>
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 70, opacity: k(t, 4, 18) * (1 - big) }}>
        <div style={{ fontFamily: F.serif, fontSize: 96, color: C.ivory, textShadow: deepText(C.ivory, "#3d2a18", "rgba(255,190,80,0.3)") }}>
          Not just fluent. <span style={{ fontStyle: "italic", color: C.goldHot }}>Aligned.</span>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// =========================================================================
// 6 · Three generations (762–978): monoliths rise out of fog, low angle
// =========================================================================
const Monolith: React.FC<{ h: number; w: number; color: string; rise: number; label: string; sub: string; x: number }> = ({ h, w, color, rise, label, sub, x }) => {
  const H = h * rise;
  const face = (extra: React.CSSProperties): React.CSSProperties => ({ position: "absolute", bottom: 0, ...extra });
  return (
    <div style={{ position: "absolute", left: x, bottom: 0, width: w, transformStyle: "preserve-3d" }}>
      <div style={face({ width: w, height: H, background: `linear-gradient(180deg, ${color} 0%, #111628 85%)`, boxShadow: `0 0 80px ${color}66`, borderTop: `3px solid #fff` })} />
      <div style={face({ width: 120, height: H, left: w, background: `linear-gradient(180deg, ${color}aa, #070a16 80%)`, transformOrigin: "left bottom", transform: "rotateY(-90deg)" })} />
      <div style={{ position: "absolute", bottom: H + 40, width: w, textAlign: "center", opacity: k(rise, 0.7, 1) }}>
        <div style={{ fontFamily: F.serif, fontSize: 110, color: "#fff", textShadow: deepText("#fff", "#2a2f48", color, 10) }}>{label}</div>
        <div style={{ fontFamily: F.mono, fontSize: 24, color }}>{sub}</div>
      </div>
    </div>
  );
};
const S6: React.FC<{ t: number }> = ({ t }) => {
  const r1 = k(t, 4, 40), r2 = k(t, 52, 92), r3 = k(t, 104, 150);
  const total = k(t, 160, 185);
  const pan = k(t, 0, 216, 220, -260, (u) => u);
  const gauge = k(t, 20, 60) * (1 - total * 0.85);
  const horizon = k(t, 104, 140) * (1 - total * 0.8);
  const gpuFill = k(t, 24, 90);
  return (
    <AbsoluteFill>
      <Sky t={t} panX={pan * 0.3} top="#03040c" bottom="#22264a" />
      <AbsoluteFill style={{ perspective: 1400, perspectiveOrigin: "50% 90%" }}>
        <div style={{ position: "absolute", left: 0, right: 0, bottom: 150, height: 900, transformStyle: "preserve-3d", transform: `translateX(${pan}px) rotateY(-14deg) translateZ(-200px)` }}>
          <Monolith x={280} w={260} h={230} color={C.indigo} rise={r1} label="14M" sub="Mac M4 · $0.00" />
          <Monolith x={760} w={300} h={420} color={C.cyan} rise={r2} label="97.6M" sub="A40 · $0.65" />
          <Monolith x={1280} w={340} h={640} color={C.gold} rise={r3} label="287M" sub="RTX 4090 · $1.70" />
        </div>
      </AbsoluteFill>
      {/* dashed frontier horizon line safely below top letterbox */}
      <div style={{ position: "absolute", left: 140, right: 140, top: 108, opacity: horizon }}>
        <div style={{ borderTop: "2px dashed rgba(251,113,133,0.72)" }} />
        <div style={{ marginTop: 10, display: "flex", justifyContent: "flex-start", gap: 18, fontFamily: F.mono, fontSize: 20, letterSpacing: "0.06em", color: C.rose, textShadow: "0 2px 14px rgba(0,0,0,0.95)" }}>
          <span>▲ GPT / CLAUDE FRONTIER SCALE:</span>
          <span style={{ color: C.ivory }}>100B+ KNOBS · 10,000+ GPUS · $100M+</span>
        </div>
      </div>
      {/* Mac M4 vs Cloud GPU 26x speed gauge above left 14M monolith */}
      <div style={{ position: "absolute", left: 140, top: 184, width: 640, padding: "16px 22px", ...glass(C.cyan), opacity: gauge }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 18, color: C.cyan, letterSpacing: "0.06em" }}>
          <span>MAC M4: 59m → CLOUD GPU: 135s</span>
          <span style={{ color: C.goldHot, fontWeight: 700 }}>26× FASTER</span>
        </div>
        <div style={{ marginTop: 10, height: 8, borderRadius: 4, background: "rgba(255,255,255,0.08)", overflow: "hidden" }}>
          <div style={{ width: `${gpuFill * 100}%`, height: "100%", borderRadius: 4, background: `linear-gradient(90deg, ${C.indigo}, ${C.cyan}, ${C.goldHot})`, boxShadow: `0 0 18px ${C.cyan}` }} />
        </div>
      </div>
      {/* glossy floor reflection */}
      <AbsoluteFill style={{ top: "auto", height: 170, background: "linear-gradient(180deg, rgba(120,130,220,0.18), rgba(3,4,12,0.95))" }} />
      <Fog t={t * 2} y={72} color="170,180,255" />
      <AbsoluteFill style={{ background: `rgba(2,3,8,${total * 0.6})` }} />
      <AbsoluteFill style={{ ...center, flexDirection: "column", opacity: total }}>
        <div style={{ fontFamily: F.serif, fontSize: 220, color: C.goldHot, transform: `scale(${1.3 - total * 0.3})`, textShadow: deepText(C.gold, "#6b4300", "rgba(255,190,80,0.7)", 18) }}>Under $3.</div>
        <div style={{ fontFamily: F.mono, fontSize: 34, color: C.ivory }}>$2.45 of rented GPU, total</div>
      </AbsoluteFill>
      <Bokeh t={t} />
    </AbsoluteFill>
  );
};

// =========================================================================
// 7 · Brain X-Ray (978–1210): the story is written word by word while a
//     hologram shows which next words the brain is weighing
// =========================================================================
const STEPS: { word: string; top: [string, number][] }[] = [
  { word: "little", top: [["little", 61], ["girl", 12], ["boy", 9], ["big", 7], ["small", 5]] },
  { word: "owl", top: [["owl", 48], ["girl", 22], ["bird", 11], ["boy", 9], ["dog", 4]] },
  { word: "named", top: [["named", 71], ["who", 14], ["that", 6], ["called", 4], ["with", 2]] },
  { word: "Pip.", top: [["Pip", 38], ["Lily", 21], ["Tim", 12], ["Max", 9], ["Sue", 6]] },
];
const S7: React.FC<{ t: number }> = ({ t }) => {
  const step = Math.min(STEPS.length - 1, Math.max(0, Math.floor((t - 40) / 42)));
  const local = (t - 40) - step * 42;
  const words = STEPS.slice(0, t < 40 ? 0 : step + (local > 30 ? 1 : 0)).map((s) => s.word);
  const scan = ((t * 9) % 1300) - 100;
  const panel = k(t, 10, 40);
  return (
    <AbsoluteFill style={{ background: "radial-gradient(ellipse at 40% 50%, #071a24 0%, #020407 70%)" }}>
      <AbsoluteFill style={{ backgroundImage: "linear-gradient(rgba(94,231,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(94,231,255,0.06) 1px, transparent 1px)", backgroundSize: "60px 60px", transform: `perspective(800px) rotateX(60deg) translateY(${t % 60}px) scale(2)`, transformOrigin: "50% 100%", opacity: 0.6 }} />
      <div style={{ position: "absolute", left: 130, top: 300, width: 900, fontFamily: F.serif, fontSize: 84, lineHeight: 1.15, color: C.ivory, textShadow: "0 0 30px rgba(94,231,255,0.35)" }}>
        Once upon a time, there was a{" "}
        {words.map((w, i) => (
          <span key={i} style={{ color: i === words.length - 1 ? C.goldHot : C.ivory, textShadow: i === words.length - 1 ? `0 0 30px ${C.gold}` : undefined }}>{w} </span>
        ))}
        <span style={{ opacity: Math.floor(t / 8) % 2 ? 1 : 0, color: C.cyan }}>▍</span>
      </div>
      <AbsoluteFill style={{ perspective: 1400 }}>
        <div style={{ position: "absolute", right: 120, top: 190, width: 640, padding: 44, ...glass(C.cyan), opacity: panel, transform: `rotateY(${-22 + panel * 6}deg) translateX(${(1 - panel) * 200}px)` }}>
          <div style={{ fontFamily: F.mono, fontSize: 20, letterSpacing: "0.2em", color: C.cyan }}>🔬 BRAIN X-RAY · NEXT WORD</div>
          {STEPS[step].top.map(([w, p], i) => {
            const grow = k(local, 4 + i * 3, 20 + i * 3);
            return (
              <div key={`${step}${i}`} style={{ marginTop: 26 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 30, color: i === 0 ? C.goldHot : C.ivory }}>
                  <span>{w}</span>
                  <span>{(p * grow).toFixed(0)}%</span>
                </div>
                <div style={{ marginTop: 8, height: 12, borderRadius: 6, background: "rgba(255,255,255,0.06)" }}>
                  <div style={{ width: `${p * grow}%`, height: "100%", borderRadius: 6, background: i === 0 ? `linear-gradient(90deg, ${C.gold}, ${C.goldHot})` : C.cyan, boxShadow: `0 0 20px ${i === 0 ? C.gold : C.cyan}` }} />
                </div>
              </div>
            );
          })}
          <AbsoluteFill style={{ borderRadius: 22, background: "repeating-linear-gradient(0deg, rgba(94,231,255,0.05) 0 2px, transparent 2px 5px)" }} />
        </div>
      </AbsoluteFill>
      <div style={{ position: "absolute", left: 0, right: 0, top: scan, height: 3, background: C.cyan, boxShadow: `0 0 30px 8px rgba(94,231,255,0.5)`, opacity: 0.55 }} />
      <div style={{ position: "absolute", left: 130, top: 170, fontFamily: F.serif, fontStyle: "italic", fontSize: 64, color: C.cyan, opacity: k(t, 0, 16), textShadow: `0 0 30px ${C.cyan}` }}>See inside the mind.</div>
      <div style={{ position: "absolute", right: 200, bottom: 250, opacity: k(t, 30, 50) }}>
        <Owl size={150} look={Math.sin(t / 20)} blink={(t % 70) > 66 ? 1 : 0} />
      </div>
    </AbsoluteFill>
  );
};

// =========================================================================
// 8 · Crushed to 4-bit (1210–1412): a hydraulic press squeezes the brain,
//     then it rockets away at 832 words-pieces a second
// =========================================================================
const S8: React.FC<{ t: number }> = ({ t }) => {
  const press = k(t, 40, 62, 0, 1, (u) => u * u);
  const impact = t >= 62 && t < 80 ? 1 - (t - 62) / 18 : 0;
  const shake = impact * 26;
  const sx = (random(`shx${t}`) - 0.5) * shake, sy = (random(`shy${t}`) - 0.5) * shake;
  const size = 420 - press * 270;
  const warp = k(t, 104, 130);
  return (
    <AbsoluteFill style={{ background: "radial-gradient(circle at 50% 55%, #1b1030 0%, #030208 70%)", transform: `translate(${sx}px, ${sy}px)` }}>
      {/* hyperspace streaks */}
      <AbsoluteFill style={{ opacity: warp }}>
        {Array.from({ length: 120 }, (_, i) => {
          const a = random(`wa${i}`) * Math.PI * 2;
          const p = (random(`wp${i}`) + t * 0.03) % 1;
          const len = 60 + p * 500;
          return <div key={i} style={{ position: "absolute", left: 960 + Math.cos(a) * p * 1100, top: 540 + Math.sin(a) * p * 700, width: len, height: 2 + p * 2, background: `linear-gradient(90deg, transparent, ${i % 3 ? C.indigo : C.goldHot})`, transform: `rotate(${(a * 180) / Math.PI}deg)`, transformOrigin: "0 50%", opacity: p }} />;
        })}
      </AbsoluteFill>
      {/* press plates */}
      {[-1, 1].map((s) => (
        <div key={s} style={{ position: "absolute", left: 560, width: 800, height: 520, top: s < 0 ? -520 + k(t, 20, 40) * 140 + press * (300 - size / 2 + 20) : 1080 - k(t, 20, 40) * 140 - press * (300 - size / 2 + 20), background: "linear-gradient(180deg, #3a3f55, #171a26)", border: "2px solid #565c78", boxShadow: "0 0 60px rgba(0,0,0,0.8)", opacity: 1 - warp }}>
          <div style={{ position: "absolute", [s < 0 ? "bottom" : "top"]: 0, left: 0, right: 0, height: 10, background: `repeating-linear-gradient(90deg, ${C.gold} 0 30px, #111 30px 60px)` }} />
        </div>
      ))}
      {/* the brain cube */}
      <AbsoluteFill style={{ ...center, perspective: 1200 }}>
        <div style={{ width: size, height: size, transformStyle: "preserve-3d", transform: `rotateX(-22deg) rotateY(${t * 1.2}deg) scale(${1 - warp * 0.4}) translateZ(${warp * -800}px)` }}>
          {[0, 90, 180, 270].map((ry) => (
            <div key={ry} style={{ position: "absolute", inset: 0, transform: `rotateY(${ry}deg) translateZ(${size / 2}px)`, background: press > 0.99 ? `linear-gradient(135deg, ${C.goldHot}, ${C.gold} 50%, #7a4a00)` : `linear-gradient(135deg, rgba(129,140,248,0.85), rgba(40,30,120,0.85))`, border: `2px solid ${press > 0.99 ? "#fff3c4" : C.indigo}`, boxShadow: `0 0 60px ${press > 0.99 ? C.gold : C.indigo}`, display: "grid", gridTemplateColumns: `repeat(${press > 0.99 ? 4 : 8}, 1fr)` }}>
              {Array.from({ length: press > 0.99 ? 16 : 64 }, (_, j) => <div key={j} style={{ border: "1px solid rgba(255,255,255,0.12)" }} />)}
            </div>
          ))}
          {[90, -90].map((rx) => <div key={rx} style={{ position: "absolute", inset: 0, transform: `rotateX(${rx}deg) translateZ(${size / 2}px)`, background: press > 0.99 ? C.gold : "rgba(80,70,200,0.8)" }} />)}
        </div>
      </AbsoluteFill>
      {/* impact shockwave + sparks */}
      {t >= 62 && t < 100 && (
        <>
          <div style={{ position: "absolute", left: 960, top: 540, width: 10, height: 10, borderRadius: "50%", border: `4px solid ${C.goldHot}`, transform: `translate(-50%,-50%) scale(${k(t, 62, 100, 1, 180)})`, opacity: 1 - k(t, 62, 100) }} />
          <Embers t={(t - 62) * 4} n={70} x={50} w={30} opacity={1 - k(t, 62, 100)} />
        </>
      )}
      <AbsoluteFill style={{ background: "#fff", opacity: impact * impact * 0.7 }} />
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 110 }}>
        <div style={{ fontFamily: F.mono, fontSize: 40, color: press > 0.99 ? C.goldHot : C.indigo, opacity: 1 - warp }}>
          {press > 0.99 ? "4-BIT · 7.9 MB" : "16-BIT · 27.8 MB"}
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ ...center, flexDirection: "column", opacity: warp }}>
        <div style={{ fontFamily: F.sans, fontWeight: 800, fontSize: 280, letterSpacing: "-0.05em", color: "#fff", transform: `scale(${k(t, 104, 130, 1.8, 1)})`, textShadow: deepText("#fff", "#3a2f80", "rgba(129,140,248,0.9)", 16) }}>832</div>
        <div style={{ fontFamily: F.mono, fontSize: 40, color: C.goldHot, letterSpacing: "0.2em" }}>TOKENS / SECOND</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// =========================================================================
// 9 · Everywhere, no cloud (1412–1610): devices on a mirror floor at dawn;
//     the clouds above part and drift away
// =========================================================================
const Screen: React.FC<{ t: number }> = ({ t }) => (
  <div style={{ position: "absolute", inset: 0, padding: 18, fontFamily: F.serif, fontSize: 22, color: "#2a2418", background: "linear-gradient(170deg, #fbf6e9, #e9e0c9)" }}>
    {"Once upon a time, a little owl named Pip found a key.".slice(0, Math.max(0, Math.floor(t * 0.6)))}
  </div>
);
const S9: React.FC<{ t: number }> = ({ t }) => {
  const a = k(t, 6, 30), b = k(t, 30, 54), c = k(t, 54, 78);
  const part = k(t, 96, 150);
  const devices = (
    <div style={{ position: "relative", width: 1500, height: 520 }}>
      <div style={{ position: "absolute", left: 60, bottom: 0, opacity: a, transform: `translateY(${(1 - a) * 80}px)` }}>
        <div style={{ position: "relative", width: 620, height: 390, borderRadius: 18, border: "10px solid #1c1f2c", overflow: "hidden", boxShadow: `0 0 80px rgba(255,200,120,0.35)` }}><Screen t={t - 6} /></div>
        <div style={{ width: 720, height: 22, marginLeft: -50, borderRadius: "0 0 20px 20px", background: "linear-gradient(180deg, #c9ccd6, #6c7080)" }} />
      </div>
      <div style={{ position: "absolute", left: 760, bottom: 0, width: 210, height: 430, borderRadius: 38, border: "10px solid #1c1f2c", overflow: "hidden", opacity: b, transform: `translateY(${(1 - b) * 80}px)`, boxShadow: `0 0 80px rgba(255,200,120,0.35)` }}><Screen t={t - 30} /></div>
      <div style={{ position: "absolute", left: 1040, bottom: 0, width: 460, height: 340, borderRadius: 14, overflow: "hidden", border: "2px solid #3a3f55", opacity: c, transform: `translateY(${(1 - c) * 80}px)`, boxShadow: `0 0 80px rgba(255,200,120,0.35)` }}>
        <div style={{ height: 40, background: "#1c1f2c", display: "flex", alignItems: "center", gap: 8, paddingLeft: 14 }}>{[C.rose, C.gold, C.emerald].map((col) => <div key={col} style={{ width: 12, height: 12, borderRadius: 6, background: col }} />)}</div>
        <div style={{ position: "absolute", inset: "40px 0 0 0" }}><Screen t={t - 54} /></div>
      </div>
    </div>
  );
  return (
    <AbsoluteFill style={{ background: "linear-gradient(180deg, #0a0f2a 0%, #3a2a5a 45%, #ff9a5a 72%, #2a1a20 72.2%, #07060c 100%)" }}>
      <div style={{ position: "absolute", left: 960 - 160, top: 590 - 160, width: 320, height: 320, borderRadius: "50%", background: `radial-gradient(circle, #fff6d8, ${C.goldHot} 35%, rgba(255,150,80,0) 70%)` }} />
      <GodRays t={t} x={50} y={55} opacity={0.5 + part * 0.5} />
      {/* clouds overhead that part */}
      {Array.from({ length: 10 }, (_, i) => {
        const left = i < 5;
        return <div key={i} style={{ position: "absolute", top: 40 + (i % 5) * 50, left: `${(left ? 10 + (i % 5) * 9 : 50 + (i % 5) * 9) + (left ? -1 : 1) * part * 70}%`, width: 520, height: 200, borderRadius: "50%", background: "radial-gradient(ellipse, rgba(200,190,230,0.55), rgba(200,190,230,0) 70%)", filter: "blur(18px)", opacity: 1 - part * 0.7 }} />;
      })}
      <AbsoluteFill style={{ ...center, paddingTop: 120 }}>
        <div style={{ position: "relative" }}>
          {devices}
          <div style={{ position: "absolute", top: 524, transform: "scaleY(-1)", opacity: 0.25, WebkitMaskImage: "linear-gradient(0deg, #000, transparent 60%)", maskImage: "linear-gradient(0deg, #000, transparent 60%)", filter: "blur(2px)" }}>{devices}</div>
        </div>
      </AbsoluteFill>
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 150 }}>
        <div style={{ display: "flex", gap: 50, fontFamily: F.sans, fontWeight: 800, fontSize: 40, letterSpacing: "0.2em", color: C.ivory }}>
          <span style={{ opacity: a }}>MAC</span><span style={{ opacity: b }}>IPHONE</span><span style={{ opacity: c }}>ANY BROWSER</span>
        </div>
        <div style={{ marginTop: 20, fontFamily: F.serif, fontStyle: "italic", fontSize: 170, color: "#fff", opacity: part, transform: `scale(${k(t, 96, 130, 1.3, 1)})`, textShadow: deepText("#fff", "#5a3a50", "rgba(255,200,140,0.8)", 14) }}>Zero cloud.</div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// =========================================================================
// 10 · Meet Hedwig (1610–1800): the owl flies across a rising moon and lands
//      above the title
// =========================================================================
const S10: React.FC<{ t: number }> = ({ t }) => {
  const moon = k(t, 0, 80, 300, 0);
  const fly = k(t, 0, 90, 0, 1, (u) => u);
  const land = k(t, 80, 110);
  const ox = interpolate(fly, [0, 1], [-300, 960]);
  const oy = 420 - Math.sin(fly * Math.PI) * 180 + land * 0;
  const flap = land > 0.9 ? Math.sin(t / 30) * 0.05 : Math.sin(t / 2.4);
  const title = k(t, 96, 130);
  const shimmer = k(t, 120, 170, -30, 130, (u) => u);
  const trail = Array.from({ length: 40 }, (_, i) => {
    const f = Math.max(0, fly - i * 0.012);
    return { x: interpolate(f, [0, 1], [-300, 960]), y: 420 - Math.sin(f * Math.PI) * 180 + 40, o: (1 - i / 40) * (1 - land) };
  });
  return (
    <AbsoluteFill>
      <Sky t={t} panY={-moon * 0.3} top="#010208" bottom="#141a40" />
      <div style={{ position: "absolute", left: 960 - 330, top: 120 + moon, width: 660, height: 660, borderRadius: "50%", background: "radial-gradient(circle at 40% 38%, #fffdf2, #e8e2cc 55%, #b8b09a 100%)", boxShadow: "0 0 160px 40px rgba(255,245,210,0.35)" }}>
        {[[180, 200, 70], [400, 330, 50], [260, 450, 90], [470, 160, 40]].map(([x, y, r], i) => <div key={i} style={{ position: "absolute", left: x, top: y, width: r, height: r, borderRadius: "50%", background: "rgba(150,140,120,0.25)" }} />)}
      </div>
      <Fog t={t} y={78} color="150,160,230" />
      {trail.map((p, i) => <div key={i} style={{ position: "absolute", left: p.x, top: p.y, width: 8, height: 8, borderRadius: "50%", background: C.goldHot, opacity: p.o, boxShadow: `0 0 16px ${C.gold}` }} />)}
      <div style={{ position: "absolute", left: ox - 130, top: oy - 130, transform: `rotate(${(1 - land) * 12}deg) scale(${1 + land * 0.1})` }}>
        <Owl size={260} flap={flap} look={land} blink={t > 150 && t < 155 ? 1 : 0} />
      </div>
      <AbsoluteFill style={{ alignItems: "center", paddingTop: 560 }}>
        <div style={{ position: "relative", fontFamily: F.serif, fontSize: 220, lineHeight: 1, color: C.ivory, opacity: title, transform: `translateY(${(1 - title) * 40}px)`, letterSpacing: "0.02em", textShadow: deepText(C.ivory, "#2a2f50", "rgba(255,220,150,0.5)", 16) }}>
          <RevealText text="Hedwig" t={t} at={96} per={3} />
          <div style={{ position: "absolute", inset: 0, background: `linear-gradient(100deg, transparent ${shimmer - 12}%, rgba(255,240,200,0.75) ${shimmer}%, transparent ${shimmer + 12}%)`, mixBlendMode: "overlay" }} />
        </div>
        <div style={{ marginTop: 18, fontFamily: F.sans, fontWeight: 600, fontSize: 30, letterSpacing: "0.3em", color: C.goldHot, opacity: k(t, 124, 140) }}>OPEN WEIGHTS · OPEN CODE · YOURS FOREVER</div>
        <div style={{ marginTop: 14, fontFamily: F.mono, fontSize: 24, color: C.muted, opacity: k(t, 140, 156) }}>github.com/ansarbuilds/hedwig</div>
      </AbsoluteFill>
      <Bokeh t={t} opacity={0.6} />
    </AbsoluteFill>
  );
};

// =========================================================================
const SCENES: [number, number, React.FC<{ t: number }>][] = [
  [0, 130, S1], [130, 238, S2], [238, 372, S3], [372, 556, S4], [556, 762, S5],
  [762, 978, S6], [978, 1210, S7], [1210, 1412, S8], [1412, 1610, S9], [1610, 1800, S10],
];

export const HedwigFilm: React.FC = () => {
  const frame = useCurrentFrame();
  const rms = AUDIO_RMS[frame] ?? 0.25;
  const peak = AUDIO_PEAK[frame] ?? 0.3;
  return (
    <AbsoluteFill style={{ backgroundColor: C.night, overflow: "hidden" }}>
      <Audio src={staticFile("demo-soundtrack.wav")} />
      <CameraRig frame={frame} rms={rms}>
        {SCENES.map(([s, e, Comp], i) => (
          <Scene key={i} frame={frame} start={s} end={e}>
            {(t) => <Comp t={t} />}
          </Scene>
        ))}
      </CameraRig>
      <CinemaPost frame={frame} peak={peak} />
      <FilmSubtitles frame={frame} />
    </AbsoluteFill>
  );
};

// =========================================================================
// Text-free 6-second loop for the portfolio hero: moon, stars, Pip perched.
// Everything moves on a 180-frame cycle so the end meets the start.
// =========================================================================
export const HedwigHeroLoop: React.FC = () => {
  const f = useCurrentFrame();
  const cyc = (f % 180) / 180;
  const w = Math.sin(cyc * Math.PI * 2);
  const blink = f % 180 > 120 && f % 180 < 126 ? 1 : 0;
  return (
    <AbsoluteFill style={{ backgroundColor: C.night }}>
      <Sky t={cyc * 28 * Math.PI} top="#010208" bottom="#141a40" />
      <div style={{ position: "absolute", left: 960 - 330, top: 150, width: 660, height: 660, borderRadius: "50%", background: "radial-gradient(circle at 40% 38%, #fffdf2, #e8e2cc 55%, #b8b09a 100%)", boxShadow: "0 0 160px 40px rgba(255,245,210,0.35)" }}>
        {[[180, 200, 70], [400, 330, 50], [260, 450, 90], [470, 160, 40]].map(([x, y, r], i) => <div key={i} style={{ position: "absolute", left: x, top: y, width: r, height: r, borderRadius: "50%", background: "rgba(150,140,120,0.25)" }} />)}
      </div>
      <Fog t={w * 40 + 40} y={74} color="150,160,230" />
      <div style={{ position: "absolute", left: 960 - 150, top: 330 + w * 6, transform: `rotate(${w * 1.5}deg)` }}>
        <Owl size={300} flap={w * 0.06} look={Math.sin(cyc * Math.PI * 4) * 0.6} blink={blink} />
      </div>
      <Bokeh t={w * 60} opacity={0.6} />
    </AbsoluteFill>
  );
};
