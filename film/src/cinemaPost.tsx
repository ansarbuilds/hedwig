// Hollywood-style finishing layer: camera, lens, grade, grain, letterbox.
// Everything here sits ON TOP of (or wraps) the existing beats, so the
// story and timing stay exactly the same. Think of it as the "post house"
// pass a real film gets after the edit is locked.
import React from "react";
import { AbsoluteFill, Easing, interpolate, random } from "remotion";

const TOTAL = 1800;
// Scene cuts (frames). Small shift before each voice line = where the picture changes.
export const CUTS = [130, 238, 372, 556, 762, 978, 1210, 1412, 1610];

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

// Frames since the most recent cut (Infinity if none yet).
const sinceCut = (f: number) => {
  let best = Infinity;
  for (const c of CUTS) if (f >= c) best = f - c;
  return best;
};

// --- 1. Camera rig: slow dolly-in, handheld breathing, punch-in on cuts ---
export const CameraRig: React.FC<{ frame: number; rms: number; children: React.ReactNode }> = ({
  frame,
  rms,
  children,
}) => {
  const t = frame / 30;
  // Layered sine "steadicam" drift (sub-pixel to a few px — subtle, not seasick)
  const dx = Math.sin(t * 0.7) * 16 + Math.sin(t * 1.9 + 1.3) * 5;
  const dy = Math.cos(t * 0.55) * 11 + Math.sin(t * 2.3) * 3.5;
  const rot = Math.sin(t * 0.42) * 0.6;
  // Each shot slowly pushes in, like a dolly
  const s = sinceCut(frame);
  const dolly = 1 + Math.min(s === Infinity ? frame : s, 220) * 0.0007;
  // Punch-in kick on the cut, settles with an ease-out
  const punch = s < 14 ? interpolate(s, [0, 14], [0.14, 0], { ...clamp, easing: Easing.out(Easing.cubic) }) : 0;
  // Bass "thump" from the soundtrack
  const thump = Math.max(0, rms - 0.35) * 0.06;
  const scale = dolly + punch + thump;
  // Whip-pan into each cut: last 5 frames of a shot + first 6 of the next smear sideways
  const next = CUTS.find((c) => c > frame) ?? Infinity;
  const toNext = next - frame;
  const blur = s < 6 ? (6 - s) * 3 : toNext <= 5 ? (6 - toNext) * 3 : 0;
  const whip = s < 6 ? -(6 - s) * 40 : toNext <= 5 ? (6 - toNext) * 40 : 0;
  return (
    <AbsoluteFill
      style={{
        transform: `translate(${dx + whip}px, ${dy}px) rotate(${rot}deg) scale(${scale})`,
        filter: `blur(${blur}px) contrast(1.18) saturate(1.3) brightness(1.04)`,
        transformOrigin: "50% 50%",
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

// --- 2. Color grade: teal shadows / warm highlights + soft bloom ---
const Grade: React.FC = () => (
  <>
    <AbsoluteFill
      style={{
        mixBlendMode: "soft-light",
        background: "linear-gradient(180deg, rgba(0,90,110,0.55) 0%, rgba(0,0,0,0) 45%, rgba(255,140,50,0.45) 100%)",
      }}
    />
    <AbsoluteFill
      style={{
        mixBlendMode: "screen",
        background: "radial-gradient(ellipse 60% 45% at 50% 48%, rgba(255,210,150,0.16), rgba(0,0,0,0) 70%)",
      }}
    />
  </>
);

// --- 3. Anamorphic lens flare: horizontal blue streak that tracks loud moments ---
const Flare: React.FC<{ frame: number; peak: number }> = ({ frame, peak }) => {
  const s = sinceCut(frame);
  const cutBurst = s < 20 ? interpolate(s, [0, 4, 20], [0, 1, 0], clamp) : 0;
  const k = Math.min(1, Math.max(cutBurst, 0.4 + (peak - 0.45) * 1.6));
  if (k <= 0.01) return null;
  const x = 50 + Math.sin(frame / 47) * 18;
  const y = 42 + Math.cos(frame / 61) * 8;
  return (
    <AbsoluteFill style={{ mixBlendMode: "screen", opacity: k }}>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: `${y}%`,
          height: 14,
          marginTop: -7,
          background: `radial-gradient(ellipse 55% 50% at ${x}% 50%, rgba(160,200,255,0.95), rgba(80,120,255,0.35) 40%, rgba(0,0,0,0) 75%)`,
          filter: "blur(2px)",
        }}
      />
      <div
        style={{
          position: "absolute",
          left: `${x}%`,
          top: `${y}%`,
          width: 260,
          height: 260,
          margin: -130,
          borderRadius: "50%",
          background: "radial-gradient(circle, rgba(255,240,220,0.55), rgba(255,180,90,0.12) 40%, rgba(0,0,0,0) 70%)",
        }}
      />
      {/* Ghost reflections on the opposite side of center, like real glass */}
      {[0.35, 0.6, 0.85].map((g, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: `${x + (50 - x) * 2 * g}%`,
            top: `${y + (50 - y) * 2 * g}%`,
            width: 40 + i * 30,
            height: 40 + i * 30,
            margin: -(20 + i * 15),
            borderRadius: "50%",
            border: "1px solid rgba(140,180,255,0.25)",
            background: "radial-gradient(circle, rgba(120,160,255,0.10), rgba(0,0,0,0) 70%)",
          }}
        />
      ))}
    </AbsoluteFill>
  );
};

// --- 4. Light leak: warm film-burn wash right after cuts ---
const LightLeak: React.FC<{ frame: number }> = ({ frame }) => {
  const s = sinceCut(frame);
  if (s > 26) return null;
  const o = interpolate(s, [0, 5, 26], [0, 0.95, 0], clamp);
  const side = Math.floor(random(`leak-${frame - s}`) * 2) ? 100 : 0;
  return (
    <AbsoluteFill
      style={{
        mixBlendMode: "screen",
        opacity: o,
        background: `radial-gradient(ellipse 45% 90% at ${side}% 40%, rgba(255,120,40,0.9), rgba(255,60,80,0.35) 45%, rgba(0,0,0,0) 75%)`,
      }}
    />
  );
};

// --- 5. White flash frame on the cut (2 frames, very fast) ---
const FlashFrame: React.FC<{ frame: number }> = ({ frame }) => {
  const s = sinceCut(frame);
  if (s > 3) return null;
  return <AbsoluteFill style={{ background: "#fff", opacity: [0.85, 0.45, 0.15, 0][s] }} />;
};

// --- 6. Film grain + dust + gate weave flicker ---
const Grain: React.FC<{ frame: number }> = ({ frame }) => {
  const seed = frame % 24; // 24 distinct grain plates, like real film stock looping
  const flicker = 0.97 + random(`flick-${frame}`) * 0.05;
  const dust = Array.from({ length: 5 }, (_, i) => {
    const r = random(`dust-${frame}-${i}`);
    if (r > 0.35) return null;
    return (
      <div
        key={i}
        style={{
          position: "absolute",
          left: `${random(`dx-${frame}-${i}`) * 100}%`,
          top: `${random(`dy-${frame}-${i}`) * 100}%`,
          width: 1 + r * 4,
          height: 1 + r * 4,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.35)",
        }}
      />
    );
  });
  return (
    <>
      <AbsoluteFill style={{ opacity: 0.22, mixBlendMode: "overlay" }}>
        <svg width="100%" height="100%">
          <filter id={`g${seed}`}>
            <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed={seed} stitchTiles="stitch" />
            <feColorMatrix type="saturate" values="0" />
          </filter>
          <rect width="100%" height="100%" filter={`url(#g${seed})`} />
        </svg>
      </AbsoluteFill>
      <AbsoluteFill>{dust}</AbsoluteFill>
      <AbsoluteFill style={{ background: "#000", opacity: 1 - flicker + 0.03 }} />
    </>
  );
};

// --- 7. Lens vignette + chromatic edge fringe ---
const Lens: React.FC = () => (
  <>
    <AbsoluteFill
      style={{
        background: "radial-gradient(ellipse 80% 75% at 50% 50%, rgba(0,0,0,0) 55%, rgba(0,0,0,0.8) 100%)",
      }}
    />
    <AbsoluteFill
      style={{
        mixBlendMode: "screen",
        boxShadow: "inset 6px 0 18px rgba(255,0,60,0.3), inset -6px 0 18px rgba(0,140,255,0.3)",
      }}
    />
  </>
);

// --- 8. Letterbox: 2.39:1 bars that open at the start, close at the end ---
const Letterbox: React.FC<{ frame: number }> = ({ frame }) => {
  const bar = 64; // ~2.1:1 — wide feel without cropping on-screen text
  const open = interpolate(frame, [0, 40], [540, bar], { ...clamp, easing: Easing.inOut(Easing.cubic) });
  const close = interpolate(frame, [TOTAL - 36, TOTAL - 1], [bar, 540], { ...clamp, easing: Easing.in(Easing.cubic) });
  // Bars breathe open to full frame for the hero typography beats (1 & 10)
  const heroOpen = interpolate(frame, [1600, 1660], [1, 0.45], clamp);
  const h = Math.max(open, close) * (frame > 1600 && frame < TOTAL - 36 ? heroOpen : 1);
  return (
    <>
      <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: h, background: "#000" }} />
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 0, height: h, background: "#000" }} />
    </>
  );
};

export const CinemaPost: React.FC<{ frame: number; peak: number }> = ({ frame, peak }) => (
  <AbsoluteFill style={{ pointerEvents: "none", zIndex: 60 }}>
    <Grade />
    <LightLeak frame={frame} />
    <Flare frame={frame} peak={peak} />
    <Lens />
    <Grain frame={frame} />
    <FlashFrame frame={frame} />
    <Letterbox frame={frame} />
  </AbsoluteFill>
);
