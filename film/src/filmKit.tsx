// Shared "movie set" pieces for the redesigned film: sky, fog, light rays,
// out-of-focus foreground dust, deep 3D type, and Pip the owl.
import React from "react";
import { AbsoluteFill, Easing, interpolate, random } from "remotion";
import { loadFont } from "@remotion/fonts";
import instrumentSerif400 from "@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff2";
import instrumentSerif400Italic from "@fontsource/instrument-serif/files/instrument-serif-latin-400-italic.woff2";
import plusJakarta600 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-600-normal.woff2";
import plusJakarta800 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-800-normal.woff2";
import jetbrains500 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2";

loadFont({ family: "Instrument Serif", url: instrumentSerif400, weight: "400", style: "normal" });
loadFont({ family: "Instrument Serif", url: instrumentSerif400Italic, weight: "400", style: "italic" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta600, weight: "600", style: "normal" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta800, weight: "800", style: "normal" });
loadFont({ family: "JetBrains Mono", url: jetbrains500, weight: "500", style: "normal" });

export const C = {
  night: "#05060c",
  navy: "#0b1024",
  ivory: "#f8f4ea",
  muted: "#8d93a8",
  gold: "#f5b841",
  goldHot: "#ffdf8a",
  ember: "#ff7a3d",
  cyan: "#5ee7ff",
  indigo: "#818cf8",
  emerald: "#34d399",
  rose: "#fb7185",
};
export const F = {
  serif: "'Instrument Serif', Georgia, serif",
  sans: "'Plus Jakarta Sans', -apple-system, sans-serif",
  mono: "'JetBrains Mono', monospace",
};

// Smooth "film" ease: fast start, long gentle landing.
export const EASE = Easing.bezier(0.16, 1, 0.3, 1);
export const k = (f: number, a: number, b: number, from = 0, to = 1, easing = EASE) =>
  interpolate(f, [a, b], [from, to], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing });

// A scene lives between start and end; children get local time t.
export const Scene: React.FC<{
  frame: number;
  start: number;
  end: number;
  children: (t: number) => React.ReactNode;
}> = ({ frame, start, end, children }) => {
  if (frame < start - 1 || frame > end + 1) return null;
  const o = Math.min(k(frame, start - 1, start + 6), 1 - k(frame, end - 6, end + 1));
  return <AbsoluteFill style={{ opacity: o, overflow: "hidden" }}>{children(frame - start)}</AbsoluteFill>;
};

// ---- Sky with parallax stars --------------------------------------------
const STARS = Array.from({ length: 260 }, (_, i) => ({
  x: random(`sx${i}`) * 2200 - 140,
  y: random(`sy${i}`) * 1500 - 200,
  r: 0.4 + Math.pow(random(`sr${i}`), 3) * 2.4,
  d: 0.2 + random(`sd${i}`) * 0.8,
  tw: random(`st${i}`) * 6.28,
}));
export const Sky: React.FC<{ t: number; panY?: number; panX?: number; top?: string; bottom?: string }> = ({
  t,
  panY = 0,
  panX = 0,
  top = "#02030a",
  bottom = "#141a3a",
}) => (
  <AbsoluteFill style={{ background: `linear-gradient(180deg, ${top} 0%, ${bottom} 100%)` }}>
    {STARS.map((s, i) => (
      <div
        key={i}
        style={{
          position: "absolute",
          left: s.x + panX * s.d,
          top: s.y + panY * s.d,
          width: s.r * 2,
          height: s.r * 2,
          borderRadius: "50%",
          background: "#fff",
          opacity: 0.35 + 0.65 * Math.abs(Math.sin(t / 14 + s.tw)) * s.d,
          boxShadow: s.r > 1.6 ? `0 0 ${s.r * 4}px rgba(190,210,255,0.9)` : undefined,
        }}
      />
    ))}
  </AbsoluteFill>
);

// ---- Drifting fog banks ---------------------------------------------------
export const Fog: React.FC<{ t: number; color?: string; y?: number; opacity?: number }> = ({
  t,
  color = "120,140,220",
  y = 70,
  opacity = 1,
}) => (
  <AbsoluteFill style={{ opacity, pointerEvents: "none" }}>
    {[0, 1, 2, 3].map((i) => (
      <div
        key={i}
        style={{
          position: "absolute",
          left: `${-30 + ((i * 37 + t * (0.15 + i * 0.05)) % 140)}%`,
          top: `${y - 10 + i * 6}%`,
          width: 1100,
          height: 340,
          borderRadius: "50%",
          background: `radial-gradient(ellipse, rgba(${color},0.22), rgba(${color},0) 70%)`,
          filter: "blur(30px)",
        }}
      />
    ))}
  </AbsoluteFill>
);

// ---- Volumetric light rays from a point ---------------------------------
export const GodRays: React.FC<{ t: number; x: number; y: number; color?: string; opacity?: number }> = ({
  t,
  x,
  y,
  color = "255,210,140",
  opacity = 1,
}) => (
  <AbsoluteFill
    style={{
      mixBlendMode: "screen",
      opacity,
      background: `repeating-conic-gradient(from ${t * 0.15}deg at ${x}% ${y}%, rgba(${color},0.20) 0deg 2.2deg, rgba(${color},0) 4deg 9deg)`,
      WebkitMaskImage: `radial-gradient(circle at ${x}% ${y}%, #000 0%, rgba(0,0,0,0.5) 30%, transparent 70%)`,
      maskImage: `radial-gradient(circle at ${x}% ${y}%, #000 0%, rgba(0,0,0,0.5) 30%, transparent 70%)`,
      filter: "blur(6px)",
    }}
  />
);

// ---- Out-of-focus foreground particles (depth of field) -----------------
const BOKEH = Array.from({ length: 22 }, (_, i) => ({
  x: random(`bx${i}`) * 100,
  y: random(`by${i}`) * 100,
  s: 20 + random(`bs${i}`) * 90,
  sp: 0.2 + random(`bp${i}`) * 0.6,
  gold: random(`bg${i}`) > 0.4,
}));
export const Bokeh: React.FC<{ t: number; opacity?: number }> = ({ t, opacity = 0.7 }) => (
  <AbsoluteFill style={{ mixBlendMode: "screen", opacity, pointerEvents: "none" }}>
    {BOKEH.map((b, i) => (
      <div
        key={i}
        style={{
          position: "absolute",
          left: `${(b.x + t * b.sp * 0.08) % 110 - 5}%`,
          top: `${b.y + Math.sin(t / 40 + i) * 3}%`,
          width: b.s,
          height: b.s,
          borderRadius: "50%",
          background: b.gold
            ? "radial-gradient(circle, rgba(255,200,120,0.35), rgba(255,200,120,0.12) 60%, transparent 72%)"
            : "radial-gradient(circle, rgba(140,160,255,0.30), rgba(140,160,255,0.10) 60%, transparent 72%)",
          filter: `blur(${b.s / 9}px)`,
        }}
      />
    ))}
  </AbsoluteFill>
);

// ---- Rising embers --------------------------------------------------------
export const Embers: React.FC<{ t: number; n?: number; x?: number; w?: number; opacity?: number }> = ({
  t,
  n = 60,
  x = 50,
  w = 60,
  opacity = 1,
}) => (
  <AbsoluteFill style={{ mixBlendMode: "screen", opacity, pointerEvents: "none" }}>
    {Array.from({ length: n }, (_, i) => {
      const life = 90 + random(`el${i}`) * 80;
      const p = ((t + random(`eo${i}`) * life) % life) / life;
      const px = x - w / 2 + random(`ex${i}`) * w + Math.sin(p * 9 + i) * 2;
      const size = 2 + random(`es${i}`) * 4;
      return (
        <div
          key={i}
          style={{
            position: "absolute",
            left: `${px}%`,
            top: `${100 - p * 90}%`,
            width: size,
            height: size,
            borderRadius: "50%",
            background: p < 0.5 ? C.goldHot : C.ember,
            opacity: Math.sin(p * Math.PI),
            boxShadow: `0 0 ${size * 4}px ${C.ember}`,
          }}
        />
      );
    })}
  </AbsoluteFill>
);

// ---- Deep 3D type: stacked shadows = solid extruded letters + glow -----
export const deepText = (face: string, side: string, glow: string, depth = 14) =>
  [
    ...Array.from({ length: depth }, (_, i) => `0 ${i + 1}px 0 ${side}`),
    `0 ${depth + 6}px 24px rgba(0,0,0,0.6)`,
    `0 0 60px ${glow}`,
  ].join(", ");

// Letters rise out of blur one by one.
export const RevealText: React.FC<{
  text: string;
  t: number;
  at: number;
  per?: number;
  style?: React.CSSProperties;
}> = ({ text, t, at, per = 1.6, style }) => (
  <span style={{ display: "inline-block", whiteSpace: "pre", ...style }}>
    {text.split("").map((ch, i) => {
      const u = k(t, at + i * per, at + i * per + 16);
      return (
        <span
          key={i}
          style={{
            display: "inline-block",
            opacity: u,
            transform: `translateY(${(1 - u) * 60}px) rotateX(${(1 - u) * -70}deg)`,
            filter: `blur(${(1 - u) * 14}px)`,
          }}
        >
          {ch}
        </span>
      );
    })}
  </span>
);

// ---- Pip / Hedwig the Snowy Owl (modeled after real snowy owl portrait) ---
// flap: -1..1 wing motion, blink: 0 open .. 1 shut, look: pupil x offset
const SPECKLES: [number, number, number, number][] = [
  [-6, 12, 4.6, -20], [6, 17, 4.2, -14], [-18, 22, 5.2, -24], [-4, 26, 4.5, -16],
  [9, 28, 4.0, -10], [-28, 32, 5.4, -26], [-14, 35, 4.8, -20], [1, 37, 4.2, -14],
  [-36, 42, 5.6, -28], [-22, 45, 5.0, -22], [-7, 47, 4.4, -16], [7, 48, 3.8, -12],
  [-44, 52, 5.5, -30], [-30, 55, 5.0, -25], [-16, 57, 4.4, -19], [-50, 62, 5.2, -32],
  [-37, 64, 4.8, -27], [-23, 66, 4.2, -22], [-56, 72, 4.8, -34], [-43, 74, 4.4, -28],
  [-60, 81, 4.2, -36], [-49, 82, 3.9, -30], [-14, 6, 3.6, -14], [-22, 16, 4.0, -20],
];
export const Owl: React.FC<{ size: number; flap?: number; blink?: number; look?: number; glow?: number }> = ({
  size,
  flap = 0,
  blink = 0,
  look = 0,
  glow = 1,
}) => {
  const flight = Math.min(1, Math.max(0, (Math.abs(flap) - 0.1) * 3.5));
  const wingTilt = flap * 26;
  const eyeH = 6.8 * (1 - blink) + 0.7;
  return (
    <svg width={size} height={size} viewBox="-100 -100 200 200" style={{ overflow: "visible" }}>
      <defs>
        <radialGradient id="snowyBody" cx="58%" cy="24%" r="76%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="68%" stopColor="#f6f2ea" />
          <stop offset="100%" stopColor="#dcd4c5" />
        </radialGradient>
        <linearGradient id="snowyWing" x1="82%" y1="12%" x2="10%" y2="96%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="60%" stopColor="#f2ede3" />
          <stop offset="100%" stopColor="#cec5b5" />
        </linearGradient>
        <radialGradient id="snowyEye" cx="38%" cy="38%" r="62%">
          <stop offset="0%" stopColor="#ffd75e" />
          <stop offset="55%" stopColor="#db8700" />
          <stop offset="100%" stopColor="#693000" />
        </radialGradient>
        <filter id="owlGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {/* subtle dark perch silhouette under feet when perched (echoing glove/branch silhouette) */}
      <g opacity={1 - flight * 0.9}>
        <path
          d="M-38 78 Q -10 66 26 70 Q 54 73 74 91 L 66 97 Q 28 84 -8 86 Q -28 89 -42 95 Z"
          fill="#1b243c"
          stroke="#314169"
          strokeWidth="1.4"
        />
        <path d="M-38 78 Q -18 72 -4 76 L -8 87 Q -24 88 -40 94 Z" fill="#3a5280" opacity="0.7" />
      </g>

      {/* rear flight wing (extends smoothly only during flight flap) */}
      {flight > 0.02 && (
        <g opacity={flight} transform={`translate(-10 -14) rotate(${-28 - flap * 44}) scale(${0.65 + flight * 0.35})`}>
          <path
            d="M0 0 Q -62 -32 -98 6 Q -70 16 -52 38 Q -30 22 -14 42 Q -4 22 0 32 Z"
            fill="url(#snowyWing)"
            stroke="#c5bcab"
            strokeWidth="1.2"
          />
          {[-26, -44, -62, -76].map((sx, idx) => (
            <path key={idx} d={`M${sx} ${6 + idx * 3} q5 2 3 8 q-4 -3 -3 -8 Z`} fill="#4e341b" opacity="0.75" />
          ))}
        </g>
      )}

      {/* tail primary tips projecting bottom-left */}
      <g transform="rotate(6)">
        <path
          d="M-36 52 L-62 96 Q-55 98 -48 92 L-24 58 Z M-26 56 L-50 98 Q-43 100 -36 93 L-16 62 Z"
          fill="#e6dfd0"
          stroke="#c3baa8"
          strokeWidth="1"
        />
        <path d="M-50 78 q4 2 2 6 q-4 -2 -2 -6 Z M-40 82 q4 2 2 6 q-4 -2 -2 -6 Z" fill="#4e341b" opacity="0.78" />
      </g>

      {/* unified snowy-white silhouette: round crown seamlessly flowing into chest & shoulder */}
      <path
        d="M -14 -50 C -2 -70 30 -70 42 -48 C 48 -36 47 -22 46 -6 C 48 24 34 56 -6 66 C -34 72 -52 46 -40 14 C -32 -8 -24 -34 -14 -50 Z"
        fill="url(#snowyBody)"
        stroke="#9a9182"
        strokeWidth="1.35"
      />
      <path d="M -6 -58 Q 14 -68 36 -56" stroke="#ffffff" strokeWidth="2.5" fill="none" opacity="0.8" />

      {/* fluffy feathered leg & talons */}
      <path
        d="M-10 48 C -14 66 -2 76 14 74 C 24 73 28 60 22 48 Z"
        fill="#f7f4ec"
        stroke="#ddd4c4"
        strokeWidth="1"
      />
      <path
        d="M2 72 Q -2 80 4 82 M10 73 Q 10 82 16 82 M18 71 Q 22 79 26 78"
        stroke="#22252e"
        strokeWidth="3.1"
        strokeLinecap="round"
        fill="none"
      />

      {/* folded diagonal foreground wing with authentic brown-speckled feathers */}
      <g transform={`translate(24 -10) rotate(${wingTilt})`}>
        <path
          d="M 8 -14 C 26 4 12 36 -18 62 C -46 86 -72 98 -82 96 C -80 76 -64 42 -42 16 C -22 -8 -4 -22 8 -14 Z"
          fill="url(#snowyWing)"
          stroke="#d5ccbc"
          strokeWidth="1.1"
        />
        {/* primary feather overlap lines */}
        <path
          d="M -24 26 Q -52 58 -76 92 M -14 34 Q -40 62 -64 88"
          stroke="#c8bfb0"
          strokeWidth="1"
          fill="none"
          opacity="0.7"
        />
        {SPECKLES.map(([x, y, s, rot], idx) => (
          <path
            key={idx}
            d={`M${x} ${y} Q${x + s} ${y + s * 0.25} ${x + s * 0.4} ${y + s * 1.1} Q${x - s * 0.2} ${y + s * 0.6} ${x} ${y} Z`}
            fill={idx % 3 === 0 ? "#3f2713" : "#5a3d21"}
            opacity={0.86 - (idx % 4) * 0.08}
            transform={`rotate(${rot} ${x} ${y})`}
          />
        ))}
      </g>

      {/* subtle facial disc shading (no harsh chin outline) */}
      <path
        d="M -6 -28 Q 14 -16 34 -28 M -10 -46 Q 14 -56 38 -46"
        stroke="#e6decb"
        strokeWidth="1.1"
        fill="none"
        opacity="0.45"
      />

      {/* expressive almond golden-amber snowy owl eyes with dark upper brow */}
      {[
        { ex: 2, ey: -40, tilt: 10 },
        { ex: 24, ey: -40, tilt: -10 },
      ].map(({ ex, ey, tilt }) => (
        <g key={ex} transform={`rotate(${tilt} ${ex} ${ey})`}>
          <ellipse cx={ex} cy={ey} rx="8.5" ry={eyeH + 1.5} fill={C.gold} opacity={0.35 * glow} filter="url(#owlGlow)" />
          <ellipse cx={ex} cy={ey} rx="7.6" ry={eyeH} fill="url(#snowyEye)" stroke="#241b11" strokeWidth="1.4" />
          <ellipse cx={ex + look * 2.8} cy={ey} rx="3.4" ry={Math.min(eyeH, 3.9)} fill="#0c0905" />
          <circle cx={ex + look * 2.8 - 1.5} cy={ey - 1.8} r={blink > 0.6 ? 0 : 1.4} fill="#ffffff" />
          <path d={`M${ex - 8} ${ey - 1} Q${ex} ${ey - 8.5} ${ex + 8} ${ey - 1}`} stroke="#2a2015" strokeWidth="1.9" fill="none" opacity={0.78 * (1 - blink * 0.5)} />
        </g>
      ))}

      {/* short dark charcoal curved beak nestled in fluffy white bristles */}
      <path d="M 10 -34 Q 13 -37 16 -34 L 14 -24 Q 13 -22 11 -24 Z" fill="#23252b" stroke="#0c0d10" strokeWidth="1" />
      <path d="M 6 -32 Q 10 -30 11 -28 M 20 -32 Q 16 -30 15 -28" stroke="#ffffff" strokeWidth="2.1" strokeLinecap="round" />
    </svg>
  );
};

// A glowing glass panel (for holograms / UI in 3D space)
export const glass = (accent: string): React.CSSProperties => ({
  background: "linear-gradient(160deg, rgba(30,36,64,0.72), rgba(10,12,26,0.82))",
  border: `1px solid ${accent}55`,
  borderRadius: 22,
  boxShadow: `0 40px 120px rgba(0,0,0,0.7), 0 0 60px ${accent}22, inset 0 1px 0 rgba(255,255,255,0.08)`,
});
