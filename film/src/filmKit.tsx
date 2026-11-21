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

// ---- Pip the owl (original design) -------------------------------------
// flap: -1..1 wing angle, blink: 0 open .. 1 shut, look: pupil x offset
export const Owl: React.FC<{ size: number; flap?: number; blink?: number; look?: number; glow?: number }> = ({
  size,
  flap = 0,
  blink = 0,
  look = 0,
  glow = 1,
}) => {
  const wing = flap * 55;
  const eyeH = 15 * (1 - blink) + 0.8;
  return (
    <svg width={size} height={size} viewBox="-100 -100 200 200" style={{ overflow: "visible" }}>
      <defs>
        <radialGradient id="owlBody" cx="50%" cy="35%" r="70%">
          <stop offset="0%" stopColor="#fffaf0" />
          <stop offset="70%" stopColor="#d9d2c3" />
          <stop offset="100%" stopColor="#8f8676" />
        </radialGradient>
        <radialGradient id="owlEye" cx="40%" cy="40%" r="60%">
          <stop offset="0%" stopColor="#fff2b0" />
          <stop offset="55%" stopColor={C.gold} />
          <stop offset="100%" stopColor="#b86a00" />
        </radialGradient>
        <filter id="owlGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" />
        </filter>
      </defs>
      {/* wings */}
      {[-1, 1].map((sd) => (
        <g key={sd} transform={`translate(${sd * 34} -8) rotate(${sd * (20 + wing)})`}>
          <path
            d={`M0 0 Q ${sd * 70} -10 ${sd * 92} 40 Q ${sd * 60} 34 ${sd * 48} 58 Q ${sd * 30} 40 ${sd * 18} 62 Q ${sd * 6} 40 0 52 Z`}
            fill="url(#owlBody)"
            stroke="#6f6758"
            strokeWidth="1.5"
          />
          {[0, 1, 2].map((r) => (
            <path
              key={r}
              d={`M${sd * (14 + r * 18)} ${14 + r * 4} q ${sd * 6} 10 ${sd * 2} 20`}
              stroke="#8a8170"
              strokeWidth="2"
              fill="none"
              opacity="0.7"
            />
          ))}
        </g>
      ))}
      {/* body + head */}
      <ellipse cx="0" cy="22" rx="46" ry="56" fill="url(#owlBody)" />
      {[0, 1, 2, 3].map((r) =>
        [-1, 0, 1].map((c) => (
          <path key={`${r}${c}`} d={`M${c * 16 - 5} ${30 + r * 12} q5 5 10 0`} stroke="#9b917f" strokeWidth="2" fill="none" />
        )),
      )}
      <path d="M-40 -40 L-30 -78 L-14 -52 Z M40 -40 L30 -78 L14 -52 Z" fill="#cfc6b4" />
      <circle cx="0" cy="-28" r="44" fill="url(#owlBody)" />
      <circle cx="-19" cy="-30" r="22" fill="#f3ecdc" />
      <circle cx="19" cy="-30" r="22" fill="#f3ecdc" />
      {/* eyes with glow */}
      {[-19, 19].map((ex) => (
        <g key={ex}>
          <ellipse cx={ex} cy="-30" rx="17" ry={eyeH + 4} fill={C.gold} opacity={0.6 * glow} filter="url(#owlGlow)" />
          <ellipse cx={ex} cy="-30" rx="15" ry={eyeH} fill="url(#owlEye)" />
          <ellipse cx={ex + look * 5} cy="-30" rx="6.5" ry={Math.min(eyeH, 7.5)} fill="#140d02" />
          <circle cx={ex + look * 5 - 3} cy="-34" r={blink > 0.6 ? 0 : 2.4} fill="#fff" />
        </g>
      ))}
      <path d="M-7 -14 L7 -14 L0 2 Z" fill="#e2a23a" stroke="#8a5a10" strokeWidth="1" />
      <path d="M-14 76 l-4 10 M-8 76 l0 11 M8 76 l0 11 M14 76 l4 10" stroke="#e2a23a" strokeWidth="3" strokeLinecap="round" />
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
