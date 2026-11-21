import React from "react";
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { loadFont } from "@remotion/fonts";
import instrumentSerif400 from "@fontsource/instrument-serif/files/instrument-serif-latin-400-normal.woff2";
import instrumentSerif400Italic from "@fontsource/instrument-serif/files/instrument-serif-latin-400-italic.woff2";
import plusJakarta400 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-400-normal.woff2";
import plusJakarta500 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-500-normal.woff2";
import plusJakarta600 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-600-normal.woff2";
import plusJakarta700 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-700-normal.woff2";
import plusJakarta800 from "@fontsource/plus-jakarta-sans/files/plus-jakarta-sans-latin-800-normal.woff2";
import jetbrains400 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2";
import jetbrains500 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-500-normal.woff2";
import jetbrains600 from "@fontsource/jetbrains-mono/files/jetbrains-mono-latin-600-normal.woff2";

import { AUDIO_PEAK, AUDIO_RMS } from "./audioData";
import {
  GoldenVortexCanvas,
  HorizonGrid3DCanvas,
  LightSpeedWarpCanvas,
  ParametricRibbon3DCanvas,
  ShatteringShards3DCanvas,
  TokenTerrain3DCanvas,
} from "./cinemaGraphics";
import { FilmSubtitles } from "./cinematic";

loadFont({ family: "Instrument Serif", url: instrumentSerif400, weight: "400", style: "normal" });
loadFont({ family: "Instrument Serif", url: instrumentSerif400Italic, weight: "400", style: "italic" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta400, weight: "400", style: "normal" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta500, weight: "500", style: "normal" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta600, weight: "600", style: "normal" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta700, weight: "700", style: "normal" });
loadFont({ family: "Plus Jakarta Sans", url: plusJakarta800, weight: "800", style: "normal" });
loadFont({ family: "JetBrains Mono", url: jetbrains400, weight: "400", style: "normal" });
loadFont({ family: "JetBrains Mono", url: jetbrains500, weight: "500", style: "normal" });
loadFont({ family: "JetBrains Mono", url: jetbrains600, weight: "600", style: "normal" });

const C = {
  bg: "#12141b",
  card: "#1b1e28",
  border: "#2c3140",
  ivory: "#f8f7f4",
  muted: "#9aa1b1",
  indigo: "#818cf8",
  indigoDeep: "#4f46e5",
  gold: "#f5b841",
  goldBright: "#ffd97d",
  emerald: "#34d399",
  rose: "#fb7185",
  parchment: "#f7f5f0",
  ink: "#1f2430",
  mutedLight: "#6b7280",
  lineLight: "#e5e1d8",
};

const F = {
  serif: "'Instrument Serif', Georgia, serif",
  sans: "'Plus Jakarta Sans', -apple-system, sans-serif",
  mono: "'JetBrains Mono', monospace",
};

const VW = 1920;
const VH = 1080;
const CX = VW / 2;
const CY = VH / 2;
const FPS = 30;

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
const seg = (t: number, a: number, b: number) =>
  b <= a ? (t >= b ? 1 : 0) : clamp((t - a) / (b - a), 0, 1);
const eo = (u: number) => 1 - Math.pow(1 - u, 3);
const eio = (u: number) =>
  u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
const bo = (u: number) => {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(u - 1, 3) + c1 * Math.pow(u - 1, 2);
};
const mb = (pxPerFrame: number) =>
  pxPerFrame > 5 ? Math.min(36, pxPerFrame * 0.22) : 0;

// ============================================================================
// 3D CAMERA RIG & DEPTH-OF-FIELD HELPERS (ported from saywhen cinema engine)
// ============================================================================
interface Cam3DKey {
  t: number;
  x?: number;
  y?: number;
  z?: number;
  rx?: number;
  ry?: number;
}

function evalCam3D(t: number, keys: Cam3DKey[], drift = true): React.CSSProperties {
  let a = keys[0];
  let b = keys[keys.length - 1];
  for (let i = 0; i < keys.length - 1; i++) {
    if (t >= keys[i].t && t <= keys[i + 1].t) {
      a = keys[i];
      b = keys[i + 1];
      break;
    }
  }
  if (t > b.t) a = b;
  const u = b.t > a.t ? eio(seg(t, a.t, b.t)) : 1;
  const x = lerp(a.x ?? 0, b.x ?? 0, u) + (drift ? 5 * Math.sin(t * 0.9 + 1.2) : 0);
  const y = lerp(a.y ?? 0, b.y ?? 0, u) + (drift ? 3.5 * Math.sin(t * 1.25 + 0.4) : 0);
  const z = lerp(a.z ?? 0, b.z ?? 0, u);
  let rx = lerp(a.rx ?? 0, b.rx ?? 0, u);
  let ry = lerp(a.ry ?? 0, b.ry ?? 0, u);
  if (drift) {
    ry += 0.35 * Math.sin(t * 0.7);
    rx += 0.25 * Math.sin(t * 0.55 + 2);
  }
  return {
    transformStyle: "preserve-3d",
    transform: `translate3d(${(-x).toFixed(2)}px, ${(-y).toFixed(2)}px, ${z.toFixed(2)}px) rotateX(${(-rx).toFixed(2)}deg) rotateY(${(-ry).toFixed(2)}deg)`,
  };
}

function p3Style(
  t: number,
  t0: number,
  dur: number,
  o: {
    x?: number;
    y?: number;
    z?: number;
    focus?: number;
    s?: number;
    op?: number;
    rx?: number;
    ry?: number;
    tilt?: boolean;
  } = {},
): React.CSSProperties {
  const u = eo(seg(t, t0, t0 + dur));
  const z = o.z ?? 0;
  const focus = o.focus ?? z;
  const d = Math.abs(z - focus);
  const blur = clamp((d - 90) / 110, 0, 3.5);
  const rx = o.rx !== undefined ? o.rx : o.tilt === false ? 0 : lerp(6, 1.2, u);
  const ry = o.ry !== undefined ? o.ry : o.tilt === false ? 0 : lerp(-8, -1.5, u);
  const y = (o.y ?? 0) + lerp(36, 0, u);
  const s = lerp(0.94, 1, u) * (o.s ?? 1);
  const op = u * (o.op ?? 1) * (1 - 0.12 * clamp(blur / 3.5, 0, 1));
  return {
    transformStyle: "preserve-3d",
    transform: `translate3d(${(o.x ?? 0).toFixed(2)}px, ${y.toFixed(2)}px, ${z.toFixed(2)}px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) scale(${s.toFixed(3)})`,
    filter: blur > 0.08 ? `blur(${blur.toFixed(2)}px)` : undefined,
    opacity: op,
  };
}

// ============================================================================
// MOTION GROUP + 180-DEGREE SHUTTER MOTION BLUR & HALFTONE IRIS WIPE
// ============================================================================
const PD = 0.28;
const FD = 0.34;
const W_SPAN = VW + 120;

function panStep(tau: number) {
  const u = seg(tau, 0, PD);
  const v =
    u > 0 && u < 1
      ? (W_SPAN * (u < 0.5 ? 12 * u * u : 12 * (1 - u) * (1 - u))) / PD / FPS
      : 0;
  return { x: W_SPAN * eio(u), v };
}

type TransKind = "none" | "pan" | "up" | "focus" | "iris" | "hold";

function calcTin(kind: TransKind, tau: number) {
  if (kind === "pan") {
    const p = panStep(tau);
    return { dx: W_SPAN - p.x, dy: 0, bx: mb(p.v), by: 0, op: 1, s: 1 };
  }
  if (kind === "up") {
    const p = panStep(tau);
    const k = VH / W_SPAN;
    return { dx: 0, dy: (W_SPAN - p.x) * k, bx: 0, by: mb(p.v * k), op: 1, s: 1 };
  }
  if (kind === "focus") {
    const e = eo(seg(tau, 0, FD));
    return { dx: 0, dy: 0, bx: 16 * (1 - e), by: 16 * (1 - e), op: e, s: lerp(1.035, 1, e) };
  }
  return { dx: 0, dy: 0, bx: 0, by: 0, op: 1, s: 1 };
}

function calcTout(kind: TransKind, tau: number) {
  if (tau <= 0) return { dx: 0, dy: 0, bx: 0, by: 0, op: 1, s: 1 };
  if (kind === "pan") {
    const p = panStep(tau);
    return { dx: -p.x, dy: 0, bx: mb(p.v), by: 0, op: 1, s: 1 };
  }
  if (kind === "up") {
    const p = panStep(tau);
    const k = VH / W_SPAN;
    return { dx: 0, dy: -p.x * k, bx: 0, by: mb(p.v * k), op: 1, s: 1 };
  }
  if (kind === "focus") {
    const e = eio(seg(tau, 0, FD));
    return { dx: 0, dy: 0, bx: 16 * e, by: 16 * e, op: 1 - e, s: lerp(1, 0.975, e) };
  }
  return { dx: 0, dy: 0, bx: 0, by: 0, op: 1, s: 1 };
}

const HalftoneIrisCanvas: React.FC<{
  radius: number;
  band?: number;
  opacity?: number;
  zIndex?: number;
}> = ({ radius, band = 130, opacity = 1, zIndex = 50 }) => {
  const ref = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);
    if (opacity <= 0) return;

    const p = 14;
    const a = 0.35;
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const Rm = radius + band;
    const n = Math.ceil(Rm / p) + 1;

    ctx.globalAlpha = opacity;
    ctx.fillStyle = C.gold;
    ctx.beginPath();
    for (let i = -n; i <= n; i++) {
      for (let j = -n; j <= n; j++) {
        const gx = i * p;
        const gy = j * p;
        const X = CX + gx * ca - gy * sa;
        const Y = CY + gx * sa + gy * ca;
        if (X < -12 || X > VW + 12 || Y < -12 || Y > VH + 12) continue;
        const d = Math.hypot(X - CX, Y - CY);
        if (d < radius || d > Rm) continue;
        const r = p * 0.46 * Math.pow(1 - (d - radius) / band, 0.8);
        if (r < 0.35) continue;
        ctx.moveTo(X + r, Y);
        ctx.arc(X, Y, r, 0, Math.PI * 2);
      }
    }
    ctx.fill();
    ctx.globalAlpha = 1;
  }, [radius, band, opacity]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, pointerEvents: "none", zIndex }}
    />
  );
};

const SceneWrap: React.FC<{
  id: string;
  frame: number;
  startFrame: number;
  endFrame: number;
  inKind?: TransKind;
  outKind?: TransKind;
  plate: string;
  zIndex: number;
  children: React.ReactNode;
}> = ({
  id,
  frame,
  startFrame,
  endFrame,
  inKind = "none",
  outKind = "none",
  plate,
  zIndex,
  children,
}) => {
  const tailFrames =
    outKind === "pan" || outKind === "up"
      ? Math.ceil(PD * FPS)
      : outKind === "focus"
        ? Math.ceil(FD * FPS)
        : outKind === "hold"
          ? Math.ceil(0.45 * FPS)
          : 0;
  if (frame < startFrame || frame >= endFrame + tailFrames) return null;

  const t = frame / FPS;
  const t0 = startFrame / FPS;
  const t1 = endFrame / FPS;

  const fi = calcTin(inKind, t - t0);
  const fo = calcTout(outKind, t - t1);

  const dx = fi.dx + fo.dx;
  const dy = fi.dy + fo.dy;
  const bx = Math.max(fi.bx, fo.bx);
  const by = Math.max(fi.by, fo.by);
  const op = fi.op * fo.op;
  const s = fi.s * fo.s;

  const irisDur = 0.42;
  const rMax = 1480;
  const irisR =
    inKind === "iris" ? lerp(0, rMax, eio(seg(t - t0, 0, irisDur))) : rMax;

  const filterId = `mb-${id}`;
  const hasBlur = bx + by > 0.08;

  return (
    <>
      <svg width="0" height="0" style={{ position: "absolute" }}>
        <defs>
          <filter
            id={filterId}
            x="-15%"
            y="-15%"
            width="130%"
            height="130%"
            colorInterpolationFilters="sRGB"
          >
            <feGaussianBlur stdDeviation={`${bx.toFixed(2)} ${by.toFixed(2)}`} />
          </filter>
        </defs>
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          width: VW,
          height: VH,
          backgroundColor: plate,
          overflow: "hidden",
          zIndex,
          transformOrigin: "960px 540px",
          transform: `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px) scale(${s.toFixed(4)})`,
          opacity: op,
          filter: hasBlur ? `url(#${filterId})` : undefined,
          clipPath:
            inKind === "iris" && irisR < rMax
              ? `circle(${irisR.toFixed(1)}px at ${CX}px ${CY}px)`
              : undefined,
        }}
      >
        {children}
      </div>
      {inKind === "iris" && irisR < rMax && (
        <HalftoneIrisCanvas
          radius={irisR}
          band={130}
          opacity={1 - seg(irisR, rMax * 0.82, rMax)}
          zIndex={zIndex + 2}
        />
      )}
    </>
  );
};

// ============================================================================
// CUSTOM VECTOR GEOMETRIC OWL CREST (No OS Emoji)
// ============================================================================
const HedwigOwlMark: React.FC<{ size?: number; glow?: number }> = ({
  size = 112,
  glow = 0.5,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 120 120"
    style={{
      filter: `drop-shadow(0 0 ${Math.round(16 + glow * 22)}px rgba(245, 184, 65, ${0.42 + glow * 0.25}))`,
    }}
  >
    <defs>
      <linearGradient id="owlGold" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stopColor={C.goldBright} />
        <stop offset="100%" stopColor={C.gold} />
      </linearGradient>
    </defs>
    {/* Outer minimal crest frame */}
    <path
      d="M60 10 L96 28 L104 68 L60 110 L16 68 L24 28 Z"
      fill="rgba(27, 30, 40, 0.72)"
      stroke="url(#owlGold)"
      strokeWidth="3"
      strokeLinejoin="round"
    />
    {/* Horned owl brow & facial disc */}
    <path
      d="M32 34 L46 44 L60 39 L74 44 L88 34 L84 62 C82 80 71 92 60 96 C49 92 38 80 36 62 Z"
      fill="none"
      stroke={C.ivory}
      strokeWidth="2.6"
      strokeLinejoin="round"
    />
    {/* Luminous amber eyes */}
    <circle cx="47" cy="56" r="7.5" fill="url(#owlGold)" />
    <circle cx="73" cy="56" r="7.5" fill="url(#owlGold)" />
    <circle cx="47" cy="56" r="3.2" fill={C.bg} />
    <circle cx="73" cy="56" r="3.2" fill={C.bg} />
    <circle cx="49" cy="54" r="1.4" fill="#ffffff" />
    <circle cx="75" cy="54" r="1.4" fill="#ffffff" />
    {/* Geometric beak */}
    <path d="M56 64 L64 64 L60 73 Z" fill="url(#owlGold)" />
  </svg>
);

// ============================================================================
// BEAT 1 (0..128): Shattering API Tags + "Everyone rents AI. We built one."
// ============================================================================
const Beat1: React.FC<{ frame: number; rms: number }> = ({ frame, rms }) => {
  const t = frame / FPS;
  const u1 = eo(seg(t, 0.35, 1.05));
  const u2 = eo(seg(t, 1.15, 1.95));
  const shatterU = eo(seg(t, 1.12, 3.6));

  const apiTags = [
    { txt: "$20 / mo", x: -480, y: -230, z: 80, vx: -520, vy: -310, vz: 320 },
    { txt: "API_KEY=sk-...", x: 470, y: -210, z: 110, vx: 540, vy: -290, vz: 360 },
    { txt: "RATE LIMIT 429", x: -440, y: 210, z: 60, vx: -480, vy: 280, vz: 280 },
    { txt: "CLOUD USAGE +$$$", x: 460, y: 220, z: 90, vx: 510, vy: 300, vz: 340 },
  ];

  return (
    <SceneWrap id="b1" frame={frame} startFrame={0} endFrame={128} outKind="hold" plate={C.bg} zIndex={1}>
      <ShatteringShards3DCanvas frame={frame} />
      <AbsoluteFill style={{ perspective: 1400, alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            position: "relative",
            width: VW,
            height: VH,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            ...evalCam3D(t, [
              { t: 0, x: 0, y: 0, z: 120, rx: 2, ry: -2 },
              { t: 1.2, x: 0, y: 0, z: 0, rx: 0, ry: 0 },
              { t: 4.2, x: 0, y: 0, z: -40, rx: -1, ry: 1 },
            ]),
          }}
        >
          {/* Floating billing/API tags that fly outward & blur away when "We built one" hits */}
          {apiTags.map((tag) => {
            const tx = tag.x + tag.vx * shatterU;
            const ty = tag.y + tag.vy * shatterU;
            const tz = tag.z + tag.vz * shatterU;
            const op = (1 - shatterU) * eo(seg(t, 0.1, 0.5)) * 0.85;
            return (
              <div
                key={tag.txt}
                style={{
                  position: "absolute",
                  padding: "10px 20px",
                  borderRadius: 999,
                  background: "rgba(251, 113, 133, 0.10)",
                  border: "1px solid rgba(251, 113, 133, 0.35)",
                  fontFamily: F.mono,
                  fontSize: 18,
                  color: C.rose,
                  opacity: op,
                  filter: `blur(${(shatterU * 6).toFixed(1)}px)`,
                  transform: `translate3d(${tx}px, ${ty}px, ${tz}px) rotate(${(shatterU * 14).toFixed(1)}deg)`,
                }}
              >
                {tag.txt}
              </div>
            );
          })}

          <div
            style={{
              textAlign: "center",
              fontFamily: F.serif,
              fontSize: 162,
              lineHeight: 0.94,
              letterSpacing: "-0.035em",
              color: C.ivory,
              transform: `scale(${1 + rms * 0.015})`,
            }}
          >
            <div
              style={{
                opacity: u1 * (1 - 0.55 * shatterU),
                transform: `translateY(${(1 - u1) * 32}px)`,
                color: shatterU > 0.2 ? C.muted : C.ivory,
              }}
            >
              Everyone rents AI.
            </div>
            <div
              style={{
                opacity: u2,
                transform: `translateY(${(1 - u2) * 38}px) scale(${lerp(0.92, 1, u2)})`,
                marginTop: 8,
              }}
            >
              We{" "}
              <span style={{ color: C.gold, fontStyle: "italic" }}>built one.</span>
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 2 (128..236): Halftone Iris + Parametric Ribbon + "From. Zero. Knobs." + Odometer
// ============================================================================
const Beat2: React.FC<{ frame: number; rms: number }> = ({ frame, rms }) => {
  const t = (frame - 128) / FPS;
  const odoU = eo(seg(frame, 134, 212));
  const knobs = Math.round(odoU * 286_663_680).toLocaleString("en-US");

  const words = [
    { text: "From.", f0: 132, f1: 168, col: C.ivory },
    { text: "Zero.", f0: 168, f1: 202, col: C.indigo },
    { text: "Knobs.", f0: 202, f1: 238, col: C.gold },
  ];

  return (
    <SceneWrap
      id="b2"
      frame={frame}
      startFrame={128}
      endFrame={236}
      inKind="iris"
      outKind="pan"
      plate={C.bg}
      zIndex={2}
    >
      <ParametricRibbon3DCanvas frame={frame} />
      {words.map((w) => {
        if (frame < w.f0 - 2 || frame > w.f1 + 2) return null;
        const win = eo(seg(frame, w.f0, w.f0 + 9));
        const wout = 1 - eo(seg(frame, w.f1 - 7, w.f1));
        const sc = 0.88 + 0.12 * bo(seg(frame, w.f0, w.f0 + 13)) + rms * 0.025;
        return (
          <AbsoluteFill
            key={w.text}
            style={{
              alignItems: "center",
              justifyContent: "center",
              paddingBottom: 60,
              opacity: Math.min(win, wout),
              transform: `scale(${sc})`,
            }}
          >
            <div
              style={{
                fontFamily: F.serif,
                fontSize: 204,
                letterSpacing: "-0.04em",
                color: w.col,
                textShadow: "0 20px 60px rgba(0,0,0,0.65)",
              }}
            >
              {w.text}
            </div>
          </AbsoluteFill>
        );
      })}

      {/* Live Monospace Parameter Odometer */}
      <div
        style={{
          position: "absolute",
          bottom: 185,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          opacity: eo(seg(t, 0.15, 0.45)),
        }}
      >
        <div
          style={{
            padding: "12px 28px",
            borderRadius: 999,
            background: "rgba(18, 20, 27, 0.84)",
            border: "1px solid rgba(245, 184, 65, 0.35)",
            fontFamily: F.mono,
            fontSize: 26,
            fontWeight: 600,
            color: C.goldBright,
            letterSpacing: "0.04em",
          }}
        >
          PARAMS: {knobs}
        </div>
      </div>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 3 (236..370): 3D Camera Rack Focus Past Architecture Glass Slabs
// ============================================================================
const Beat3: React.FC<{ frame: number }> = ({ frame }) => {
  const t = (frame - 236) / FPS;
  const focusZ = interpolate(frame, [246, 345], [160, -200], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });

  const slabs = [
    { name: "RoPE", sub: "Rotary Position Embeddings", z: 160, y: -155, col: C.indigo },
    { name: "RMSNorm", sub: "Zero-mean Pre-Norm Stability", z: 40, y: -50, col: C.ivory },
    { name: "SwiGLU", sub: "Gated Linear Projection MLP", z: -80, y: 55, col: C.emerald },
    { name: "Flash SDPA", sub: "20 Layers · 16 Heads · 16,384 BPE", z: -200, y: 160, col: C.gold },
  ];

  return (
    <SceneWrap
      id="b3"
      frame={frame}
      startFrame={236}
      endFrame={370}
      inKind="pan"
      outKind="up"
      plate={C.bg}
      zIndex={3}
    >
      <AbsoluteFill style={{ perspective: 1400 }}>
        <div
          style={{
            width: VW,
            height: VH,
            position: "relative",
            ...evalCam3D(t, [
              { t: 0, x: -70, y: -20, z: 220, rx: 3, ry: -10 },
              { t: 2.2, x: 40, y: 0, z: 40, rx: 1, ry: -4 },
              { t: 4.4, x: 110, y: 15, z: -120, rx: -2, ry: 3 },
            ]),
          }}
        >
          {/* Left Hero Editorial Copy */}
          <div style={{ position: "absolute", left: 130, top: 270, width: 720 }}>
            <div
              style={{
                fontFamily: F.mono,
                fontSize: 15,
                color: C.gold,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
              }}
            >
              tinygpt.py · Pure PyTorch
            </div>
            <div
              style={{
                fontFamily: F.serif,
                fontSize: 122,
                lineHeight: 0.94,
                letterSpacing: "-0.03em",
                color: C.ivory,
                marginTop: 18,
              }}
            >
              No shortcuts.
              <br />
              <span style={{ color: C.indigo, fontStyle: "italic" }}>Pure PyTorch.</span>
            </div>
            <div
              style={{
                fontFamily: F.sans,
                fontSize: 22,
                color: C.muted,
                marginTop: 22,
                lineHeight: 1.5,
              }}
            >
              No Hugging Face Trainer. No Lightning abstractions. Every matrix multiply stays visible.
            </div>
          </div>

          {/* Right Z-spaced 3D Glass Architecture Slabs with Rack Focus */}
          {slabs.map((sl, i) => (
            <div
              key={sl.name}
              style={{
                position: "absolute",
                left: 1020,
                top: 470,
                width: 620,
                padding: "22px 32px",
                borderRadius: 20,
                background: "rgba(27, 30, 40, 0.86)",
                border: `1px solid ${Math.abs(sl.z - focusZ) < 65 ? sl.col : C.border}`,
                boxShadow:
                  Math.abs(sl.z - focusZ) < 65
                    ? `0 24px 64px rgba(0,0,0,0.55), 0 0 28px ${sl.col}33`
                    : "0 18px 40px rgba(0,0,0,0.4)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                ...p3Style(t, 0.15 + i * 0.25, 0.55, {
                  x: (i - 1.5) * 32,
                  y: sl.y,
                  z: sl.z,
                  focus: focusZ,
                  rx: 4,
                  ry: -12,
                }),
              }}
            >
              <div>
                <div style={{ fontFamily: F.mono, fontSize: 28, fontWeight: 600, color: sl.col }}>
                  {sl.name}
                </div>
                <div style={{ fontFamily: F.sans, fontSize: 16, color: C.muted, marginTop: 4 }}>
                  {sl.sub}
                </div>
              </div>
              <div
                style={{
                  fontFamily: F.mono,
                  fontSize: 14,
                  color: C.ivory,
                  opacity: 0.7,
                }}
              >
                L{i + 1}
              </div>
            </div>
          ))}
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 4 (370..550): Full-Bleed Neon Loss Curve + Laser Scanner + 3D Terrain
// ============================================================================
const Beat4: React.FC<{ frame: number }> = ({ frame }) => {
  const curveU = eio(seg(frame, 382, 520));
  const curLoss = (8.47 - curveU * (8.47 - 1.795)).toFixed(3);
  const scanX = 120 + curveU * 1680;

  // Full-bleed curve coordinates across 1920x1080
  const pts = [
    [120, 390],
    [340, 560],
    [600, 650],
    [900, 710],
    [1220, 755],
    [1520, 785],
    [1800, 805],
  ];
  const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");
  const len = 1860;

  const chips = ["Once", " upon", " a", " time,", " a", " little", " owl", " named", " Pip."];

  return (
    <SceneWrap
      id="b4"
      frame={frame}
      startFrame={370}
      endFrame={550}
      inKind="up"
      outKind="iris"
      plate={C.bg}
      zIndex={4}
    >
      <TokenTerrain3DCanvas frame={frame} />

      {/* Full-bleed SVG Neon Curve + Vertical Laser Scanner */}
      <svg width={VW} height={VH} style={{ position: "absolute", inset: 0 }}>
        <defs>
          <linearGradient id="neonLoss" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={C.indigo} />
            <stop offset="55%" stopColor={C.emerald} />
            <stop offset="100%" stopColor={C.gold} />
          </linearGradient>
        </defs>
        <path
          d={pathD}
          fill="none"
          stroke="url(#neonLoss)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={len}
          strokeDashoffset={len * (1 - curveU)}
        />
        {/* Vertical Laser Scanner */}
        <line
          x1={scanX}
          y1="210"
          x2={scanX}
          y2="920"
          stroke="rgba(245, 184, 65, 0.55)"
          strokeWidth="2"
          strokeDasharray="6 6"
        />
      </svg>

      {/* Top Left Editorial Copy + Colossal Loss Counter */}
      <div style={{ position: "absolute", left: 120, top: 96, right: 120, display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontFamily: F.mono, fontSize: 15, color: C.indigo, letterSpacing: "0.14em" }}>
            STAGE 1 · 1.22 GB CORPUS · 16,384 BPE MERGES
          </div>
          <div style={{ fontFamily: F.serif, fontSize: 98, color: C.ivory, lineHeight: 0.96, marginTop: 10 }}>
            One billion characters.
            <br />
            <span style={{ color: C.gold, fontStyle: "italic" }}>Learned from scratch.</span>
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontFamily: F.mono, fontSize: 14, color: C.muted }}>VALIDATION LOSS</div>
          <div style={{ fontFamily: F.serif, fontSize: 162, lineHeight: 0.9, color: C.goldBright }}>
            {curLoss}
          </div>
        </div>
      </div>

      {/* Streaming BPE Token Chips above subtitle zone */}
      <div style={{ position: "absolute", left: 120, bottom: 118, display: "flex", gap: 12 }}>
        {chips.map((tk, i) => {
          const u = bo(seg(frame, 392 + i * 14, 410 + i * 14));
          return (
            <div
              key={i}
              style={{
                padding: "8px 16px",
                borderRadius: 12,
                background: "rgba(27, 30, 40, 0.88)",
                border: `1px solid ${i % 2 === 0 ? C.indigo : C.gold}`,
                fontFamily: F.mono,
                fontSize: 18,
                color: C.ivory,
                opacity: clamp(u, 0, 1),
                transform: `translateY(${(1 - u) * 18}px)`,
              }}
            >
              {tk}
            </div>
          );
        })}
      </div>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 5 (550..760): Warm Parchment Polarity Flip — 3D Preference Duel & 98.3%
// ============================================================================
const Beat5: React.FC<{ frame: number }> = ({ frame }) => {
  const t = (frame - 550) / FPS;
  const winU = eio(seg(frame, 588, 690));
  const winPct = (winU * 98.3).toFixed(1);
  const duelU = eio(seg(frame, 585, 650));

  return (
    <SceneWrap
      id="b5"
      frame={frame}
      startFrame={550}
      endFrame={760}
      inKind="iris"
      outKind="up"
      plate={C.parchment}
      zIndex={5}
    >
      <AbsoluteFill style={{ perspective: 1400 }}>
        <div
          style={{
            width: VW,
            height: VH,
            position: "relative",
            ...evalCam3D(t, [
              { t: 0, x: -30, y: 0, z: 60, rx: 1, ry: -3 },
              { t: 3.5, x: 20, y: 0, z: 0, rx: 0, ry: 1 },
            ]),
          }}
        >
          {/* Left: Monumental 98.3% Numeral */}
          <div style={{ position: "absolute", left: 130, top: 155, width: 700 }}>
            <div style={{ fontFamily: F.mono, fontSize: 15, color: C.indigoDeep, fontWeight: 600, letterSpacing: "0.12em" }}>
              STAGE 2 SFT (-100 MASK) · STAGE 3 DPO
            </div>
            <div style={{ fontFamily: F.serif, fontSize: 96, lineHeight: 0.96, color: C.ink, marginTop: 16 }}>
              Not just fluent.
              <br />
              <span style={{ color: C.indigoDeep, fontStyle: "italic" }}>Aligned.</span>
            </div>

            <div style={{ marginTop: 46 }}>
              <div style={{ fontFamily: F.serif, fontSize: 196, lineHeight: 0.88, color: C.ink }}>
                {winPct}
                <span style={{ fontSize: 92, color: "#059669" }}>%</span>
              </div>
              <div
                style={{
                  height: 6,
                  width: `${winU * 480}px`,
                  borderRadius: 999,
                  background: "linear-gradient(90deg, #4f46e5, #059669)",
                  marginTop: 14,
                }}
              />
              <div style={{ fontFamily: F.mono, fontSize: 16, color: C.mutedLight, marginTop: 14 }}>
                DPO PREFERENCE WIN RATE · 52,798 CHOSEN VS REJECTED PAIRS
              </div>
            </div>
          </div>

          {/* Right: 3D Z-Space Preference Duel — Rejected Tilts & Blurs Away, Chosen Locks Forward */}
          <div
            style={{
              position: "absolute",
              left: 960,
              top: 230,
              width: 780,
              padding: "32px 36px",
              borderRadius: 24,
              background: "#fff1f2",
              border: "1px solid #fecdd3",
              transform: `translate3d(${(-duelU * 80).toFixed(1)}px, ${(180 + duelU * 60).toFixed(1)}px, ${(-80 - duelU * 240).toFixed(1)}px) rotateY(${(-duelU * 14).toFixed(1)}deg) rotateZ(${(-duelU * 4).toFixed(1)}deg)`,
              filter: `blur(${(duelU * 3.2).toFixed(2)}px)`,
              opacity: 1 - duelU * 0.45,
            }}
          >
            <div style={{ fontFamily: F.mono, fontSize: 15, color: "#be123c", fontWeight: 600 }}>
              👎 REJECTED DRAFT
            </div>
            <div style={{ fontFamily: F.sans, fontSize: 22, color: "#881337", marginTop: 10, lineHeight: 1.45 }}>
              “Pip went to find the key. He looked and looked. Then Max said thank you.”
            </div>
          </div>

          <div
            style={{
              position: "absolute",
              left: 960,
              top: 240,
              width: 780,
              padding: "38px 42px",
              borderRadius: 24,
              background: "#ffffff",
              border: "2px solid #10b981",
              boxShadow: "0 30px 80px rgba(31, 36, 48, 0.14)",
              transform: `translate3d(0px, ${lerp(90, 0, duelU).toFixed(1)}px, ${(duelU * 70).toFixed(1)}px) scale(${lerp(0.94, 1.02, duelU).toFixed(3)})`,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontFamily: F.mono, fontSize: 15, color: "#059669", fontWeight: 600 }}>
                👍 CHOSEN BY DPO · 286.7M FLAGSHIP
              </div>
              <div style={{ fontFamily: F.mono, fontSize: 14, color: "#059669" }}>
                ANCHORED LOSS 1.200
              </div>
            </div>
            <div style={{ fontFamily: F.serif, fontSize: 38, color: C.ink, marginTop: 16, lineHeight: 1.28 }}>
              “Then, they saw a big, shiny key under a bush. Pip and Lily brought it home—and everyone played happily ever after.”
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 6 (760..972): Low-Angle 3D Rising Scale Monoliths -> $2.45 Total Cloud
// ============================================================================
const Beat6: React.FC<{ frame: number }> = ({ frame }) => {
  const t = (frame - 760) / FPS;
  const monoliths = [
    { size: "14M", tag: "Mac M4 · Local", stat: "Val 1.795 · 4k BPE", cost: "$0.00", h: 210, x: -460, z: -80, col: C.indigo, f0: 778 },
    { size: "97.6M", tag: "NVIDIA A40 · 66m", stat: "95.0% Win · 8k BPE", cost: "$0.65", h: 320, x: 0, z: 0, col: C.emerald, f0: 818 },
    { size: "286.7M", tag: "RTX 4090 · 115m", stat: "98.3% Win · 16k BPE", cost: "$1.70", h: 430, x: 460, z: 80, col: C.gold, f0: 856 },
  ];
  const punchU = bo(seg(frame, 898, 924));

  return (
    <SceneWrap
      id="b6"
      frame={frame}
      startFrame={760}
      endFrame={972}
      inKind="up"
      outKind="focus"
      plate={C.bg}
      zIndex={6}
    >
      {/* Volumetric Golden God-Rays from Top Right */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse 70% 60% at 75% 15%, rgba(245, 184, 65, 0.22), rgba(18, 20, 27, 0))",
        }}
      />
      <AbsoluteFill style={{ perspective: 1400, alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            position: "relative",
            width: VW,
            height: VH,
            ...evalCam3D(t, [
              { t: 0, x: -120, y: 40, z: 120, rx: -8, ry: 10 },
              { t: 4.5, x: 80, y: -15, z: -20, rx: -4, ry: -6 },
              { t: 7.0, x: 0, y: -30, z: 40, rx: -2, ry: 0 },
            ]),
          }}
        >
          <div style={{ position: "absolute", left: 130, top: 88 }}>
            <div style={{ fontFamily: F.serif, fontSize: 82, color: C.ivory, lineHeight: 0.96 }}>
              Three generations.{" "}
              <span style={{ color: C.gold, fontStyle: "italic" }}>Scaled 20×.</span>
            </div>
          </div>

          {monoliths.map((m) => {
            const u = eo(seg(frame, m.f0, m.f0 + 34));
            const curH = 80 + m.h * u;
            return (
              <div
                key={m.size}
                style={{
                  position: "absolute",
                  left: CX + m.x - 185,
                  bottom: 205,
                  width: 370,
                  height: curH,
                  borderRadius: 22,
                  background: `linear-gradient(180deg, rgba(27,30,40,0.95), rgba(18,20,27,0.98))`,
                  border: `1.5px solid ${m.col}`,
                  boxShadow:
                    m.col === C.gold
                      ? "0 30px 90px rgba(245, 184, 65, 0.28)"
                      : "0 24px 60px rgba(0,0,0,0.45)",
                  padding: "28px 30px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  transform: `translate3d(0, 0, ${m.z}px)`,
                  opacity: clamp(u * 2.2, 0, 1),
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 15, color: m.col }}>
                  <span>{m.tag}</span>
                  <span>{m.cost}</span>
                </div>
                <div>
                  <div style={{ fontFamily: F.serif, fontSize: 78, color: m.col, lineHeight: 0.9 }}>
                    {m.size}
                  </div>
                  <div style={{ fontFamily: F.mono, fontSize: 14, color: C.muted, marginTop: 8 }}>
                    {m.stat}
                  </div>
                </div>
              </div>
            );
          })}

          {/* Total Cloud Cost Punchline */}
          <div
            style={{
              position: "absolute",
              bottom: 105,
              left: 0,
              right: 0,
              display: "flex",
              justifyContent: "center",
              opacity: clamp(punchU, 0, 1),
              transform: `scale(${lerp(0.88, 1, punchU)})`,
            }}
          >
            <div
              style={{
                padding: "12px 30px",
                borderRadius: 999,
                background: "rgba(245, 184, 65, 0.14)",
                border: "1px solid rgba(245, 184, 65, 0.55)",
                fontFamily: F.mono,
                fontSize: 22,
                fontWeight: 600,
                color: C.goldBright,
              }}
            >
              TOTAL CLOUD COMPUTE: ~$2.45 ON RUNPOD
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 7 (972..1208): Angled 3D Studio Window Orbiting Straight-On + Live Brain X-Ray
// ============================================================================
const Beat7: React.FC<{ frame: number }> = ({ frame }) => {
  const t = (frame - 972) / FPS;
  const orbitU = eio(seg(frame, 976, 1085));
  const rx = lerp(7, 0, orbitU);
  const ry = lerp(-18, 0, orbitU);
  const z = lerp(-240, 40, orbitU);

  const tokens = [
    "Once", " upon", " a", " time,", " there", " was", " an", " owl", " named", " Pip.",
    " He", " lived", " with", " Lily.", " One", " day...",
  ];
  // Synced exactly to 12-frame tick cadence in generate-audio.py (996, 1008, ...)
  const visibleCount = clamp(Math.floor((frame - 984) / 12), 1, tokens.length);

  const xrays = [
    [{ t: "Pip.", p: 72.4 }, { t: "Barnaby.", p: 11.2 }, { t: "Oliver.", p: 6.8 }, { t: "Hoot.", p: 5.1 }, { t: "Max.", p: 2.1 }],
    [{ t: "owl", p: 84.1 }, { t: "bird", p: 7.4 }, { t: "fox", p: 3.9 }, { t: "mouse", p: 2.6 }, { t: "girl", p: 1.2 }],
    [{ t: "Lily.", p: 79.3 }, { t: "Mom.", p: 9.8 }, { t: "Pip.", p: 5.2 }, { t: "Ben.", p: 3.4 }, { t: "Sam.", p: 1.6 }],
  ];
  const activeXray = xrays[Math.floor(visibleCount / 5) % xrays.length];

  return (
    <SceneWrap
      id="b7"
      frame={frame}
      startFrame={972}
      endFrame={1208}
      inKind="focus"
      outKind="pan"
      plate={C.bg}
      zIndex={7}
    >
      <AbsoluteFill style={{ perspective: 1500, alignItems: "center", justifyContent: "center" }}>
        <div
          style={{
            width: 1560,
            height: 760,
            borderRadius: 26,
            background: C.card,
            border: `1px solid ${C.border}`,
            boxShadow: "0 45px 120px rgba(0,0,0,0.68), 0 0 60px rgba(129, 140, 248, 0.14)",
            overflow: "hidden",
            transform: `translate3d(0px, -24px, ${z.toFixed(1)}px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`,
          }}
        >
          {/* Studio Top Chrome */}
          <div
            style={{
              height: 54,
              background: "#161922",
              borderBottom: `1px solid ${C.border}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0 26px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <HedwigOwlMark size={30} glow={0.3} />
              <span style={{ fontFamily: F.sans, fontSize: 16, fontWeight: 700, color: C.ivory }}>
                Hedwig Offline Studio · 286.7M Flagship
              </span>
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 13, color: C.emerald }}>
              ● 127.0.0.1:8000 · MPS KV-CACHE
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1.18fr 0.82fr", height: 706 }}>
            {/* Streaming Output */}
            <div style={{ padding: "40px 44px", borderRight: `1px solid ${C.border}` }}>
              <div style={{ fontFamily: F.mono, fontSize: 13, color: C.indigo }}>
                PROMPT → “Write a story about Pip and Lily.”
              </div>
              <div style={{ marginTop: 28, fontFamily: F.serif, fontSize: 52, lineHeight: 1.35, color: C.ivory }}>
                {tokens.slice(0, visibleCount).map((tok, i) => {
                  const active = i === visibleCount - 1;
                  return (
                    <span
                      key={i}
                      style={{
                        background: active ? "rgba(245, 184, 65, 0.25)" : "transparent",
                        color: active ? C.goldBright : C.ivory,
                        borderBottom: active ? `3px solid ${C.gold}` : "none",
                        borderRadius: 6,
                        padding: active ? "0 6px" : "0",
                      }}
                    >
                      {tok}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Live Brain X-Ray Top-5 Probabilities */}
            <div style={{ padding: "40px 42px", background: "#161922" }}>
              <div style={{ fontFamily: F.mono, fontSize: 14, color: C.gold, letterSpacing: "0.08em" }}>
                🔬 BRAIN X-RAY · LIVE SOFTMAX
              </div>
              <div style={{ fontFamily: F.sans, fontSize: 24, fontWeight: 700, color: C.ivory, marginTop: 6 }}>
                Top-5 Candidate Tokens
              </div>
              <div style={{ marginTop: 30, display: "flex", flexDirection: "column", gap: 22 }}>
                {activeXray.map((cand, idx) => (
                  <div key={cand.t}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 17 }}>
                      <span style={{ color: idx === 0 ? C.goldBright : C.ivory, fontWeight: idx === 0 ? 700 : 400 }}>
                        {cand.t}
                      </span>
                      <span style={{ color: idx === 0 ? C.emerald : C.muted }}>{cand.p.toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 12, borderRadius: 999, background: C.bg, overflow: "hidden", marginTop: 8 }}>
                      <div
                        style={{
                          width: `${cand.p}%`,
                          height: "100%",
                          borderRadius: 999,
                          background:
                            idx === 0
                              ? `linear-gradient(90deg, ${C.gold}, ${C.emerald})`
                              : C.indigo,
                          boxShadow: idx === 0 ? `0 0 18px ${C.gold}66` : undefined,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 8 (1208..1410): LightSpeedWarp + Physical 4-Bit Crush + 832 tok/s
// ============================================================================
const Beat8: React.FC<{ frame: number }> = ({ frame }) => {
  const crushU = eio(seg(frame, 1234, 1300));
  const mb = (574.7 - crushU * (574.7 - 161.9)).toFixed(1);
  const tokSec = Math.round(eo(seg(frame, 1230, 1320)) * 832);
  const blockH = lerp(380, 118, crushU);

  return (
    <SceneWrap
      id="b8"
      frame={frame}
      startFrame={1208}
      endFrame={1410}
      inKind="pan"
      outKind="up"
      plate={C.bg}
      zIndex={8}
    >
      <LightSpeedWarpCanvas frame={frame} />
      <AbsoluteFill style={{ padding: "88px 140px", justifyContent: "center" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 90, alignItems: "center" }}>
          {/* Left: Physical Weight Block Hydraulic Compression */}
          <div>
            <div style={{ fontFamily: F.mono, fontSize: 15, color: C.emerald, letterSpacing: "0.14em" }}>
              QUANTIZE.PY · PACKED 4-BIT
            </div>
            <div style={{ height: 410, display: "flex", alignItems: "flex-end", marginTop: 24 }}>
              <div
                style={{
                  width: 520,
                  height: blockH,
                  borderRadius: 22,
                  background: `linear-gradient(180deg, rgba(52, 211, 153, 0.25), rgba(129, 140, 248, 0.16))`,
                  border: `2px solid ${crushU > 0.7 ? C.emerald : C.indigo}`,
                  boxShadow: `0 0 60px rgba(52, 211, 153, ${0.15 + crushU * 0.25})`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0 34px",
                }}
              >
                <div>
                  <div style={{ fontFamily: F.serif, fontSize: 66, color: C.ivory, lineHeight: 1 }}>
                    {mb} <span style={{ fontSize: 34, color: C.emerald }}>MB</span>
                  </div>
                  <div style={{ fontFamily: F.mono, fontSize: 14, color: C.muted, marginTop: 4 }}>
                    286.7M · 574.7 MB → 161.9 MB (14M → 7.9 MB)
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: Monumental 832 tok/s MLX Speed Readout */}
          <div>
            <div style={{ fontFamily: F.mono, fontSize: 15, color: C.gold, letterSpacing: "0.14em" }}>
              APPLE SILICON M4 · MLX_BENCH.PY
            </div>
            <div style={{ fontFamily: F.serif, fontSize: 210, lineHeight: 0.85, color: C.goldBright, marginTop: 18 }}>
              {tokSec}
            </div>
            <div style={{ fontFamily: F.serif, fontSize: 68, color: C.ivory }}>
              tokens per second.
            </div>
            <div style={{ fontFamily: F.mono, fontSize: 16, color: C.emerald, marginTop: 20 }}>
              + SSD Flash Layer Streaming: 52.9 MB → 18.8 MB RAM
            </div>
          </div>
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 9 (1410..1606): HorizonGrid3D + Vertical Slot Reel -> "Zero cloud."
// ============================================================================
const Beat9: React.FC<{ frame: number }> = ({ frame }) => {
  const slotItems = ["No API keys.", "No subscriptions.", "No cloud servers.", "Zero cloud."];
  const slotU = eio(seg(frame, 1418, 1500));
  const rowH = 148;
  const offset = -slotU * (slotItems.length - 1) * rowH;

  const platforms = [
    { name: "Mac Offline", sub: "app.py · 9 brains · M4 MPS", col: C.indigo, f0: 1445 },
    { name: "iPhone Core ML", sub: "27.7 MB · 30/30 parity", col: C.gold, f0: 1470 },
    { name: "Static Browser", sub: "hedwig.js · HF Space", col: C.emerald, f0: 1495 },
  ];

  return (
    <SceneWrap
      id="b9"
      frame={frame}
      startFrame={1410}
      endFrame={1606}
      inKind="up"
      outKind="focus"
      plate={C.bg}
      zIndex={9}
    >
      <HorizonGrid3DCanvas frame={frame} />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", paddingBottom: 70 }}>
        {/* High-speed Slot Reel */}
        <div
          style={{
            height: rowH,
            overflow: "hidden",
            width: 1200,
            textAlign: "center",
          }}
        >
          <div style={{ transform: `translateY(${offset}px)` }}>
            {slotItems.map((item, i) => (
              <div
                key={item}
                style={{
                  height: rowH,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: F.serif,
                  fontSize: i === 3 ? 156 : 126,
                  letterSpacing: "-0.03em",
                  color: i === 3 ? C.gold : C.ivory,
                }}
              >
                {item}
              </div>
            ))}
          </div>
        </div>

        {/* 3 Floating Platform Pills */}
        <div style={{ display: "flex", gap: 28, marginTop: 64 }}>
          {platforms.map((p, idx) => {
            const u = bo(seg(frame, p.f0, p.f0 + 22));
            return (
              <div
                key={p.name}
                style={{
                  width: 390,
                  padding: "26px 30px",
                  borderRadius: 20,
                  background: "rgba(27, 30, 40, 0.88)",
                  border: `1px solid ${p.col}`,
                  boxShadow: "0 24px 60px rgba(0,0,0,0.45)",
                  opacity: clamp(u, 0, 1),
                  transform: `translateY(${(1 - u) * 32}px) rotate(${(idx - 1) * 1.5}deg)`,
                }}
              >
                <div style={{ fontFamily: F.serif, fontSize: 44, color: p.col }}>{p.name}</div>
                <div style={{ fontFamily: F.mono, fontSize: 15, color: C.ivory, marginTop: 8 }}>
                  {p.sub}
                </div>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// BEAT 10 (1606..1800): GoldenVortex + Custom SVG Owl Crest + Hedwig Lockup
// ============================================================================
const Beat10: React.FC<{ frame: number; rms: number }> = ({ frame, rms }) => {
  const logoU = eo(seg(frame, 1612, 1656));
  const subU = eo(seg(frame, 1642, 1684));

  return (
    <SceneWrap
      id="b10"
      frame={frame}
      startFrame={1606}
      endFrame={1800}
      inKind="focus"
      plate={C.bg}
      zIndex={10}
    >
      <GoldenVortexCanvas frame={frame} />
      <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
        <div style={{ opacity: logoU, transform: `scale(${0.86 + 0.14 * logoU})`, marginBottom: 14 }}>
          <HedwigOwlMark size={118} glow={rms} />
        </div>
        <div
          style={{
            fontFamily: F.serif,
            fontSize: 176,
            lineHeight: 0.9,
            letterSpacing: "-0.035em",
            color: C.ivory,
            opacity: logoU,
            transform: `translateY(${(1 - logoU) * 18}px)`,
          }}
        >
          Hedwig
        </div>
        <div
          style={{
            fontFamily: F.sans,
            fontSize: 28,
            fontWeight: 500,
            color: C.goldBright,
            marginTop: 18,
            opacity: subU,
            transform: `translateY(${(1 - subU) * 12}px)`,
          }}
        >
          Open weights. Open code. Yours forever.
        </div>
        <div style={{ display: "flex", gap: 22, marginTop: 34, opacity: subU }}>
          {[
            "github.com/ansarbuilds/hedwig",
            "huggingface.co/spaces/ansarbuilds/hedwig-demo",
          ].map((url) => (
            <div
              key={url}
              style={{
                padding: "11px 22px",
                borderRadius: 999,
                background: "rgba(27, 30, 40, 0.88)",
                border: `1px solid ${C.border}`,
                fontFamily: F.mono,
                fontSize: 15,
                color: C.ivory,
              }}
            >
              {url}
            </div>
          ))}
        </div>
      </AbsoluteFill>
    </SceneWrap>
  );
};

// ============================================================================
// MAIN COMPOSITION
// ============================================================================
export const HedwigDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const rms = AUDIO_RMS[frame] ?? 0.25;
  const _peak = AUDIO_PEAK[frame] ?? 0.3;

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg, overflow: "hidden" }}>
      <Audio src={staticFile("demo-soundtrack.wav")} />

      <Beat1 frame={frame} rms={rms} />
      <Beat2 frame={frame} rms={rms} />
      <Beat3 frame={frame} />
      <Beat4 frame={frame} />
      <Beat5 frame={frame} />
      <Beat6 frame={frame} />
      <Beat7 frame={frame} />
      <Beat8 frame={frame} />
      <Beat9 frame={frame} />
      <Beat10 frame={frame} rms={rms} />

      {/* Cinema vignette */}
      <AbsoluteFill
        style={{
          pointerEvents: "none",
          zIndex: 60,
          background:
            "radial-gradient(ellipse 85% 80% at 50% 50%, rgba(0,0,0,0) 62%, rgba(0,0,0,0.36) 100%)",
        }}
      />

      <FilmSubtitles frame={frame} />
    </AbsoluteFill>
  );
};
