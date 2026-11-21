import React, { useEffect, useMemo, useRef } from "react";
import { AUDIO_PEAK, AUDIO_RMS } from "./audioData";

const VW = 1920;
const VH = 1080;
const CX = VW / 2;
const CY = VH / 2;

const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));

function project3D(
  x: number,
  y: number,
  z: number,
  rx: number,
  ry: number,
  rz: number,
  cx = CX,
  cy = CY,
  fov = 1150,
) {
  const cosX = Math.cos(rx);
  const sinX = Math.sin(rx);
  const y1 = y * cosX - z * sinX;
  const z1 = y * sinX + z * cosX;

  const cosY = Math.cos(ry);
  const sinY = Math.sin(ry);
  const x2 = x * cosY + z1 * sinY;
  const z2 = -x * sinY + z1 * cosY;

  const cosZ = Math.cos(rz);
  const sinZ = Math.sin(rz);
  const x3 = x2 * cosZ - y1 * sinZ;
  const y3 = x2 * sinZ + y1 * cosZ;

  const scale = fov / Math.max(120, fov - z2);
  return {
    sx: cx + x3 * scale,
    sy: cy + y3 * scale,
    z: z2,
    scale,
  };
}

// ============================================================================
// 1. SHATTERING SHARDS 3D — Translucent API/glass shards exploding outward (Beat 1)
// ============================================================================
export const ShatteringShards3DCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  const shards = useMemo(() => {
    let sd = 4242;
    const rn = () => {
      sd = (sd * 1664525 + 1013904223) >>> 0;
      return sd / 4294967296;
    };
    return Array.from({ length: 48 }, (_, i) => {
      const ang = (i / 48) * Math.PI * 2 + (rn() - 0.5) * 0.4;
      const dist = 60 + rn() * 220;
      const spd = 220 + rn() * 680;
      return {
        x0: Math.cos(ang) * dist,
        y0: Math.sin(ang) * dist * 0.6,
        z0: (rn() - 0.5) * 260,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd * 0.72,
        vz: 140 + rn() * 520,
        size: 26 + rn() * 54,
        rotX: rn() * Math.PI,
        rotY: rn() * Math.PI,
        rotZ: rn() * Math.PI,
        vRot: (rn() - 0.5) * 3.8,
        gold: i % 4 === 0,
      };
    });
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.25;
    const pk = AUDIO_PEAK[frame] ?? 0.35;

    // Central ignition shockwave bloom
    const burstT = Math.max(0, t - 1.1);
    const bloom = ctx.createRadialGradient(CX, CY, 20, CX, CY, 880);
    bloom.addColorStop(0, `rgba(245, 184, 65, ${0.22 + pk * 0.22})`);
    bloom.addColorStop(0.38, `rgba(129, 140, 248, ${0.20 + rms * 0.14})`);
    bloom.addColorStop(1, "rgba(18, 20, 28, 0)");
    ctx.fillStyle = bloom;
    ctx.fillRect(0, 0, VW, VH);

    // Depth-sorted 3D triangular crystal shards
    const rendered = shards.map((s) => {
      const exp = Math.pow(burstT, 0.88);
      const px = s.x0 + s.vx * exp;
      const py = s.y0 + s.vy * exp;
      const pz = s.z0 + s.vz * exp;
      const rx = s.rotX + t * s.vRot;
      const ry = s.rotY + t * s.vRot * 0.8;
      const rz = s.rotZ + t * s.vRot * 0.6;

      const pts = [
        project3D(px - s.size, py - s.size * 0.6, pz, rx * 0.08, ry * 0.08, 0),
        project3D(px + s.size * 0.9, py - s.size * 0.3, pz + 24, rx * 0.08, ry * 0.08, 0),
        project3D(px + s.size * 0.2, py + s.size * 0.95, pz - 18, rx * 0.08, ry * 0.08, 0),
      ];
      return { pts, z: pz, gold: s.gold, spec: 0.5 + 0.5 * Math.sin(rx + ry) };
    });

    rendered.sort((a, b) => a.z - b.z);
    for (const sh of rendered) {
      const fade = clamp(1 - Math.max(0, sh.z - 620) / 360, 0.1, 0.92);
      ctx.beginPath();
      ctx.moveTo(sh.pts[0].sx, sh.pts[0].sy);
      ctx.lineTo(sh.pts[1].sx, sh.pts[1].sy);
      ctx.lineTo(sh.pts[2].sx, sh.pts[2].sy);
      ctx.closePath();
      ctx.fillStyle = sh.gold
        ? `rgba(245, 184, 65, ${(0.12 + sh.spec * 0.22) * fade})`
        : `rgba(129, 140, 248, ${(0.10 + sh.spec * 0.20) * fade})`;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = sh.gold
        ? `rgba(255, 224, 150, ${(0.35 + sh.spec * 0.45) * fade})`
        : `rgba(199, 210, 254, ${(0.28 + sh.spec * 0.4) * fade})`;
      ctx.stroke();
    }
  }, [frame, shards]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};

// ============================================================================
// 2. PARAMETRIC RIBBON 3D — 32-strand volumetric indigo-to-gold silk light ribbon
// ============================================================================
export const ParametricRibbon3DCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.3;
    const pk = AUDIO_PEAK[frame] ?? 0.35;

    const glow = ctx.createRadialGradient(CX, CY, 30, CX, CY, 820);
    glow.addColorStop(0, `rgba(129, 140, 248, ${0.22 + rms * 0.16})`);
    glow.addColorStop(0.5, `rgba(245, 184, 65, ${0.12 + pk * 0.08})`);
    glow.addColorStop(1, "rgba(18, 20, 28, 0)");
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, VW, VH);

    const strands = 32;
    const steps = 82;
    for (let s = 0; s < strands; s++) {
      const frac = s / (strands - 1);
      const offset = (frac - 0.5) * 320;
      ctx.beginPath();
      for (let i = 0; i <= steps; i++) {
        const u = i / steps;
        const x = (u - 0.5) * 2040;
        const wave = Math.sin(u * 5.6 - t * 3.1 + frac * 2.8) * (135 + rms * 95);
        const twist = Math.cos(u * 3.9 + t * 2.1 + frac * 2.1) * 165;
        const y = wave + offset * Math.cos(u * 2.8 + t * 1.4);
        const z = twist + offset * Math.sin(u * 2.8 + t * 1.4);
        const p = project3D(x, y, z, 0.28, Math.sin(t * 0.5) * 0.16, -0.11);
        if (i === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      const core = Math.abs(frac - 0.5) < 0.18;
      const isGold = core || s >= strands - 5;
      const alpha = core ? 0.58 + rms * 0.32 : 0.22 + rms * 0.18;
      ctx.strokeStyle = isGold
        ? `rgba(245, 184, 65, ${alpha})`
        : `rgba(129, 140, 248, ${alpha})`;
      ctx.lineWidth = core ? 2.8 : 1.35;
      ctx.stroke();
    }
  }, [frame]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};

// ============================================================================
// 3. TOKEN TERRAIN 3D — Undulating 3D wireframe terrain rushing beneath loss curve
// ============================================================================
export const TokenTerrain3DCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.3;

    const cols = 34;
    const rows = 20;
    const speed = t * 5.2;

    for (let r = 0; r < rows; r++) {
      const zFrac = r / (rows - 1);
      const z = -420 + zFrac * 1050;
      const alpha = clamp((1 - zFrac * 0.72) * (0.28 + rms * 0.35), 0.05, 0.75);
      ctx.beginPath();
      for (let c = 0; c < cols; c++) {
        const x = (c - (cols - 1) / 2) * 85;
        const h =
          Math.sin(c * 0.42 + speed * 0.6 + r * 0.3) * (36 + rms * 42) +
          Math.cos(c * 0.22 - speed * 0.9 + r * 0.5) * 28;
        const p = project3D(x, 265 + h, z, 0.24, 0, 0, CX, CY + 120, 920);
        if (c === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.strokeStyle =
        r % 4 === 0
          ? `rgba(245, 184, 65, ${alpha * 0.85})`
          : `rgba(129, 140, 248, ${alpha})`;
      ctx.lineWidth = r % 4 === 0 ? 1.9 : 1.25;
      ctx.stroke();
    }
  }, [frame]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};

// ============================================================================
// 4. LIGHT-SPEED WARP — 3D hyperspace starfield/token streaks accelerating radially
// ============================================================================
export const LightSpeedWarpCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  const stars = useMemo(() => {
    let sd = 918273;
    const rn = () => {
      sd = (sd * 1664525 + 1013904223) >>> 0;
      return sd / 4294967296;
    };
    return Array.from({ length: 180 }, (_, i) => {
      const ang = rn() * Math.PI * 2;
      const rad = 110 + rn() * 980;
      return {
        x: Math.cos(ang) * rad,
        y: Math.sin(ang) * rad * 0.62,
        zOff: rn(),
        spd: 0.75 + rn() * 1.45,
        gold: i % 5 === 0,
        emerald: i % 3 === 0,
      };
    });
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.3;

    const core = ctx.createRadialGradient(CX, CY, 10, CX, CY, 700);
    core.addColorStop(0, `rgba(52, 211, 153, ${0.18 + rms * 0.16})`);
    core.addColorStop(0.5, `rgba(129, 140, 248, ${0.14 + rms * 0.08})`);
    core.addColorStop(1, "rgba(20, 22, 28, 0)");
    ctx.fillStyle = core;
    ctx.fillRect(0, 0, VW, VH);

    for (const s of stars) {
      const zProg = (s.zOff + t * s.spd * (0.9 + rms * 0.5)) % 1;
      const zHead = -700 + zProg * 1650;
      const zTail = zHead - (150 + rms * 180);
      const p0 = project3D(s.x, s.y, zTail, 0, 0, 0, CX, CY, 980);
      const p1 = project3D(s.x, s.y, zHead, 0, 0, 0, CX, CY, 980);

      const alpha = clamp(zProg * 1.2, 0.05, 0.92);
      ctx.strokeStyle = s.gold
        ? `rgba(245, 184, 65, ${alpha})`
        : s.emerald
          ? `rgba(52, 211, 153, ${alpha})`
          : `rgba(199, 210, 254, ${alpha * 0.85})`;
      ctx.lineWidth = 1.2 + zProg * 2.4;
      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.stroke();
    }
  }, [frame, stars]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};

// ============================================================================
// 5. HORIZON GRID 3D — Deep perspective floor/horizon with volumetric bloom
// ============================================================================
export const HorizonGrid3DCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.3;

    const horizonY = 590;
    const bloom = ctx.createRadialGradient(CX, horizonY, 20, CX, horizonY, 960);
    bloom.addColorStop(0, `rgba(245, 184, 65, ${0.22 + rms * 0.16})`);
    bloom.addColorStop(0.45, `rgba(129, 140, 248, ${0.18 + rms * 0.1})`);
    bloom.addColorStop(1, "rgba(20, 22, 28, 0)");
    ctx.fillStyle = bloom;
    ctx.fillRect(0, 0, VW, VH);

    const cols = 18;
    const rows = 14;
    const scroll = (t * 2.6) % 1;

    ctx.lineWidth = 1.3;
    for (let c = -cols; c <= cols; c++) {
      const xWorld = c * 150;
      const p0 = project3D(xWorld, 240, -500, 0.12, 0, 0, CX, horizonY, 850);
      const p1 = project3D(xWorld, 240, 650, 0.12, 0, 0, CX, horizonY, 850);
      ctx.strokeStyle = "rgba(129, 140, 248, 0.22)";
      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.stroke();
    }

    for (let r = 0; r < rows; r++) {
      const z = -480 + ((r + scroll) / rows) * 1120;
      const alpha = clamp((z + 480) / 900, 0.05, 0.42);
      const pL = project3D(-cols * 150, 240, z, 0.12, 0, 0, CX, horizonY, 850);
      const pR = project3D(cols * 150, 240, z, 0.12, 0, 0, CX, horizonY, 850);
      ctx.strokeStyle = `rgba(129, 140, 248, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(pL.sx, pL.sy);
      ctx.lineTo(pR.sx, pR.sy);
      ctx.stroke();
    }
  }, [frame]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};

// ============================================================================
// 6. GOLDEN VORTEX — Swirling 320-particle Fibonacci golden-indigo vortex (Beat 10)
// ============================================================================
export const GoldenVortexCanvas: React.FC<{ frame: number }> = ({ frame }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, VW, VH);

    const t = frame / 30;
    const rms = AUDIO_RMS[frame] ?? 0.3;

    const halo = ctx.createRadialGradient(CX, CY - 30, 20, CX, CY - 30, 680);
    halo.addColorStop(0, `rgba(245, 184, 65, ${0.26 + rms * 0.16})`);
    halo.addColorStop(0.45, `rgba(129, 140, 248, ${0.15 + rms * 0.10})`);
    halo.addColorStop(1, "rgba(20, 22, 28, 0)");
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, VW, VH);

    const N = 320;
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
      const frac = i / N;
      const r = 110 + Math.sqrt(frac) * 740;
      const angle = i * goldenAngle + t * (0.48 - frac * 0.25);
      const x = Math.cos(angle) * r;
      const y = Math.sin(angle) * r * 0.65;
      const z = (frac - 0.5) * 340 + Math.sin(t * 2.1 + i * 0.2) * 38;

      const p = project3D(x, y, z, 0.22, Math.sin(t * 0.3) * 0.12, 0, CX, CY - 20);
      const isGold = i % 3 !== 0;
      const alpha = (1 - frac * 0.42) * (0.38 + rms * 0.48);
      const rad = (isGold ? 3.4 : 2.5) * p.scale;

      ctx.fillStyle = isGold
        ? `rgba(245, 184, 65, ${alpha})`
        : `rgba(129, 140, 248, ${alpha})`;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [frame]);

  return (
    <canvas
      ref={ref}
      width={VW}
      height={VH}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
    />
  );
};
