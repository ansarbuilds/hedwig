import React from "react";
import { Easing, interpolate } from "remotion";
import { VO_FRAMES } from "./audioData";

export interface VoiceCue {
  id: string;
  start: number;
  frames: number;
  text: string;
}

// Locked 60.0s (1800 frames @ 30fps) voiceover cue timeline synced to generate-audio.py
export const VOICE_CUES: VoiceCue[] = [
  { id: "01", start: 20,   frames: VO_FRAMES["01"] ?? 76,  text: "Everyone rents AI. We built one." },
  { id: "02", start: 136,  frames: VO_FRAMES["02"] ?? 56,  text: "From zero knobs." },
  { id: "03", start: 244,  frames: VO_FRAMES["03"] ?? 84,  text: "No shortcuts. Pure PyTorch." },
  { id: "04", start: 378,  frames: VO_FRAMES["04"] ?? 104, text: "One billion characters. Learned from scratch." },
  { id: "05", start: 562,  frames: VO_FRAMES["05"] ?? 140, text: "Not just fluent. Aligned. Ninety-eight percent preference win rate." },
  { id: "06", start: 768,  frames: VO_FRAMES["06"] ?? 172, text: "Fourteen million. Ninety-seven million. Two hundred eighty-seven million. Under three dollars." },
  { id: "07", start: 984,  frames: VO_FRAMES["07"] ?? 112, text: "See inside the mind. Every token, live in Brain X-Ray." },
  { id: "08", start: 1216, frames: VO_FRAMES["08"] ?? 124, text: "Crushed to four-bit. Eight hundred thirty-two tokens a second." },
  { id: "09", start: 1418, frames: VO_FRAMES["09"] ?? 118, text: "Mac. iPhone. Any browser. Zero cloud." },
  { id: "10", start: 1616, frames: VO_FRAMES["10"] ?? 132, text: "Meet Hedwig. Open weights. Open code. Yours forever." },
];

export const FilmSubtitles: React.FC<{ frame: number }> = ({ frame }) => {
  const active = VOICE_CUES.find(
    (c) => frame >= c.start - 3 && frame <= c.start + c.frames + 8,
  );
  // Beats 1, 2, and 10 feature giant hero typography owning the screen
  if (!active || active.id === "01" || active.id === "02" || active.id === "10") {
    return null;
  }

  const enterOpacity = interpolate(frame, [active.start - 3, active.start + 5], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitOpacity = interpolate(
    frame,
    [active.start + active.frames - 2, active.start + active.frames + 8],
    [1, 0],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    },
  );
  const y = interpolate(frame, [active.start - 3, active.start + 5], [8, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 24,
        display: "flex",
        justifyContent: "center",
        pointerEvents: "none",
        opacity: Math.min(enterOpacity, exitOpacity),
        transform: `translateY(${y}px)`,
        zIndex: 80,
      }}
    >
      <div
        style={{
          padding: "9px 22px",
          borderRadius: 999,
          background: "rgba(20, 22, 28, 0.86)",
          border: "1px solid rgba(245, 184, 65, 0.28)",
          boxShadow:
            "0 10px 28px rgba(0, 0, 0, 0.44), 0 0 0 1px rgba(129, 140, 248, 0.14) inset",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          color: "#f8f7f4",
          fontFamily: "'Plus Jakarta Sans', -apple-system, sans-serif",
          fontSize: 19,
          fontWeight: 600,
          letterSpacing: "-0.01em",
          lineHeight: 1.25,
          textAlign: "center",
          maxWidth: 1120,
        }}
      >
        {active.text}
      </div>
    </div>
  );
};
