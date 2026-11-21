#!/usr/bin/env python3
"""
Studio soundtrack + Neural VO + Kenney Foley SFX pipeline for Hedwig's 60.0s launch film.

Pipeline:
1. 10 punchy Neural VO cues via edge-tts (en-US-AvaMultilingualNeural, rate="+4%"), saved to
   film/public/vo/01..10.mp3, silence-trimmed, and broadcast-EQ'd via FFmpeg.
2. Real studio music bed (happy-beats-business-moves-vol-9-by-ende-dot-app.mp3) decoded to
   44.1 kHz stereo (60.0s), shaped with 0.8s intro ramp, responsive stereo sidechain ducking
   (~55% dip) under VO, and 2.4s tail fade.
3. Precision Foley SFX choreography from film/assets-src/sfx/ (Kenney soft impacts, bell hits,
   UI clicks, card-slides) + synthesized sub-drops (55 -> 31 Hz) & filtered air-whoosh risers
   on all 9 scene cuts.
4. EBU R128 loudness mastering (-14 LUFS / -1.2 dBTP) via FFmpeg loudnorm.
5. Exports 1800-frame AUDIO_RMS, AUDIO_PEAK, and measured VO_FRAMES to film/src/audioData.ts.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import wave
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILM_DIR = os.path.dirname(SCRIPT_DIR)
VO_DIR = os.path.join(FILM_DIR, "public", "vo")
SFX_DIR = os.path.join(FILM_DIR, "assets-src", "sfx")
MUSIC_DIR = os.path.join(FILM_DIR, "assets-src", "music")
OUT_WAV = os.path.join(FILM_DIR, "public", "demo-soundtrack.wav")
OUT_AUDIODATA = os.path.join(FILM_DIR, "src", "audioData.ts")

SR = 44100
DURATION = 60.0
FPS = 30
TOTAL_FRAMES = int(DURATION * FPS)  # 1800
TOTAL_SAMPLES = int(SR * DURATION)

EDGE_VOICE = "en-US-AvaMultilingualNeural"
EDGE_RATE = "+4%"

# (id, start_frame, text)
VOICE_CUES = [
    ("01", 20,   "Everyone rents AI. We built one."),
    ("02", 136,  "From zero knobs."),
    ("03", 244,  "No shortcuts. Pure PyTorch."),
    ("04", 378,  "One billion characters. Learned from scratch."),
    ("05", 562,  "Not just fluent. Aligned. Ninety-eight percent preference win rate."),
    ("06", 768,  "Fourteen million. Ninety-seven million. Two hundred eighty-seven million. Under three dollars."),
    ("07", 984,  "See inside the mind. Every token, live in Brain X-Ray."),
    ("08", 1216, "Crushed to four-bit. Eight hundred thirty-two tokens a second."),
    ("09", 1418, "Mac. iPhone. Any browser. Zero cloud."),
    ("10", 1616, "Meet Hedwig. Open weights. Open code. Yours forever."),
]

SCENE_CUTS = [128, 236, 370, 550, 760, 972, 1208, 1410, 1606]

LEAD_TRIM_THRESH = 10.0 ** (-36.0 / 20.0)
TAIL_KEEP_THRESH = 10.0 ** (-38.0 / 20.0)


async def _tts_save(text: str, out_mp3: str) -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE)
    await comm.save(out_mp3)


def read_wav_stereo(path: str) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        n = wf.getnframes()
        raw = np.frombuffer(wf.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
    if ch == 1:
        return raw, raw.copy()
    return raw[0::2].copy(), raw[1::2].copy()


def write_wav_stereo(path: str, left: np.ndarray, right: np.ndarray) -> None:
    stereo = np.stack([left, right], axis=1)
    pcm = np.clip(stereo * 32767.0, -32767.0, 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def decode_audio_to_stereo(src_path: str, tmp_dir: str, tag: str, max_secs: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    wav_path = os.path.join(tmp_dir, f"{tag}.wav")
    cmd = ["ffmpeg", "-y", "-i", src_path]
    if max_secs is not None:
        cmd.extend(["-t", f"{max_secs:.3f}"])
    cmd.extend(["-ar", str(SR), "-ac", "2", wav_path])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return read_wav_stereo(wav_path)


def trim_silence(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = np.maximum(np.abs(left), np.abs(right))
    nz = np.where(amp > LEAD_TRIM_THRESH)[0]
    if len(nz) == 0:
        return left, right
    s0 = max(0, int(nz[0]) - int(0.012 * SR))
    nz_tail = np.where(amp > TAIL_KEEP_THRESH)[0]
    s1 = min(len(left), int(nz_tail[-1]) + int(0.024 * SR) + 1)
    tl = left[s0:s1].copy()
    tr = right[s0:s1].copy()
    fade = min(len(tl), int(0.012 * SR))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        tl[:fade] *= ramp
        tr[:fade] *= ramp
        tl[-fade:] *= ramp[::-1]
        tr[-fade:] *= ramp[::-1]
    return tl, tr


def generate_vo_cues(tmp_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    os.makedirs(VO_DIR, exist_ok=True)
    out: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for cid, _, text in VOICE_CUES:
        raw_mp3 = os.path.join(tmp_dir, f"{cid}_raw.mp3")
        clean_wav = os.path.join(tmp_dir, f"{cid}_clean.wav")
        pub_mp3 = os.path.join(VO_DIR, f"{cid}.mp3")

        asyncio.run(_tts_save(text, raw_mp3))
        af = (
            "highpass=f=85,"
            "equalizer=f=220:t=q:w=1.2:g=1.2,"
            "equalizer=f=3600:t=q:w=1.1:g=2.4,"
            "acompressor=threshold=-16dB:ratio=2.4:attack=10:release=140,"
            "aresample=44100"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_mp3, "-af", af, "-ac", "2", "-ar", str(SR), clean_wav],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        l, r = read_wav_stereo(clean_wav)
        tl, tr = trim_silence(l, r)
        trimmed_wav = os.path.join(tmp_dir, f"{cid}_trim.wav")
        write_wav_stereo(trimmed_wav, tl, tr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", trimmed_wav, "-c:a", "libmp3lame", "-b:a", "192k", pub_mp3],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        n_frames = max(1, int(round(len(tl) / SR * FPS)))
        out[cid] = (tl, tr, n_frames)
        print(f"  ✓ VO {cid}: {len(tl)/SR:.2f}s ({n_frames} frames) — {text}", flush=True)
    return out


def mix_in(dst_l: np.ndarray, dst_r: np.ndarray, src_l: np.ndarray, src_r: np.ndarray, frame: float, gain: float = 1.0, pan: float = 0.0) -> None:
    i0 = int(round((frame / FPS) * SR))
    if i0 >= TOTAL_SAMPLES:
        return
    if i0 < 0:
        src_l = src_l[-i0:]
        src_r = src_r[-i0:]
        i0 = 0
    n = min(len(src_l), TOTAL_SAMPLES - i0)
    if n <= 0:
        return
    gl = gain * np.sqrt(1.0 - np.clip(pan, -0.9, 0.9))
    gr = gain * np.sqrt(1.0 + np.clip(pan, -0.9, 0.9))
    dst_l[i0 : i0 + n] += src_l[:n] * gl
    dst_r[i0 : i0 + n] += src_r[:n] * gr


def add_foley_sfx(sfx_l: np.ndarray, sfx_r: np.ndarray, tmp_dir: str, rng: np.random.Generator) -> None:
    sfx_files = {
        "heavy": os.path.join(SFX_DIR, "impact", "impactSoft_heavy_003.ogg"),
        "med1": os.path.join(SFX_DIR, "impact", "impactSoft_medium_001.ogg"),
        "med4": os.path.join(SFX_DIR, "impact", "impactSoft_medium_004.ogg"),
        "bell": os.path.join(SFX_DIR, "impact", "impactBell_heavy_000.ogg"),
        "click2": os.path.join(SFX_DIR, "interface", "click_002.ogg"),
        "click3": os.path.join(SFX_DIR, "interface", "click_003.ogg"),
        "bong": os.path.join(SFX_DIR, "interface", "bong_001.ogg"),
        "switch": os.path.join(SFX_DIR, "interface", "switch_007.ogg"),
        "card": os.path.join(SFX_DIR, "casino", "card-slide-1.ogg"),
    }
    sfx = {k: decode_audio_to_stereo(p, tmp_dir, f"sfx_{k}") for k, p in sfx_files.items()}

    # 1. Scene cuts: soft impact + sub-drop (55 -> 31 Hz) + filtered air-whoosh riser
    for idx, cut_f in enumerate(SCENE_CUTS):
        ct = cut_f / FPS
        # Air whoosh riser into cut
        w_dur = 0.42
        wi = max(0, int((ct - 0.36) * SR))
        wn = min(int(w_dur * SR), TOTAL_SAMPLES - wi)
        if wn > 0:
            tw = np.arange(wn, dtype=np.float64) / SR
            noise = rng.standard_normal(wn)
            spec = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(wn, d=1.0 / SR)
            bp = np.exp(-((freqs - 950.0) / 680.0) ** 2)
            whoosh = np.fft.irfft(spec * bp, n=wn)
            whoosh /= np.std(whoosh) + 1e-8
            w_env = (tw / w_dur) ** 2.6 * np.exp(-np.maximum(0.0, tw - 0.34) * 28.0)
            sfx_l[wi : wi + wn] += whoosh * w_env * 0.08
            sfx_r[wi : wi + wn] += whoosh * w_env * 0.08

        # Sub-drop (55 -> 31 Hz) on cut
        di = int(ct * SR)
        dn = min(int(0.55 * SR), TOTAL_SAMPLES - di)
        if dn > 0:
            td = np.arange(dn, dtype=np.float64) / SR
            dfreq = 31.0 + 24.0 * np.exp(-td * 8.0)
            dph = 2 * np.pi * np.cumsum(dfreq) / SR
            denv = np.exp(-td * 4.8) * (1.0 - np.exp(-td * 450.0))
            drop = np.sin(dph) * denv * 0.24
            sfx_l[di : di + dn] += drop
            sfx_r[di : di + dn] += drop

        imp = "heavy" if idx in (0, 3, 4, 8) else ("med1" if idx % 2 == 1 else "med4")
        mix_in(sfx_l, sfx_r, *sfx[imp], frame=cut_f, gain=0.45 if imp == "heavy" else 0.38)

    # 2. Beat 1 shard impact & Beat 2 odometer lock
    mix_in(sfx_l, sfx_r, *sfx["heavy"], frame=36, gain=0.42)
    mix_in(sfx_l, sfx_r, *sfx["bong"], frame=208, gain=0.35)

    # 3. Beat 3 3D rack-focus card slides
    for f in (260, 285, 310, 335):
        mix_in(sfx_l, sfx_r, *sfx["card"], frame=f, gain=0.28, pan=0.2)

    # 4. Beat 5 98.3% bell, Beat 6 $2.45 bell, Beat 10 Hedwig lockup bell
    mix_in(sfx_l, sfx_r, *sfx["bell"], frame=650, gain=0.36)
    mix_in(sfx_l, sfx_r, *sfx["bell"], frame=905, gain=0.38)
    mix_in(sfx_l, sfx_r, *sfx["bell"], frame=1635, gain=0.44)

    # 5. Beat 7 Brain X-Ray token stream clicks (every 12 frames)
    for i, f in enumerate(range(996, 1188, 12)):
        ck = "click2" if i % 2 == 0 else "click3"
        mix_in(sfx_l, sfx_r, *sfx[ck], frame=f, gain=0.26, pan=-0.25 + 0.5 * ((i % 3) / 2))

    # 6. Beat 8 hydraulic crush + Beat 9 platform card locks
    mix_in(sfx_l, sfx_r, *sfx["switch"], frame=1240, gain=0.32)
    for idx, f in enumerate((1445, 1470, 1495)):
        mix_in(sfx_l, sfx_r, *sfx["card"], frame=f, gain=0.28, pan=-0.3 + idx * 0.3)


def main() -> None:
    rng = np.random.default_rng(42)
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_AUDIODATA), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("🎙 Synthesizing 10 Neural VO cues...", flush=True)
        vo_data = generate_vo_cues(tmp_dir)

        print("🎵 Decoding studio music bed + Kenney Foley SFX...", flush=True)
        vol9 = os.path.join(MUSIC_DIR, "happy-beats-business-moves-vol-9-by-ende-dot-app.mp3")
        vol1 = os.path.join(MUSIC_DIR, "happy-beats-business-moves-vol-1-by-ende-dot-app.mp3")
        music_src = vol9 if os.path.exists(vol9) else vol1
        bed_l, bed_r = decode_audio_to_stereo(music_src, tmp_dir, "music_bed", max_secs=DURATION)
        # Pad or trim cleanly to exact TOTAL_SAMPLES
        if len(bed_l) < TOTAL_SAMPLES:
            bed_l = np.pad(bed_l, (0, TOTAL_SAMPLES - len(bed_l)))
            bed_r = np.pad(bed_r, (0, TOTAL_SAMPLES - len(bed_r)))
        else:
            bed_l = bed_l[:TOTAL_SAMPLES].copy()
            bed_r = bed_r[:TOTAL_SAMPLES].copy()

        # Intro ramp (0.8s) & tail fade (2.4s)
        r_in = int(0.8 * SR)
        bed_l[:r_in] *= np.linspace(0.0, 1.0, r_in)
        bed_r[:r_in] *= np.linspace(0.0, 1.0, r_in)
        r_out = int(2.4 * SR)
        bed_l[-r_out:] *= np.linspace(1.0, 0.0, r_out)
        bed_r[-r_out:] *= np.linspace(1.0, 0.0, r_out)

        sfx_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
        sfx_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
        add_foley_sfx(sfx_l, sfx_r, tmp_dir, rng)

        # Align VO & compute responsive stereo sidechain ducking (~55% dip)
        vo_l = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
        vo_r = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
        duck_mask = np.zeros(TOTAL_SAMPLES, dtype=np.float64)

        for cid, start_f, _ in VOICE_CUES:
            tl, tr, _ = vo_data[cid]
            i0 = int(round((start_f / FPS) * SR))
            i1 = min(TOTAL_SAMPLES, i0 + len(tl))
            if i1 > i0:
                vo_l[i0:i1] += tl[: i1 - i0]
                vo_r[i0:i1] += tr[: i1 - i0]
                pad_pre = max(0, i0 - int(0.05 * SR))
                pad_post = min(TOTAL_SAMPLES, i1 + int(0.16 * SR))
                duck_mask[pad_pre:pad_post] = 1.0

        win = int(0.11 * SR)
        kernel = np.hanning(win)
        kernel /= kernel.sum()
        smooth_duck = np.convolve(duck_mask, kernel, mode="same")
        bed_gain = 0.56 * (1.0 - 0.55 * np.clip(smooth_duck, 0.0, 1.0))

        mix_l = bed_l * bed_gain + sfx_l + vo_l * 0.95
        mix_r = bed_r * bed_gain + sfx_r + vo_r * 0.95

        peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-6)
        if peak > 0.92:
            s = 0.92 / peak
            mix_l *= s
            mix_r *= s

        dry_wav = os.path.join(tmp_dir, "mix_dry.wav")
        write_wav_stereo(dry_wav, mix_l, mix_r)

        print("🔊 Mastering via FFmpeg loudnorm (-14 LUFS / -1.2 dBTP)...", flush=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", dry_wav,
                "-af", "loudnorm=I=-14:TP=-1.2:LRA=11",
                "-ar", str(SR), "-ac", "2", OUT_WAV,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    m_l, m_r = read_wav_stereo(OUT_WAV)
    mono = 0.5 * (m_l + m_r)
    spf = SR // FPS
    rms_arr = []
    peak_arr = []
    for f in range(TOTAL_FRAMES):
        chunk = mono[f * spf : (f + 1) * spf]
        if len(chunk) == 0:
            rms_arr.append(0.0)
            peak_arr.append(0.0)
        else:
            rms_arr.append(float(np.sqrt(np.mean(chunk * chunk))))
            peak_arr.append(float(np.max(np.abs(chunk))))

    max_rms = max(rms_arr) or 1.0
    max_pk = max(peak_arr) or 1.0
    rms_norm = [round(min(1.0, v / max_rms), 4) for v in rms_arr]
    peak_norm = [round(min(1.0, v / max_pk), 4) for v in peak_arr]

    ts_lines = [
        "// Auto-generated by film/scripts/generate-audio.py — do not hand-edit.",
        f"export const AUDIO_RMS: number[] = {rms_norm};",
        f"export const AUDIO_PEAK: number[] = {peak_norm};",
        "export const VO_FRAMES: Record<string, number> = {",
    ]
    for cid, _, _ in VOICE_CUES:
        ts_lines.append(f'  "{cid}": {vo_data[cid][2]},')
    ts_lines.append("};")
    with open(OUT_AUDIODATA, "w") as f:
        f.write("\n".join(ts_lines) + "\n")

    print(f"✅ Soundtrack -> {OUT_WAV}", flush=True)
    print(f"✅ Audio envelopes -> {OUT_AUDIODATA}", flush=True)


if __name__ == "__main__":
    main()
