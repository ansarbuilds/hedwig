"""
Original procedural score & soundscape generator for 'Pip and the Lost Key'.

Synthesizes a warm D-major / B-minor Studio Ghibli-inspired score in pure NumPy:
  1. Warm Felt Piano / Harp arpeggios (harmonics + 8ms soft hammer + stereo spread)
  2. Lush String Pad (detuned multi-voice oscillators + gentle 4.5 Hz vibrato)
  3. Warm Cello / Bass Drone anchoring the harmony
  4. Soft stereo forest breeze (low-pass pink noise with slow organic swell)
  5. Crystalline harmonic shimmer triggered precisely when the golden key is found

Mastered through ffmpeg with concert-hall reverb, warmth EQ, and smooth fades.
"""
import argparse
import os
import subprocess
import wave
import numpy as np

SR = 44100


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _felt_note(freq: float, dur: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Warm felt piano / harp note with 8ms hammer attack and soft upper harmonics."""
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / SR
    att = max(1, int(0.008 * SR))
    env = np.exp(-t * 1.45)
    env[:att] *= np.linspace(0.0, 1.0, att)
    # Taper the last 40ms so cutoffs never click
    tail = min(n, int(0.04 * SR))
    env[-tail:] *= np.linspace(1.0, 0.0, tail)

    # Felt hammer damps high harmonics quickly
    h2_env = np.exp(-t * 2.8)
    h3_env = np.exp(-t * 4.5)
    h4_env = np.exp(-t * 6.5)
    phase_jit = rng.uniform(0, 2 * np.pi)
    sig = (
        1.00 * np.sin(2 * np.pi * freq * t + phase_jit)
        + 0.34 * h2_env * np.sin(2 * np.pi * 2.0 * freq * t + 0.3)
        + 0.12 * h3_env * np.sin(2 * np.pi * 3.0 * freq * t + 0.7)
        + 0.04 * h4_env * np.sin(2 * np.pi * 4.0 * freq * t)
    )
    return sig * env * amp


def _string_chord(freqs: list[float], dur: float, amp: float) -> tuple[np.ndarray, np.ndarray]:
    """Detuned multi-voice string pad with subtle 4.5 Hz vibrato and stereo width."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    fade = min(n // 2, int(0.9 * SR))
    env = np.ones(n, dtype=np.float64)
    env[:fade] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    env[-fade:] = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade - 1, -1, -1) / fade)
    # Slow expressive swell across the chord
    env *= 0.82 + 0.18 * np.sin(np.pi * t / max(dur, 1e-3))

    vib = 0.0022 * np.sin(2 * np.pi * 4.5 * t)
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    detunes = [(-0.0035, 0.75, 0.25), (0.0, 0.5, 0.5), (0.0035, 0.25, 0.75)]
    for idx, f in enumerate(freqs):
        # Gentle phase integration for vibrato
        phase_base = 2 * np.pi * f * (t + np.cumsum(vib) / SR)
        voice = (
            0.70 * np.sin(phase_base + idx * 0.4)
            + 0.22 * np.sin(2.0 * phase_base + 0.5)
            + 0.08 * np.sin(3.0 * phase_base + 1.1)
        )
        for det, pan_l, pan_r in detunes:
            d_sig = 0.35 * np.sin(phase_base * (1.0 + det) + idx * 0.7) + 0.65 * voice * (
                1.0 if det == 0.0 else 0.7
            )
            left += d_sig * pan_l
            right += d_sig * pan_r
    norm = max(1.0, len(freqs) * 1.8)
    return (left / norm) * env * amp, (right / norm) * env * amp


def _cello_drone(freq: float, dur: float, amp: float) -> np.ndarray:
    """Warm low-register cello / bass anchor."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    fade = min(n // 2, int(0.6 * SR))
    env = np.ones(n, dtype=np.float64)
    env[:fade] = np.linspace(0.0, 1.0, fade)
    env[-fade:] = np.linspace(1.0, 0.0, fade)
    vib = 0.0015 * np.sin(2 * np.pi * 4.2 * t)
    ph = 2 * np.pi * freq * (t + np.cumsum(vib) / SR)
    sig = 0.72 * np.sin(ph) + 0.26 * np.sin(2 * ph + 0.2) + 0.10 * np.sin(3 * ph + 0.4)
    return sig * env * amp


def _forest_breeze(total_secs: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Soft stereo pink-ish forest breeze with slow organic swells."""
    n = int(total_secs * SR)
    # 1/f-ish noise via simple multi-pole running average smoothing in chunks
    white_l = rng.standard_normal(n)
    white_r = rng.standard_normal(n)
    # Two-stage exponential low-pass (~380 Hz cutoff for warm distant breeze)
    alpha = 0.045
    def lpf(x: np.ndarray, a: float) -> np.ndarray:
        # Fast FFT low-pass filter
        freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
        spec = np.fft.rfft(x)
        # Gentle pink slope + low-pass rolloff around 420 Hz
        gain = 1.0 / np.sqrt(1.0 + (freqs / 90.0) ** 2)
        gain *= 1.0 / (1.0 + (freqs / 450.0) ** 4)
        gain[0] = 0.0
        out = np.fft.irfft(spec * gain, n=len(x))
        return out / (np.std(out) + 1e-8)

    breeze_l = lpf(white_l, alpha)
    breeze_r = lpf(white_r, alpha)
    t = np.arange(n, dtype=np.float64) / SR
    swell = 0.55 + 0.45 * np.sin(2 * np.pi * t / 13.5 + 0.4)
    swell *= 0.8 + 0.2 * np.sin(2 * np.pi * t / 7.1)
    return breeze_l * swell * 0.028, breeze_r * swell * 0.028


def _key_shimmer(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Luminous crystalline chime / harmonic sparkle for Scene 4 key discovery."""
    dur = 4.2
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)
    # Sparkling D major 9th upper cluster (D6, F#6, A6, E7, A7)
    notes = [(86, 0.00, -0.35), (90, 0.09, 0.30), (93, 0.19, -0.20),
             (100, 0.31, 0.40), (105, 0.44, 0.00)]
    for midi, offset, pan in notes:
        i0 = int(offset * SR)
        seg = t[: n - i0]
        att = max(1, int(0.015 * SR))
        env = np.exp(-seg * 1.15)
        env[:att] *= np.linspace(0.0, 1.0, att)
        f = midi_to_hz(midi)
        tone = (
            0.75 * np.sin(2 * np.pi * f * seg)
            + 0.25 * np.sin(2 * np.pi * f * 2.003 * seg) * np.exp(-seg * 2.0)
        ) * env * 0.14
        pl = np.sqrt(0.5 * (1.0 - pan))
        pr = np.sqrt(0.5 * (1.0 + pan))
        left[i0:] += tone * pl
        right[i0:] += tone * pr
    return left, right


def generate_score(
    total_secs: float,
    scene_starts: list[float],
    out_wav: str,
    seed: int = 42,
) -> str:
    """Build and master the complete stereo orchestral/ambient score to `out_wav`."""
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    rng = np.random.default_rng(seed)
    n_total = int(np.ceil(total_secs * SR))
    left = np.zeros(n_total, dtype=np.float64)
    right = np.zeros(n_total, dtype=np.float64)

    # Scene 4 is index 4 (key found); credits is last index
    key_found_t = scene_starts[4] if len(scene_starts) > 4 else total_secs * 0.5
    credits_t = scene_starts[-1] if len(scene_starts) > 1 else total_secs - 7.0

    # Harmonic language in D major / B minor (Hisaishi-inspired)
    # Opening/Search cycle: Bm9 -> Gmaj7 -> Dmaj9 -> Aadd9
    cycle_a = [
        (47, [59, 62, 66, 69, 73], [62, 66, 69, 73, 74, 78]),   # Bm9
        (43, [55, 59, 62, 66, 71], [59, 62, 66, 67, 71, 74]),   # Gmaj7
        (50, [62, 66, 69, 73, 76], [66, 69, 73, 76, 78, 81]),   # Dmaj9
        (45, [57, 61, 64, 69, 71], [61, 64, 69, 71, 76, 81]),   # Aadd9
    ]
    # Discovery & Finale cycle: Gmaj9 -> A/C# -> Bm7 -> Dmaj9
    cycle_b = [
        (43, [55, 59, 62, 66, 73], [62, 66, 69, 71, 74, 78]),   # Gmaj9
        (49, [61, 64, 69, 73, 76], [64, 69, 73, 76, 81, 85]),   # A/C#
        (47, [59, 62, 66, 69, 74], [62, 66, 69, 74, 78, 83]),   # Bm7
        (50, [62, 66, 69, 73, 76], [66, 69, 73, 76, 78, 86]),   # Dmaj9
    ]
    final_dmaj9 = (38, [50, 57, 62, 66, 69, 73, 76], [62, 66, 69, 73, 76, 86])

    bar_dur = 3.6  # gentle ~67 BPM 4/4 feel
    t_cur = 0.0
    bar_idx = 0

    while t_cur < total_secs - 0.5:
        if t_cur >= credits_t - 0.8:
            bass_m, pad_ms, arp_ms = final_dmaj9
            this_dur = max(3.5, total_secs - t_cur)
            is_final = True
        else:
            cycle = cycle_a if t_cur < key_found_t - 0.3 else cycle_b
            bass_m, pad_ms, arp_ms = cycle[bar_idx % len(cycle)]
            this_dur = min(bar_dur + 0.4, total_secs - t_cur)
            is_final = False

        i0 = int(t_cur * SR)
        # Dynamic arc: gentler in opening, warmer lift at discovery & ending
        lift = 1.22 if t_cur >= key_found_t - 0.3 else 0.95
        if is_final:
            lift = 1.05

        # 1. Lush String Pad
        pad_freqs = [midi_to_hz(m) for m in pad_ms]
        pl, pr = _string_chord(pad_freqs, this_dur + 0.5, amp=0.21 * lift)
        end_i = min(n_total, i0 + len(pl))
        left[i0:end_i] += pl[: end_i - i0]
        right[i0:end_i] += pr[: end_i - i0]

        # 2. Warm Cello Drone
        cello = _cello_drone(midi_to_hz(bass_m), this_dur + 0.3, amp=0.14 * lift)
        end_c = min(n_total, i0 + len(cello))
        left[i0:end_c] += cello[: end_c - i0] * 0.72
        right[i0:end_c] += cello[: end_c - i0] * 0.72

        # 3. Felt Piano / Harp Arpeggio
        if is_final:
            # Slow delicate rising resolving arpeggio on final Dmaj9
            pattern = [0, 1, 2, 3, 4, 5]
            step = 0.42
        else:
            # Gentle flowing 8-note pattern (up and back)
            pattern = [0, 2, 3, 4, 5, 4, 2, 1]
            step = bar_dur / len(pattern)

        for k, note_idx in enumerate(pattern):
            nt = t_cur + k * step
            if nt >= total_secs - 1.2:
                break
            m = arp_ms[note_idx % len(arp_ms)]
            vel = rng.uniform(0.86, 1.06) * (0.14 if not is_final else 0.16) * lift
            if k == 0:
                vel *= 1.15
            note_sig = _felt_note(midi_to_hz(m), 2.8, vel, rng)
            pan = -0.35 + 0.70 * (note_idx / max(1, len(arp_ms) - 1))
            pan_l = np.sqrt(0.5 * (1.0 - pan))
            pan_r = np.sqrt(0.5 * (1.0 + pan))
            ni = int(nt * SR)
            ne = min(n_total, ni + len(note_sig))
            left[ni:ne] += note_sig[: ne - ni] * pan_l
            right[ni:ne] += note_sig[: ne - ni] * pan_r

        if is_final:
            break
        t_cur += bar_dur
        bar_idx += 1

    # 4. Soft Forest Breeze ambient bed
    bl, br = _forest_breeze(total_secs, rng)
    left[: len(bl)] += bl
    right[: len(br)] += br

    # 5. Magical Key Discovery Shimmer at Scene 4 timestamp
    sl, sr = _key_shimmer(rng)
    si = int((key_found_t + 0.15) * SR)
    if si < n_total:
        se = min(n_total, si + len(sl))
        left[si:se] += sl[: se - si]
        right[si:se] += sr[: se - si]

    # Soft limiting & headroom before ffmpeg mastering
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6)
    scale = 0.78 / peak if peak > 0.78 else 1.0
    stereo = np.stack([left * scale, right * scale], axis=1)
    pcm = np.clip(stereo * 32767.0, -32767.0, 32767.0).astype("<i2")

    raw_wav = out_wav + ".dry.wav"
    with wave.open(raw_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    fade_out_st = max(0.5, total_secs - 3.0)
    af = (
        f"aecho=0.8:0.88:45|90:0.28|0.18,"
        f"equalizer=f=220:t=q:w=1.2:g=2.0,"
        f"equalizer=f=5500:t=q:w=1.0:g=-1.5,"
        f"afade=t=in:st=0:d=1.5,"
        f"afade=t=out:st={fade_out_st:.2f}:d=3.0"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_wav, "-af", af, "-ar", str(SR), "-ac", "2", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    os.remove(raw_wav)
    return out_wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=65.0)
    p.add_argument("--out", default="movie/build/score.wav")
    args = p.parse_args()
    starts = [0.0, 6.0, 17.0, 25.0, 34.0, 41.0, 51.0, 58.0]
    generate_score(args.duration, starts, args.out)
    print(f"Score written to {args.out}")


if __name__ == "__main__":
    main()
