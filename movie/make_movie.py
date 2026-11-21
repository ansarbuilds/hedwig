"""
Cinema-grade short film pipeline for 'Pip and the Lost Key'.

Combines:
  - Hedwig's exact seed-8 story (286.7M Flagship brain, word for word)
  - 6 painterly concept-art film stills (movie/scenes_cinema/) with a 9-beat bookend edit
  - 2.39:1 widescreen letterbox (1920x816 active frame inside 1920x1080)
  - Cubic ease-in-out camera choreography + additive bokeh particles + light bloom + vignette
  - Animated Didot / Futura cinema title & credits, Baskerville subtitles in the bottom bar
  - 0.85s xfade scene transitions + 35mm cinema color grade & subtle film grain
  - Neural storytelling voice (edge-tts en-GB-RyanNeural, fallback macOS Daniel)
  - Original synthesized score + soundscape (make_score.py) with sidechain ducking & loudnorm

Run:  .venv/bin/python movie/make_movie.py
"""
import asyncio
import json
import math
import os
import subprocess
import sys
import textwrap
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_score

SCENES_DIR = os.path.join(HERE, "scenes_cinema")
BUILD_DIR = os.path.join(HERE, "build")
OUT_MP4 = os.path.join(HERE, "pip_and_the_lost_key.mp4")

W, H, FPS = 1920, 1080, 30
CW, CH = 1920, 816          # 2.39:1 cinema frame
BAR_H = (H - CH) // 2       # 132 px top & bottom letterbox
XFADE = 0.85                # seconds per scene crossfade
TITLE_CX = CW * 0.27        # centered over wide-open sky on the left

DIDOT_PATH = "/System/Library/Fonts/Supplemental/Didot.ttc"
FUTURA_PATH = "/System/Library/Fonts/Supplemental/Futura.ttc"
BASKERVILLE_PATH = "/System/Library/Fonts/Supplemental/Baskerville.ttc"

EDGE_VOICE = "en-GB-RyanNeural"
EDGE_RATE = "-8%"
EDGE_PITCH = "-2Hz"

# (image_file, subtitle_text, spoken_override, camera_params, particle_style)
# camera_params = (zoom0, zoom1, cx0, cy0, cx1, cy1)
# Hedwig's seed-8 story is kept 100% word-for-word across scenes 1..7.
SCENES = [
    (
        "0_title.jpg",
        None,
        "Pip and the Lost Key. A short film written by Hedwig.",
        (1.12, 1.00, 0.46, 0.42, 0.50, 0.48),
        "pollen",
    ),
    (
        "1_tree.jpg",
        "Once upon a time, there was an owl named Pip. Pip lived in a big tree with his friend, "
        "a little girl named Lily. They loved to play together and have fun.",
        None,
        (1.02, 1.13, 0.48, 0.45, 0.52, 0.52),
        "pollen",
    ),
    (
        "2_promise.jpg",
        "One sunny day, Pip said to Lily, \u201cI will find your key. I am very good at finding it.\u201d",
        None,
        (1.04, 1.14, 0.52, 0.48, 0.55, 0.46),
        "pollen",
    ),
    (
        "3_search.jpg",
        "So, they looked all around the tree for the key. They asked other animals if they had seen it, "
        "but no one did.",
        None,
        (1.10, 1.14, 0.43, 0.52, 0.57, 0.50),
        "fireflies",
    ),
    (
        "4_found.jpg",
        "Then, they saw a big, shiny key under a bush. They were so happy!",
        None,
        (1.00, 1.12, 0.50, 0.48, 0.52, 0.53),
        "sparkles",
    ),
    (
        "5_mom.jpg",
        "Pip and Lily picked up the key and brought it home to show their friend. When her mom came back, "
        "she was very surprised when she found her lost key!",
        None,
        (1.00, 1.07, 0.50, 0.50, 0.52, 0.48),
        "dusk",
    ),
    (
        "5_mom.jpg",
        "She said, \u201cThank you for helping me find my key. I will never lose it again.\u201d",
        None,
        (1.12, 1.20, 0.55, 0.46, 0.57, 0.44),
        "dusk",
    ),
    (
        "1_tree.jpg",
        "And they all played together happily ever after.",
        None,
        (1.16, 1.00, 0.52, 0.44, 0.50, 0.42),
        "sparkles",
    ),
    (
        None,
        None,
        "The end.",
        (1.0, 1.0, 0.5, 0.5, 0.5, 0.5),
        "stardust",
    ),
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


async def _edge_tts_save(text: str, out_mp3: str) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH)
    await comm.save(out_mp3)


def synth_narration(idx: int, text: str) -> tuple[str, float]:
    """Generate and master narration line to a clean 44.1 kHz stereo WAV."""
    raw_mp3 = os.path.join(BUILD_DIR, f"voice_{idx}_raw.mp3")
    raw_aiff = os.path.join(BUILD_DIR, f"voice_{idx}_raw.aiff")
    master_wav = os.path.join(BUILD_DIR, f"voice_{idx}.wav")

    used_edge = False
    try:
        asyncio.run(_edge_tts_save(text, raw_mp3))
        if os.path.exists(raw_mp3) and os.path.getsize(raw_mp3) > 1000:
            used_edge = True
    except Exception as exc:
        print(f"  ! edge-tts failed ({exc}); falling back to macOS Daniel", flush=True)

    src_audio = raw_mp3
    if not used_edge:
        run(["say", "-v", "Daniel", "-r", "148", "-o", raw_aiff, text])
        src_audio = raw_aiff

    af = (
        "highpass=f=80,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=15:release=180,"
        "aecho=0.8:0.9:18:0.08,"
        "aresample=44100"
    )
    run(["ffmpeg", "-y", "-i", src_audio, "-af", af, "-ac", "2", "-ar", "44100", master_wav])
    dur = ffprobe_duration(master_wav)
    return master_wav, dur


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def make_vignette_lut() -> np.ndarray:
    """Smooth radial vignette multiplier (CH x CW x 1) float32."""
    y = np.linspace(-1.0, 1.0, CH, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, CW, dtype=np.float32)[None, :]
    r2 = (x * x) * 0.85 + (y * y) * 1.05
    vig = 1.0 - 0.30 * np.clip((r2 - 0.35) / 1.25, 0.0, 1.0) ** 1.4
    return vig[:, :, None]


def make_bokeh_sprite(radius: int, color: tuple[int, int, int]) -> np.ndarray:
    """Soft radial bokeh glow sprite (S x S x 3) float32."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / max(1.0, float(radius))
    core = np.exp(-((d * 2.2) ** 2))
    halo = 0.45 * np.exp(-((d * 1.1) ** 2)) * (d < 1.0)
    falloff = np.clip(1.0 - d, 0.0, 1.0)
    intensity = np.clip((core + halo) * falloff, 0.0, 1.0)[:, :, None]
    rgb = np.array(color, dtype=np.float32)[None, None, :]
    return intensity * rgb


def make_particles(style: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    counts = {"pollen": 44, "fireflies": 48, "sparkles": 56, "dusk": 36, "stardust": 42}
    n = counts.get(style, 40)
    particles = []
    for _ in range(n):
        if style == "pollen":
            color = (255, 224, 150) if rng.random() < 0.7 else (255, 240, 200)
            rad = int(rng.integers(6, 16))
            vx, vy = rng.uniform(8.0, 22.0), rng.uniform(-10.0, -3.0)
            base_a = rng.uniform(0.22, 0.55)
        elif style == "fireflies":
            color = (255, 210, 95) if rng.random() < 0.75 else (195, 235, 130)
            rad = int(rng.integers(8, 20))
            vx, vy = rng.uniform(-14.0, 14.0), rng.uniform(-14.0, -3.0)
            base_a = rng.uniform(0.32, 0.78)
        elif style == "sparkles":
            color = (255, 230, 130) if rng.random() < 0.65 else (255, 248, 210)
            rad = int(rng.integers(7, 22))
            vx, vy = rng.uniform(-8.0, 8.0), rng.uniform(-24.0, -9.0)
            base_a = rng.uniform(0.35, 0.82)
        elif style == "dusk":
            color = (255, 215, 160)
            rad = int(rng.integers(6, 14))
            vx, vy = rng.uniform(4.0, 14.0), rng.uniform(-7.0, -2.0)
            base_a = rng.uniform(0.18, 0.42)
        else:  # stardust on credits
            color = (240, 214, 150) if rng.random() < 0.6 else (185, 205, 245)
            rad = int(rng.integers(4, 11))
            vx, vy = rng.uniform(-5.0, 5.0), rng.uniform(-9.0, -2.0)
            base_a = rng.uniform(0.18, 0.48)

        particles.append(
            {
                "x0": float(rng.uniform(0, CW)),
                "y0": float(rng.uniform(0, CH)),
                "vx": float(vx),
                "vy": float(vy),
                "amp": float(rng.uniform(8.0, 24.0)),
                "w_freq": float(rng.uniform(0.4, 1.1)),
                "p_freq": float(rng.uniform(0.6, 2.0)),
                "phase": float(rng.uniform(0, 2 * math.pi)),
                "alpha": float(base_a),
                "sprite": make_bokeh_sprite(rad, color),
                "rad": rad,
            }
        )
    return particles


def make_light_bloom(style: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Pre-render subtle warm upper-right sunbeam glow with smooth zeroed edges."""
    if style in ("fireflies", "stardust"):
        return None
    bw, bh = 960, 600
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    nx = (bw - xx) / bw
    ny = yy / bh
    diag = np.exp(-(((ny - 0.42 * nx) * 2.1) ** 2))
    falloff = np.exp(-((nx * 1.15) ** 2 + (ny * 1.15) ** 2))
    # Smooth window to exact zero at left (xx=0) and bottom (yy=bh-1) borders
    edge_x = np.sin(0.5 * np.pi * np.clip(xx / (bw * 0.45), 0.0, 1.0))
    edge_y = np.sin(0.5 * np.pi * np.clip((bh - 1 - yy) / (bh * 0.45), 0.0, 1.0))
    glow = np.clip(diag * falloff * edge_x * edge_y, 0.0, 1.0)[:, :, None]
    tint = np.array([255.0, 216.0, 140.0], dtype=np.float32)[None, None, :]
    if style == "dusk":
        tint = np.array([255.0, 198.0, 128.0], dtype=np.float32)[None, None, :]
    return glow * tint, (CW - bw, 0)


def composite_additive(frame_u8: np.ndarray, sprite: np.ndarray, cx: int, cy: int, weight: float) -> None:
    """In-place additive composite of `sprite * weight` centered at (cx, cy)."""
    sh, sw, _ = sprite.shape
    rx, ry = sw // 2, sh // 2
    x0, y0 = cx - rx, cy - ry
    x1, y1 = x0 + sw, y0 + sh

    sx0, sy0 = 0, 0
    sx1, sy1 = sw, sh
    if x0 < 0:
        sx0 = -x0
        x0 = 0
    if y0 < 0:
        sy0 = -y0
        y0 = 0
    if x1 > CW:
        sx1 -= x1 - CW
        x1 = CW
    if y1 > CH:
        sy1 -= y1 - CH
        y1 = CH
    if x0 >= x1 or y0 >= y1:
        return

    patch = frame_u8[y0:y1, x0:x1].astype(np.float32)
    patch += sprite[sy0:sy1, sx0:sx1] * weight
    np.clip(patch, 0, 255, out=patch)
    frame_u8[y0:y1, x0:x1] = patch.astype(np.uint8)


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont,
    tracking: int,
    fill: tuple[int, int, int, int],
    shadow_fill: tuple[int, int, int, int] | None = None,
) -> None:
    widths = [draw.textlength(ch, font=font) for ch in text]
    total_w = sum(widths) + tracking * max(0, len(text) - 1)
    x = center_x - total_w / 2.0
    for ch, cw in zip(text, widths):
        if shadow_fill is not None:
            draw.text((x + 2, y + 3), ch, font=font, fill=shadow_fill)
        draw.text((x, y), ch, font=font, fill=fill)
        x += cw + tracking


def make_title_overlays() -> tuple[Image.Image, Image.Image]:
    """Pre-render crisp Didot title and Futura subtitle RGBA overlays centered at TITLE_CX."""
    f_title = ImageFont.truetype(DIDOT_PATH, 54, index=0)
    f_sub = ImageFont.truetype(FUTURA_PATH, 19, index=0)

    ov_main = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d1 = ImageDraw.Draw(ov_main)
    draw_tracked_text(
        d1,
        TITLE_CX,
        175,
        "PIP AND THE LOST KEY",
        f_title,
        tracking=7,
        fill=(252, 243, 222, 255),
        shadow_fill=(15, 10, 6, 165),
    )

    ov_sub = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(ov_sub)
    draw_tracked_text(
        d2,
        TITLE_CX,
        276,
        "A SHORT FILM WRITTEN BY HEDWIG",
        f_sub,
        tracking=6,
        fill=(235, 206, 146, 255),
        shadow_fill=(15, 10, 6, 160),
    )
    return ov_main, ov_sub


def make_subtitle_strip(text: str) -> np.ndarray:
    """Pre-render crisp Baskerville ivory subtitles for the lower 132px black bar."""
    img = Image.new("RGB", (W, BAR_H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(BASKERVILLE_PATH, 35, index=0)
    lines = textwrap.wrap(text, width=78)
    line_h = 44
    total_h = line_h * len(lines)
    y0 = (BAR_H - total_h) // 2 - 2
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=font)
        d.text(((W - tw) / 2.0, y0 + i * line_h), line, font=font, fill=(246, 240, 226))
    return np.asarray(img, dtype=np.float32)


def make_credits_bg_and_overlay() -> tuple[np.ndarray, np.ndarray]:
    """Deep midnight-indigo gradient canvas + centered Didot/Futura/Baskerville credits RGBA."""
    yy = np.linspace(-1.0, 1.0, CH, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, CW, dtype=np.float32)[None, :]
    r2 = np.clip(0.55 * xx * xx + 0.85 * yy * yy, 0.0, 1.4)
    center = np.array([15.0, 18.0, 38.0], dtype=np.float32)
    edge = np.array([5.0, 6.0, 14.0], dtype=np.float32)
    w = np.clip(r2 / 1.2, 0.0, 1.0)[:, :, None]
    bg = (center * (1.0 - w) + edge * w).astype(np.uint8)

    ov = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_end = ImageFont.truetype(DIDOT_PATH, 78, index=0)
    f_line1 = ImageFont.truetype(BASKERVILLE_PATH, 32, index=0)
    f_meta = ImageFont.truetype(FUTURA_PATH, 21, index=0)
    f_foot = ImageFont.truetype(BASKERVILLE_PATH, 27, index=2)

    draw_tracked_text(d, CW / 2, 185, "THE END", f_end, tracking=18, fill=(252, 243, 222, 255))
    d.line([(CW // 2 - 180, 295), (CW // 2 + 180, 295)], fill=(224, 190, 120, 200), width=2)

    lines = [
        ("Story written word-for-word by Hedwig \u2014 286.7M Flagship", f_line1, 345, (240, 234, 220, 245), 0),
        ("TRAINED FROM SCRATCH  \u00b7  SEED 8  \u00b7  TEMPERATURE 0.7", f_meta, 405, (218, 190, 132, 230), 4),
        ("Visuals, Original Score & Direction", f_foot, 495, (205, 202, 214, 220), 0),
    ]
    for txt, fnt, y, col, trk in lines:
        if trk > 0:
            draw_tracked_text(d, CW / 2, y, txt, fnt, tracking=trk, fill=col)
        else:
            tw = d.textlength(txt, font=fnt)
            d.text(((CW - tw) / 2.0, y), txt, font=fnt, fill=col)

    return bg, np.asarray(ov, dtype=np.float32)


def crop_camera_frame(src_img: Image.Image, zoom: float, cx: float, cy: float) -> np.ndarray:
    """Sub-pixel bicubic crop from 16:9 image into 1920x816 cinema frame."""
    sw, sh = src_img.size
    target_aspect = CW / CH
    if sw / sh > target_aspect:
        base_h = float(sh)
        base_w = base_h * target_aspect
    else:
        base_w = float(sw)
        base_h = base_w / target_aspect

    box_w = base_w / zoom
    box_h = base_h / zoom
    x0 = cx * sw - box_w / 2.0
    y0 = cy * sh - box_h / 2.0
    x0 = max(0.0, min(sw - box_w, x0))
    y0 = max(0.0, min(sh - box_h, y0))

    cropped = src_img.transform(
        (CW, CH),
        Image.Transform.EXTENT,
        (x0, y0, x0 + box_w, y0 + box_h),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(cropped, dtype=np.uint8).copy()


def alpha_blend_rgba_into(dst_rgb: np.ndarray, ov_rgba: np.ndarray, alpha_scale: float, y_shift: int = 0) -> None:
    """Composite pre-rendered RGBA overlay onto `dst_rgb` with global alpha multiplier & vertical drift."""
    if alpha_scale <= 0.005:
        return
    a = (ov_rgba[:, :, 3:4] / 255.0) * alpha_scale
    rgb = ov_rgba[:, :, :3]
    if y_shift != 0:
        if y_shift > 0:
            a_s = a[:-y_shift]
            rgb_s = rgb[:-y_shift]
            dst_slice = dst_rgb[y_shift:]
        else:
            s = -y_shift
            a_s = a[s:]
            rgb_s = rgb[s:]
            dst_slice = dst_rgb[:-s]
    else:
        a_s = a
        rgb_s = rgb
        dst_slice = dst_rgb

    blended = dst_slice.astype(np.float32) * (1.0 - a_s) + rgb_s * a_s
    dst_slice[:] = np.clip(blended, 0, 255).astype(np.uint8)


def render_scene_video(
    idx: int,
    pic_name: str | None,
    subtitle: str | None,
    cam: tuple[float, float, float, float, float, float],
    p_style: str,
    total_secs: float,
    lead_in: float,
    voice_dur: float,
    vignette: np.ndarray,
    title_ovs: tuple[Image.Image, Image.Image],
    credits_assets: tuple[np.ndarray, np.ndarray],
) -> tuple[str, int]:
    """Render one animated scene clip (video only) by piping RGB24 frames to ffmpeg."""
    frames = int(round(total_secs * FPS))
    exact_secs = frames / FPS
    clip_path = os.path.join(BUILD_DIR, f"scene_{idx}.mp4")

    src_img = None
    if pic_name is not None:
        raw = Image.open(os.path.join(SCENES_DIR, pic_name)).convert("RGB")
        if raw.width < 2400:
            scale = 2400 / raw.width
            raw = raw.resize((2400, int(round(raw.height * scale))), Image.Resampling.LANCZOS)
        src_img = raw

    particles = make_particles(p_style, seed=100 + idx * 17)
    bloom = make_light_bloom(p_style) if pic_name is not None else None
    sub_strip = make_subtitle_strip(subtitle) if subtitle else None

    if idx == 0:
        ov_title_np = np.asarray(title_ovs[0], dtype=np.float32)
        ov_sub_np = np.asarray(title_ovs[1], dtype=np.float32)
    credits_bg, credits_ov = credits_assets

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p",
        clip_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    z0, z1, cx0, cy0, cx1, cy1 = cam
    sub_start = lead_in - 0.15
    sub_end = lead_in + voice_dur + 0.35

    for f in range(frames):
        t = f / FPS
        u = f / max(1, frames - 1)
        e = ease_in_out_cubic(u)

        if src_img is not None:
            zoom = z0 + (z1 - z0) * e
            cx = cx0 + (cx1 - cx0) * e
            cy = cy0 + (cy1 - cy0) * e
            frame = crop_camera_frame(src_img, zoom, cx, cy)
        else:
            frame = credits_bg.copy()

        # Subtle breathing warm sunbeam bloom
        if bloom is not None:
            b_sprite, (bx, by) = bloom
            bh, bw, _ = b_sprite.shape
            b_weight = 0.11 + 0.05 * math.sin(2.0 * math.pi * t / 5.5 + idx)
            if p_style == "sparkles":
                b_weight *= 1.45
            region = frame[by : by + bh, bx : bx + bw].astype(np.float32)
            region += b_sprite * b_weight
            frame[by : by + bh, bx : bx + bw] = np.clip(region, 0, 255).astype(np.uint8)

        # Animated atmospheric bokeh particles
        for p in particles:
            px = (p["x0"] + p["vx"] * t + p["amp"] * math.sin(p["w_freq"] * t + p["phase"])) % CW
            py = (p["y0"] + p["vy"] * t + 0.5 * p["amp"] * math.cos(p["w_freq"] * 0.8 * t + p["phase"])) % CH
            pulse = 0.55 + 0.45 * math.sin(p["p_freq"] * t * 2.2 + p["phase"])
            composite_additive(frame, p["sprite"], int(px), int(py), p["alpha"] * pulse)

        # Vignette
        if src_img is not None:
            frame = np.clip(frame.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        # Scene 0 animated cinema title centered at TITLE_CX over open sky
        if idx == 0:
            fade_in_t = clamp01((t - 0.65) / 1.15)
            fade_out_t = clamp01((exact_secs - 0.90 - t) / 0.85)
            ta = ease_in_out_cubic(fade_in_t) * fade_out_t
            drift = int(round((1.0 - ease_in_out_cubic(fade_in_t)) * 18))
            alpha_blend_rgba_into(frame, ov_title_np, ta, y_shift=drift)

            line_prog = ease_in_out_cubic(clamp01((t - 1.15) / 0.95))
            line_a = ta * line_prog
            if line_a > 0.02:
                half_w = int(round(190 * line_prog))
                lx0 = max(0, int(TITLE_CX) - half_w)
                lx1 = min(CW, int(TITLE_CX) + half_w)
                ly = 254
                col = np.array([232.0, 198.0, 130.0], dtype=np.float32)
                la = 0.85 * line_a
                frame[ly : ly + 2, lx0:lx1] = np.clip(
                    frame[ly : ly + 2, lx0:lx1].astype(np.float32) * (1.0 - la) + col * la,
                    0,
                    255,
                ).astype(np.uint8)

            sub_in = clamp01((t - 1.35) / 1.05)
            alpha_blend_rgba_into(frame, ov_sub_np, ease_in_out_cubic(sub_in) * fade_out_t, y_shift=drift // 2)

        # End credits
        if src_img is None:
            ca = ease_in_out_cubic(clamp01((t - 0.55) / 1.25))
            fade_end = clamp01((exact_secs - 0.4 - t) / 1.2)
            alpha_blend_rgba_into(frame, credits_ov, ca * fade_end, y_shift=int((1.0 - ca) * 14))

        # Place cinema frame inside 2.39:1 letterbox canvas
        canvas[BAR_H : BAR_H + CH] = frame

        # Smoothly faded Baskerville subtitles inside bottom black bar
        if sub_strip is not None:
            sa = clamp01((t - sub_start) / 0.35) * clamp01((sub_end - t) / 0.35)
            if sa > 0.005:
                canvas[BAR_H + CH :] = np.clip(sub_strip * sa, 0, 255).astype(np.uint8)
            else:
                canvas[BAR_H + CH :] = 0

        # Smooth fade from black on very first clip, fade to black on credits
        if idx == 0 and t < 0.7:
            canvas[:] = np.clip(canvas.astype(np.float32) * clamp01(t / 0.7), 0, 255).astype(np.uint8)
        elif src_img is None and t > exact_secs - 1.4:
            canvas[:] = np.clip(canvas.astype(np.float32) * clamp01((exact_secs - t) / 1.4), 0, 255).astype(np.uint8)

        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed on scene {idx}: {proc.stderr.read().decode()}")
    return clip_path, frames


def assemble_video_with_xfade(clips: list[str], frame_counts: list[int], out_video: str) -> tuple[list[float], float]:
    """Chain all scene clips via 0.85s xfade + final cinema color grade & subtle 35mm grain."""
    inputs = []
    for c in clips:
        inputs.extend(["-i", c])

    scene_starts = [0.0]
    cur_start = 0.0
    filters = []
    last_label = "[0:v]"
    for i in range(len(clips) - 1):
        d_i = frame_counts[i] / FPS
        offset = cur_start + d_i - XFADE
        nxt = f"[v{i+1}]" if i < len(clips) - 2 else "[vcat]"
        filters.append(f"{last_label}[{i+1}:v]xfade=transition=fade:duration={XFADE:.4f}:offset={offset:.4f}{nxt}")
        cur_start = offset
        scene_starts.append(cur_start)
        last_label = nxt

    grade = (
        f"{last_label}"
        "colorbalance=rs=-0.02:bs=0.03:rh=0.04:gh=0.01:bh=-0.02,"
        "noise=alls=3:allf=t+u,"
        "format=yuv420p[vout]"
    )
    filters.append(grade)

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-r", str(FPS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        out_video,
    ]
    run(cmd)
    total_dur = ffprobe_duration(out_video)
    return scene_starts, total_dur


def mix_audio_and_mux(
    video_path: str,
    voice_wavs: list[str],
    scene_starts: list[float],
    lead_ins: list[float],
    total_dur: float,
    out_mp4: str,
) -> None:
    """Build exact narration track, generate score, sidechain duck, loudnorm, and mux."""
    narration_wav = os.path.join(BUILD_DIR, "narration_full.wav")
    score_wav = os.path.join(BUILD_DIR, "score.wav")

    v_inputs = []
    v_filters = []
    labels = []
    for i, wav in enumerate(voice_wavs):
        v_inputs.extend(["-i", wav])
        delay_ms = int(round((scene_starts[i] + lead_ins[i]) * 1000.0))
        lbl = f"[a{i}]"
        v_filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}{lbl}")
        labels.append(lbl)

    v_filters.append(
        f"{''.join(labels)}amix=inputs={len(voice_wavs)}:normalize=0,"
        f"apad,atrim=0:{total_dur:.4f}[vtrack]"
    )
    run([
        "ffmpeg", "-y", *v_inputs,
        "-filter_complex", ";".join(v_filters),
        "-map", "[vtrack]", "-ar", "44100", "-ac", "2", narration_wav,
    ])

    print("  ♪ Composing original score & soundscape...", flush=True)
    make_score.generate_score(total_dur, scene_starts, score_wav)

    mix_filter = (
        "[1:a]asplit=2[voice][sc];"
        "[2:a][sc]sidechaincompress=threshold=0.015:ratio=3.2:attack=45:release=500:makeup=1[ducked];"
        "[ducked][voice]amix=inputs=2:duration=first:weights=0.92 1.18:normalize=0,"
        "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
    )
    run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", narration_wav,
        "-i", score_wav,
        "-filter_complex", mix_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        out_mp4,
    ])


def main() -> None:
    t_start = time.time()
    os.makedirs(BUILD_DIR, exist_ok=True)
    print("🎬 Building cinema-grade short film: 'Pip and the Lost Key' ...", flush=True)

    for pic, *_ in SCENES:
        if pic is not None:
            p = os.path.join(SCENES_DIR, pic)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cinema artwork: {p}")

    print("  🎙 Generating neural narration...", flush=True)
    voice_wavs: list[str] = []
    voice_durs: list[float] = []
    lead_ins: list[float] = []
    scene_durs: list[float] = []

    for i, (pic, subtitle, spoken, *_) in enumerate(SCENES):
        text = spoken or subtitle
        wav, v_dur = synth_narration(i, text)
        voice_wavs.append(wav)
        voice_durs.append(v_dur)

        if i == 0:
            lead = 1.15
            dur = max(6.2, lead + v_dur + 1.85)
        elif pic is None:
            lead = 1.35
            dur = max(7.2, lead + v_dur + 4.4)
        else:
            lead = 1.25
            dur = lead + v_dur + 1.75

        lead_ins.append(lead)
        scene_durs.append(dur)
        print(f"    • Scene {i}: voice {v_dur:.2f}s -> clip {dur:.2f}s", flush=True)

    print("  ✨ Rendering animated 2.39:1 cinema scenes...", flush=True)
    vignette = make_vignette_lut()
    title_ovs = make_title_overlays()
    credits_assets = make_credits_bg_and_overlay()

    clips: list[str] = []
    frame_counts: list[int] = []
    for i, (pic, subtitle, _, cam, p_style) in enumerate(SCENES):
        clip_path, n_frames = render_scene_video(
            i, pic, subtitle, cam, p_style, scene_durs[i], lead_ins[i], voice_durs[i],
            vignette, title_ovs, credits_assets,
        )
        clips.append(clip_path)
        frame_counts.append(n_frames)
        print(f"    ✓ Scene {i} rendered ({n_frames} frames)", flush=True)

    print("  🎞 Assembling xfade transitions & film grade...", flush=True)
    video_full = os.path.join(BUILD_DIR, "video_full.mp4")
    scene_starts, total_dur = assemble_video_with_xfade(clips, frame_counts, video_full)

    print("  🔊 Mixing voice, score & mastering...", flush=True)
    mix_audio_and_mux(video_full, voice_wavs, scene_starts, lead_ins, total_dur, OUT_MP4)

    size_mb = os.path.getsize(OUT_MP4) / 1e6
    elapsed = time.time() - t_start
    print(f"✅ Finished in {elapsed:.1f}s -> {OUT_MP4} ({total_dur:.1f}s, {size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
