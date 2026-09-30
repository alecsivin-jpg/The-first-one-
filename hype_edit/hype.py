"""Render a vertical 9:16 football hype edit from one long game clip.

    python hype.py GAME.mov --music track.wav --beats track.json --out hype.mp4

Pipeline:
  1. analyze   scan the whole game at low res: camera-compensated motion,
               white-jersey coverage, brightness (day/night)
  2. plan      pick 3 calm buildup shots, big-impact moments and quick cuts,
               lay them on the beat grid (written to plan.json; pass --plan
               to re-render from an edited plan)
  3. render    per frame: crop to 9:16, orange/teal grade with white-jersey
               players lit and everyone else crushed dark and cold, flames +
               embers pouring off the white jerseys, and on every impact:
               0.3x ramp into contact / 2x out, zoom punch, white flash frame,
               shake, RGB split, ground cracks, shockwave and debris.
  4. mux       music only; the game audio is never used.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", message="The frame size for reading")

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# output size: 1080x1920 vertical by default, HYPE_SIZE=1920x1080 for widescreen
OW, OH = (int(v) for v in os.environ.get("HYPE_SIZE", "1080x1920").split("x"))
FPS = 30
LW, LH = OW // 4, OH // 4  # low-res working size for fire/particles/masks
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rng = np.random.default_rng(928)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ============================================================ hero detection
_yolo = None


def white_players(rgb, weights):
    """Full-res mask of people wearing white/light shirts (the heroes).

    A person-segmentation model finds every player; each one's torso is then checked:
    mostly bright + unsaturated = white shirt. Black shirts and the ref's stripes fail.
    """
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(weights)
    H, W = rgb.shape[:2]
    r = _yolo.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=1280, conf=0.15, classes=[0],
                      verbose=False, retina_masks=True)[0]
    out = np.zeros((H, W), np.float32)
    if r.masks is None:
        return out
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for mk, box in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        mk = cv2.resize(mk, (W, H)) > 0.5
        x0, y0, x1, y1 = box.astype(int)
        h = y1 - y0
        torso = np.zeros_like(mk)
        torso[y0 + int(0.18 * h): y0 + int(0.5 * h), x0:x1] = True
        t = mk & torso
        if t.sum() < 30:
            continue
        s, v = hsv[..., 1][t].astype(np.float32), hsv[..., 2][t].astype(np.float32)
        # sunset light tints white shirts warm, so allow some saturation; brightness is the key
        whitish = ((v > 150) & (s < 125)).mean()
        darkish = (v < 120).mean()  # ref stripes / black shirts always have dark pixels
        stripes = np.abs(np.diff(gray, axis=1, append=0))[t].mean()
        if whitish > 0.45 and darkish < 0.12 and stripes < 6:
            out[mk] = 1.0
    return out


# =================================================================== analysis
def probe(src):
    gen = imageio_ffmpeg.read_frames(src)
    meta = next(gen)
    gen.close()
    return meta


def white_mask(rgb_small, k=2):
    """Mask of white-jersey pixels on a small RGB frame, with yard lines/sky removed."""
    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    m = ((v > 165) & (s < 60)).astype(np.uint8)
    # opening wipes out anything thinner than k px (painted lines, hash marks)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    area_max = 0.04 * m.size
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        fill = a / float(w * h)
        if 4 <= a <= area_max and fill > 0.3 and min(w, h) >= k and 0.25 < h / max(w, 1) < 5:
            keep[i] = True
    # white numbers on the opponents' dark jerseys are small blobs; a white jersey is
    # a big one. Drop anything much smaller than the biggest white body in frame.
    # floodlights / sun glare: clipped-bright blobs high in the frame are not players
    for i in np.nonzero(keep)[0]:
        x, y, w, h, a = st[i]
        cy = (y + h / 2) / m.shape[0]
        if cy < 0.3 or (cy < 0.5 and v[lab == i].mean() > 225):  # stadium lights / sky glare
            keep[i] = False
    if keep.any():
        biggest = st[keep, 4].max()
        keep &= st[:, 4] >= max(3 * k * k, 0.2 * biggest)
    return keep[lab].astype(np.uint8)


def analyze(src, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    meta = probe(src)
    dur = meta["duration"]
    step_fps = 6
    log(f"analyzing {dur:.0f}s of footage at {step_fps} fps ...")
    gen = imageio_ffmpeg.read_frames(
        src, output_params=["-vf", f"fps={step_fps},scale=320:-2"], pix_fmt="rgb24")
    m = next(gen)
    w, h = m["size"] if "scale" not in m else m["size"]
    # the output size isn't in meta when filtered; detect from the first frame
    out = {"t": [], "motion": [], "hero_motion": [], "white": [], "cx": [], "bright": [], "pan": []}
    prev = None
    fw = 320
    fh = None
    for i, buf in enumerate(gen):
        if fh is None:
            fh = len(buf) // (fw * 3)
        f = np.frombuffer(buf, np.uint8).reshape(fh, fw, 3)
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
        wm = white_mask(f)
        t = i / step_fps
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            warped = cv2.warpAffine(prev, M, (fw, fh), borderMode=cv2.BORDER_REFLECT)
            d = np.abs(g - warped)[12:-12, 12:-12]
            mot = float((d > 18).mean())
            wmc = cv2.dilate(wm, np.ones((9, 9), np.uint8))[12:-12, 12:-12]
            hm = float((d * wmc).sum() / (wmc.sum() * 255 + 1e-6)) if wmc.sum() > 20 else 0.0
            pan = float(math.hypot(dx, dy))
        else:
            mot, hm, pan = 0.0, 0.0, 0.0
        ys, xs = np.nonzero(wm)
        out["t"].append(t)
        out["motion"].append(mot)
        out["hero_motion"].append(hm)
        out["white"].append(float(wm.mean()))
        out["cx"].append(float(xs.mean() / fw) if len(xs) > 8 else 0.5)
        out["bright"].append(float(g.mean() / 255))
        out["pan"].append(pan)
        prev = g
        if i % 600 == 0:
            log(f"  {t:.0f}s")
    json.dump(out, open(cache, "w"))
    return out


def smooth(x, k):
    k = max(1, int(k))
    return np.convolve(x, np.ones(k) / k, mode="same")


# ======================================================================= plan
def build_plan(an, beats, src_dur, mode, joins=()):
    t = np.array(an["t"])
    mot = np.array(an["motion"])
    hm = np.array(an["hero_motion"])
    wh = np.array(an["white"])
    cx = np.array(an["cx"])
    pan = np.array(an["pan"])
    bright = float(np.median(an["bright"]))
    if mode == "auto":
        mode = "night" if bright < 0.33 else "day"

    whn = wh / (np.percentile(wh, 95) + 1e-9)
    act = smooth(mot, 2) + 0.6 * smooth(hm, 2)
    # an impact is a sudden jump in action, not sustained motion
    jump = np.maximum(0, act - smooth(act, 12))
    score = (0.6 * act + jump) * (0.35 + np.clip(whn, 0, 1)) * (pan < np.percentile(pan, 97))
    valid = (t > 1.0) & (t < src_dur - 1.5)
    for j in joins:  # never pick a moment whose source window crosses a clip join
        valid &= (t < j - 1.8) | (t > j + 0.5)
    score = np.where(valid, score, 0)

    def center(ts, span=0.8):
        sel = (t >= ts - span / 2) & (t <= ts + span / 2) & (wh > 0)
        return float(np.average(cx[sel], weights=wh[sel])) if sel.any() else 0.5

    used = []

    def free(ts, gap):
        return all(abs(ts - u) >= gap for u in used)

    # ---- structure on the beat grid
    B = beats["beat"]
    drop = beats["drop"]
    end_hit = beats["end_hit"]
    drop_beats = int(round((end_hit - drop) / B))
    groups = drop_beats // 8

    order = np.argsort(-score)
    impacts = []
    for i in order:
        if len(impacts) >= groups or score[i] <= 0:
            break
        if free(t[i], 5.0):
            impacts.append(float(t[i]))
            used.append(float(t[i]))
    impacts.sort()

    quick_needed = drop_beats - 0  # upper bound, trimmed below
    quicks = []
    for i in order:
        if len(quicks) >= quick_needed or score[i] <= 0:
            break
        if free(t[i], 1.6):
            quicks.append(float(t[i]))
            used.append(float(t[i]))
    rng.shuffle(quicks)

    # calm buildup shots: low motion, low pan, white players visible; spread out
    calm = np.where(valid, (1.0 - smooth(act, 12) / (act.max() + 1e-9)) * np.clip(whn, 0.1, 1)
                    * (smooth(pan, 12) < np.percentile(pan, 60)), 0)
    builds = []
    for i in np.argsort(-calm):
        if len(builds) >= 3:
            break
        if calm[i] > 0 and free(t[i], 6.0) and all(abs(t[i] - b) > 20 for b in builds):
            builds.append(float(t[i]))
            used.append(float(t[i]))
    builds.sort()
    while len(builds) < 3:
        builds.append(float(src_dur * (0.2 + 0.3 * len(builds))))

    clips = []
    # buildup: 3 x 2 bars at 0.5x, then the silent beat is a title card
    bar = beats["bar"]
    for k, bt in enumerate(builds):
        clips.append({"kind": "build", "start": k * 2 * bar, "dur": 2 * bar,
                      "src": bt, "speed": 0.5, "cx": center(bt, 2)})
    clips[-1]["dur"] -= B
    clips.append({"kind": "title", "start": drop - B, "dur": B})

    # drop: per 8-beat group -> impact (3 beats) + quick cuts (1,1,.5,.5,1,1)
    qi = 0
    pattern = [1, 1, 0.5, 0.5, 1, 1]
    alt = [1, 0.5, 0.5, 1, 1, 1]
    for g in range(groups):
        gt = drop + g * 8 * B
        if g < len(impacts):
            it = impacts[g]
            clips.append({"kind": "impact", "start": gt, "dur": 3 * B, "src": it,
                          "cx": center(it), "contact": 2 * B})
            cur, pat = gt + 3 * B, (pattern if g % 2 == 0 else alt)
        else:
            cur, pat = gt, pattern + [1, 1]
        for nb in pat:
            src_t = quicks[qi % len(quicks)] if quicks else float(src_dur / 2)
            flip = qi >= len(quicks)
            qi += 1
            clips.append({"kind": "quick", "start": cur, "dur": nb * B, "src": src_t - 0.15,
                          "speed": 1.25 if nb < 1 else 1.0, "cx": center(src_t), "flip": flip})
            cur += nb * B
    # final hit: freeze on the strongest impact, title, fade
    best = impacts[int(np.argmax([score[np.searchsorted(t, x)] for x in impacts]))] if impacts else quicks[0]
    clips.append({"kind": "final", "start": end_hit, "dur": beats["duration"] - end_hit,
                  "src": best, "cx": center(best)})
    return {"mode": mode, "brightness": bright, "impacts": impacts, "builds": builds, "clips": clips}


def refine_contact(src_path, t, win=0.8):
    """Snap an impact to the exact frame of the biggest motion jump (full frame rate)."""
    t0 = max(0.0, t - win)
    gen = imageio_ffmpeg.read_frames(src_path, input_params=["-ss", f"{t0:.3f}"],
                                     output_params=["-t", f"{2 * win:.3f}", "-vf", "scale=320:-2"])
    meta = next(gen)
    fps = meta["fps"]
    prev, best, best_i, fh = None, -1.0, 0, None
    for i, buf in enumerate(gen):
        fh = fh or len(buf) // (320 * 3)
        g = cv2.GaussianBlur(cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(fh, 320, 3),
                                          cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            w = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (320, fh),
                               borderMode=cv2.BORDER_REFLECT)
            d = float((np.abs(g - w)[12:-12, 12:-12] > 18).mean())
            if d > best:
                best, best_i = d, i
        prev = g
    return t0 + best_i / fps


# ====================================================================== video
class Source:
    def __init__(self, path):
        self.path = path
        meta = probe(path)
        self.w, self.h = meta["size"]
        self.fps = meta["fps"]
        self.dur = meta["duration"]

    def crop_box(self, cx):
        if self.w / self.h > OW / OH:
            ch = self.h - self.h % 2
            cw = int(ch * OW / OH) // 2 * 2
            x = int(np.clip(cx * self.w - cw / 2, 0, self.w - cw))
            return cw, ch, x, 0
        cw = self.w - self.w % 2
        ch = min(self.h, int(cw * OH / OW)) // 2 * 2
        return cw, ch, 0, (self.h - ch) // 2

    def read(self, t0, dur, cx):
        t0 = max(0.0, t0)
        cw, ch, x, y = self.crop_box(cx)
        gen = imageio_ffmpeg.read_frames(
            self.path, input_params=["-ss", f"{t0:.3f}"],
            output_params=["-t", f"{dur:.3f}", "-vf",
                           f"crop={cw}:{ch}:{x}:{y},scale={OW}:{OH}:flags=bicubic"],
            pix_fmt="rgb24")
        next(gen)
        frames = [np.frombuffer(b, np.uint8).reshape(OH, OW, 3) for b in gen]
        if not frames:
            frames = [np.zeros((OH, OW, 3), np.uint8)]
        return frames


_dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)


def interp(frames, fi):
    """Frame at fractional index fi using optical-flow interpolation."""
    fi = float(np.clip(fi, 0, len(frames) - 1))
    a = int(math.floor(fi))
    b = min(a + 1, len(frames) - 1)
    w = fi - a
    if b == a or w < 0.08:
        return frames[a]
    if w > 0.92:
        return frames[b]
    ga = cv2.cvtColor(cv2.resize(frames[a], (OW // 2, OH // 2)), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(cv2.resize(frames[b], (OW // 2, OH // 2)), cv2.COLOR_RGB2GRAY)
    flow = cv2.resize(_dis.calc(ga, gb, None), (OW, OH)) * 2
    gx, gy = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
    wa = cv2.remap(frames[a], gx - w * flow[..., 0], gy - w * flow[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)
    wb = cv2.remap(frames[b], gx + (1 - w) * flow[..., 0], gy + (1 - w) * flow[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(wa, 1 - w, wb, w, 0)


# ====================================================================== looks
def fire_lut():
    stops = [(0, (0, 0, 0)), (0.22, (0.35, 0.02, 0.0)), (0.45, (0.95, 0.25, 0.02)),
             (0.7, (1.0, 0.62, 0.12)), (0.9, (1.0, 0.9, 0.55)), (1.0, (1.0, 1.0, 0.9))]
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, [s[0] for s in stops], [s[1][c] for s in stops])
    return lut


FIRE = fire_lut()
GX, GY = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
LX, LY = np.meshgrid(np.arange(LW, dtype=np.float32), np.arange(LH, dtype=np.float32))
VIG = np.clip(1.25 - 0.95 * (((GX - OW / 2) / (OW * 0.62)) ** 2 + ((GY - OH / 2) / (OH * 0.6)) ** 2),
              0.45, 1)[..., None].astype(np.float32)


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def up(x, interp=cv2.INTER_LINEAR):
    return cv2.resize(x, (OW, OH), interpolation=interp)


class Noise:
    """Scrolling smooth noise fields for fire turbulence and heat haze."""

    def __init__(self):
        base = rng.standard_normal((LH * 2, LW)).astype(np.float32)
        self.a = cv2.GaussianBlur(base, (0, 0), 6)
        self.a /= self.a.std()
        base = rng.standard_normal((LH * 2, LW)).astype(np.float32)
        self.b = cv2.GaussianBlur(base, (0, 0), 3)
        self.b /= self.b.std()

    def at(self, f, speed=6):
        o = (f * speed) % LH
        return self.a[o:o + LH], self.b[o:o + LH]


NOISE = Noise()


class Particles:
    def __init__(self):
        self.p = np.zeros((0, 2), np.float32)
        self.v = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.age = np.zeros(0, np.float32)
        self.kind = np.zeros(0, np.int8)  # 0 ember, 1 debris, 2 spark

    def spawn(self, pos, vel, life, kind):
        self.p = np.vstack([self.p, pos.astype(np.float32)])
        self.v = np.vstack([self.v, vel.astype(np.float32)])
        self.life = np.concatenate([self.life, life.astype(np.float32)])
        self.age = np.concatenate([self.age, np.zeros(len(life), np.float32)])
        self.kind = np.concatenate([self.kind, np.full(len(life), kind, np.int8)])

    def step(self, dt, wind):
        k = self.kind
        self.v[:, 1] += np.where(k == 0, -260, 1500) * dt  # embers rise, debris falls
        self.v[:, 0] += wind * dt + rng.standard_normal(len(k)) * np.where(k == 0, 240, 30) * dt
        self.v *= np.where(k[:, None] == 0, 0.97, 0.985)
        self.p += self.v * dt
        self.age += dt
        keep = (self.age < self.life) & (self.p[:, 1] > -50) & (self.p[:, 1] < OH + 50)
        for n in ("p", "v", "life", "age", "kind"):
            setattr(self, n, getattr(self, n)[keep])

    def render(self):
        """Returns (glow RGB float, debris darkness mask) at full res."""
        hw, hh = OW // 2, OH // 2
        glow = np.zeros((hh, hw), np.float32)
        dark = np.zeros((hh, hw), np.float32)
        if len(self.p):
            x = (self.p[:, 0] / 2).astype(int)
            y = (self.p[:, 1] / 2).astype(int)
            ok = (x >= 1) & (x < hw - 1) & (y >= 1) & (y < hh - 1)
            fade = np.clip(1 - self.age / self.life, 0, 1) * np.clip(self.age * 20, 0, 1)
            for kind, buf, gain in ((0, glow, 1.0), (2, glow, 1.6), (1, dark, 1.0)):
                s = ok & (self.kind == kind)
                np.add.at(buf, (y[s], x[s]), fade[s] * gain)
                if kind != 1:
                    np.add.at(buf, (y[s] + 1, x[s]), fade[s] * gain * 0.6)
        core = cv2.GaussianBlur(glow, (0, 0), 1.2) * 3
        halo = cv2.GaussianBlur(glow, (0, 0), 6) * 14
        g = np.clip(core, 0, 1)[..., None] * np.float32([1, 0.85, 0.5]) + \
            np.clip(halo, 0, 1)[..., None] * np.float32([1, 0.38, 0.05])
        d = np.clip(cv2.GaussianBlur(dark, (0, 0), 2.5) * 6, 0, 0.9)
        return up(g), up(d)


def make_cracks(ox, oy):
    """Branching ground fissures from (ox, oy) in full-res coords (perspective-flattened)."""
    paths = []

    def walk(x, y, ang, length, width, depth):
        pts = [(x, y)]
        seg = 26
        for _ in range(int(length / seg)):
            ang += rng.normal(0, 0.35)
            x += math.cos(ang) * seg
            y += math.sin(ang) * seg * 0.38
            pts.append((x, y))
            if depth < 2 and rng.random() < 0.12:
                walk(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1),
                     length * rng.uniform(0.3, 0.5), width * 0.6, depth + 1)
        paths.append((np.array(pts, np.float32), width))

    for k in range(rng.integers(7, 11)):
        ang = 2 * math.pi * k / 9 + rng.normal(0, 0.3)
        walk(ox, oy, ang, rng.uniform(320, 700), rng.uniform(7, 12), 0)
    return paths


def draw_cracks(paths, reveal):
    m = np.zeros((OH, OW), np.float32)
    for pts, width in paths:
        n = max(2, int(len(pts) * reveal))
        for i in range(n - 1):
            w = max(1, int(width * (1 - i / len(pts)) + 1))
            p1 = tuple(int(v) for v in pts[i])
            p2 = tuple(int(v) for v in pts[i + 1])
            cv2.line(m, p1, p2, 1.0, w, cv2.LINE_AA)
    return m


def text_layer(text, size, glow=True):
    """RGBA float layer with big glowing centered text."""
    img = Image.new("L", (OW, OH), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, size)
    bb = d.textbbox((0, 0), text, font=font)
    d.text(((OW - (bb[2] - bb[0])) / 2 - bb[0], (OH - (bb[3] - bb[1])) / 2 - bb[1]), text,
           font=font, fill=255)
    a = np.asarray(img, np.float32) / 255
    col = a[..., None] * np.float32([1, 0.97, 0.9])
    if glow:
        g = cv2.GaussianBlur(a, (0, 0), 28)[..., None] * np.float32([1, 0.35, 0.02]) * 2.2
        col = screen(col, np.clip(g, 0, 1))
    return col, a


# ===================================================================== render
class Renderer:
    def __init__(self, src, plan, beats, seg=None):
        self.seg = seg
        self.src = src
        self.plan = plan
        self.beats = beats
        self.night = plan["mode"] == "night"
        self.beat_frames = {int(round(b * FPS)) for b in beats["beats"] if b >= beats["drop"]}

    def hero_masks(self, frame):
        if self.seg:
            m = (cv2.resize(white_players(frame, self.seg), (LW, LH), interpolation=cv2.INTER_AREA) > 0.35
                 ).astype(np.float32)
        else:
            small = cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA)
            m = white_mask(small, 3).astype(np.float32)
        aura = cv2.GaussianBlur(cv2.dilate(m, np.ones((9, 9), np.uint8)), (0, 0), 7)
        aura = np.clip(aura * 2.2, 0, 1)
        edge = np.clip(cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
        return m, aura, edge

    def grade(self, img, aura_full, calm):
        """Orange/teal, crushed blacks, heroes warm and lit, everyone else dark and cold."""
        lum = img @ np.float32([0.299, 0.587, 0.114])
        l3 = lum[..., None]
        # S-curve contrast with crushed blacks
        c = np.clip((img - 0.06) / 0.9, 0, 1)
        c = c * c * (3 - 2 * c)
        # split tone: shadows teal, highlights orange
        sh = np.clip(1 - l3 * 2.2, 0, 1)
        hi = np.clip(l3 * 1.8 - 0.7, 0, 1)
        c = c + sh * np.float32([-0.05, 0.035, 0.07]) + hi * np.float32([0.08, 0.02, -0.07])
        # background: desaturate, darken, cool
        gray = (c @ np.float32([0.299, 0.587, 0.114]))[..., None]
        bg = (0.45 * c + 0.55 * gray) * np.float32([0.62, 0.74, 0.88]) * (0.62 if calm else 0.72)
        hero = np.clip(c * np.float32([1.1, 1.02, 0.94]) * 1.06 + 0.02, 0, 1)
        a = aura_full[..., None]
        return np.clip(bg * (1 - a) + hero * a, 0, 1)

    def lighting(self, f, calm):
        """Night: flickering floodlight god-rays. Day: low sun flare and haze."""
        layer = np.zeros((LH, LW), np.float32)
        if self.night:
            for k, (x0, ang) in enumerate(((0.12, 0.5), (0.88, -0.5), (0.5, 0.05))):
                flick = 0.75 + 0.25 * math.sin(f * 0.37 + k * 2.1)
                d = (LX - x0 * LW) * math.cos(ang) - (LY + 10) * math.sin(ang) * 0.35
                beam = np.exp(-(d / (18 + LY * 0.18)) ** 2) * np.exp(-LY / (LH * 0.55))
                layer += beam * 0.35 * flick
            col = np.float32([0.85, 0.92, 1.0])
        else:
            layer = np.zeros((LH, LW), np.float32)
            col = np.float32([0.8, 0.86, 0.95])
        haze = NOISE.at(f, 2)[0] * 0.03 + 0.04
        layer = layer + haze * np.exp(-LY / LH * 2)
        return up(np.clip(layer, 0, 1))[..., None] * col * (0.7 if calm else 1.0)

    def render_clip(self, clip, enc):
        kind = clip["kind"]
        n = int(round(clip["dur"] * FPS))
        start_f = int(round(clip["start"] * FPS))
        if kind == "title":  # the silent beat before the drop: hard black
            for i in range(n):
                enc.write(self.post(np.zeros((OH, OW, 3), np.float32), i, start_f + i, {}))
            return

        # --- source time mapping
        if kind == "impact":
            tc = clip["src"]
            slow_n = int(round(clip["contact"] * FPS))
            fast_n = n - slow_n
            src_t0 = tc - slow_n / FPS * 0.3
            times = [src_t0 + i / FPS * 0.3 for i in range(slow_n)] + \
                    [tc + i / FPS * 2.0 for i in range(fast_n)]
        elif kind == "final":
            times = [clip["src"]] * n
        else:
            sp = clip.get("speed", 1.0)
            times = [clip["src"] + i / FPS * sp for i in range(n)]
        t0 = min(times)
        frames = self.src.read(t0, max(times) - t0 + 2 / self.src.fps, clip["cx"])
        if clip.get("flip"):
            frames = [np.ascontiguousarray(fr[:, ::-1]) for fr in frames]

        calm = kind == "build"
        heat = np.zeros((LH, LW), np.float32)
        parts = Particles()
        cracks = None
        impact_f = int(clip.get("contact", 0) * FPS) if kind == "impact" else (0 if kind == "final" else None)
        state = {}
        for i in range(n):
            gf = start_f + i
            fi = (times[i] - t0) * self.src.fps
            base = interp(frames, fi) if kind in ("impact", "build") else frames[int(np.clip(round(fi), 0, len(frames) - 1))]
            img = base.astype(np.float32) / 255
            m_raw, aura, _ = self.hero_masks(base)
            ema = m_raw if i == 0 else 0.55 * ema + 0.45 * m_raw
            m = (ema > 0.6).astype(np.float32) * m_raw
            k3 = np.ones((3, 3), np.uint8)
            edge = np.clip(cv2.dilate(m, k3) - cv2.erode(m, k3), 0, 1)

            since = i - impact_f if impact_f is not None and i >= impact_f else None
            pre = impact_f is not None and i < impact_f
            if kind == "final":
                since = i

            # heat haze around the heroes
            na, nb = NOISE.at(gf, 9)
            hz = up(cv2.GaussianBlur(aura, (0, 0), 4) * (na * 3.0))
            img = cv2.remap(img, GX + hz, GY + hz * 0.6, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            aura_f = up(aura)
            img = self.grade(img, aura_f, calm)

            # --- fire simulation (flames rise and get torn backward)
            intensity = 0.45 if calm else (1.25 if since is not None and since < 12 else 0.95)
            if pre:
                intensity = 0.7 + 0.5 * i / max(1, impact_f)
            rise = 2.2 if calm else 3.6
            wind = 1.6 + 0.8 * math.sin(gf * 0.05)
            heat = cv2.remap(heat, LX + na * 1.4 - wind * 0.35, LY + rise + nb * 0.6, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            heat *= 0.915 - 0.03 * np.clip(nb, 0, 2)
            flick = 0.75 + 0.25 * np.clip(na, -1, 1)
            heat = np.maximum(heat, (edge * 1.0 + m * 0.22) * flick * intensity)
            fire = FIRE[np.clip(heat * 255, 0, 255).astype(np.uint8)]
            fire = cv2.GaussianBlur(up(fire), (0, 0), 2)
            bloom = up(cv2.GaussianBlur(heat, (0, 0), 10))[..., None] * np.float32([1, 0.35, 0.03]) * 0.9
            img = screen(img, np.clip(fire * (0.75 if calm else 1.0) + bloom, 0, 1))

            # --- embers off the heroes
            ys, xs = np.nonzero(edge > 0)
            if len(xs):
                k = int((6 if calm else 22) * intensity)
                idx = rng.integers(0, len(xs), k)
                pos = np.stack([xs[idx] * OW / LW, ys[idx] * OH / LH], 1) + rng.normal(0, 6, (k, 2))
                vel = np.stack([rng.normal(-60, 90, k), rng.uniform(-420, -120, k)], 1)
                parts.spawn(pos, vel, rng.uniform(0.4, 1.3, k), 0)

            # --- impact: cracks, shockwave, debris burst
            if since == 0:
                n_c, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
                if n_c > 1:
                    # the biggest white player nearest the middle is the one making the hit
                    dist = np.abs(st[1:, 0] + st[1:, 2] / 2 - LW / 2) / LW
                    j = 1 + int(np.argmax(st[1:, 4] * (1.2 - dist)))
                    ox = float((st[j, 0] + st[j, 2] / 2) * OW / LW)
                    oy = float((st[j, 1] + st[j, 3]) * OH / LH) + 15
                else:
                    ox, oy = OW / 2, OH * 0.7
                oy = min(max(oy, OH * 0.45), OH * 0.9)
                state["origin"] = (ox, oy)
                cracks = make_cracks(ox, oy)
                k = 420
                ang = rng.uniform(math.pi * 1.05, math.pi * 1.95, k)
                spd = rng.uniform(500, 2400, k)
                parts.spawn(np.tile([ox, oy], (k, 1)) + rng.normal(0, 25, (k, 2)),
                            np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1),
                            rng.uniform(0.4, 1.2, k), 2)
                k = 160
                ang = rng.uniform(math.pi * 1.15, math.pi * 1.85, k)
                spd = rng.uniform(400, 1500, k)
                parts.spawn(np.tile([ox, oy], (k, 1)) + rng.normal(0, 40, (k, 2)),
                            np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1),
                            rng.uniform(0.5, 1.1, k), 1)
            if cracks is not None:
                reveal = min(1.0, (since + 1) / 5)
                pulse = 0.85 + 0.15 * math.sin(i * 0.9)
                cm = draw_cracks(cracks, reveal)
                glow_s = cv2.resize(cm, (LW, LH), interpolation=cv2.INTER_AREA)
                g1 = up(cv2.GaussianBlur(glow_s, (0, 0), 3)) * 3.0
                g2 = up(cv2.GaussianBlur(glow_s, (0, 0), 10)) * 4.0
                scorch = np.clip(up(cv2.GaussianBlur(glow_s, (0, 0), 5)) * 3, 0, 0.6)[..., None]
                img = img * (1 - scorch)
                lava = cm[..., None] * np.float32([1, 0.95, 0.7]) + \
                    np.clip(g1, 0, 1)[..., None] * np.float32([1, 0.5, 0.05]) + \
                    np.clip(g2, 0, 1)[..., None] * np.float32([0.9, 0.2, 0.0])
                img = screen(img, np.clip(lava * pulse, 0, 1))
                # shockwave ring along the ground plane
                if since < 12:
                    ox, oy = state["origin"]
                    r = 60 + since * 95
                    ring = np.zeros((OH // 2, OW // 2), np.float32)
                    cv2.ellipse(ring, (int(ox / 2), int(oy / 2)), (int(r / 2), int(r * 0.33 / 2)), 0, 0, 360,
                                1.0, 6, cv2.LINE_AA)
                    ring = up(cv2.GaussianBlur(ring, (0, 0), 3)) * (1 - since / 12)
                    img = screen(img, np.clip(ring[..., None] * np.float32([1, 0.8, 0.55]), 0, 1))
                    dust = np.zeros((OH // 4, OW // 4), np.float32)
                    cv2.ellipse(dust, (int(ox / 4), int(oy / 4)), (int(r / 5), int(r / 14)), 0, 0, 360, 1.0, -1)
                    dust = up(cv2.GaussianBlur(dust, (0, 0), 8))[..., None] * 0.35 * (1 - since / 12)
                    img = img * (1 - dust) + dust * np.float32([0.45, 0.4, 0.36])

            parts.step(1 / FPS * (0.3 if pre else 1.0), -120 * wind)
            glow, dark = parts.render()
            img = img * (1 - dark[..., None] * np.float32([1, 1, 1]))
            img = screen(img, np.clip(glow, 0, 1))
            img = screen(img, self.lighting(gf, calm))
            img = np.clip(img * VIG, 0, 1)

            # --- camera: push-in, beat pulse, zoom punch, shake
            fx = {}
            zoom = 1.0
            if calm:
                zoom = 1.0 + 0.1 * i / n
            elif kind == "quick":
                zoom = 1.06 - 0.06 * min(1, i / 4)  # punch-in on every cut
            if gf in self.beat_frames and not calm:
                zoom += 0.025
            elif gf - 1 in self.beat_frames and not calm:
                zoom += 0.012
            shake = 0.0
            if since is not None:
                if since == 0:
                    zoom = max(zoom, 1.075)
                elif since < 12:
                    zoom = max(zoom, 1.15 - 0.15 * min(1, (since - 1) / 11) ** 0.5)
                if since < 9:
                    shake = 42 * (1 - since / 9)
                fx["flash"] = 0.92 if since == 0 else (0.35 if since == 1 else 0)
                fx["rgb"] = [26, 18, 10][since] if since < 3 else 0
                if kind == "final":
                    zoom = max(zoom, 1.0 + 0.05 * i / n)
            elif gf in self.beat_frames and not calm:
                shake = 7
            dx, dy = rng.normal(0, shake, 2) if shake else (0, 0)
            rot = rng.normal(0, shake * 0.03) if shake else 0
            M = cv2.getRotationMatrix2D((OW / 2, OH / 2), rot, zoom)
            M[:, 2] += (dx, dy)
            img = cv2.warpAffine(img, M, (OW, OH), borderMode=cv2.BORDER_REFLECT)

            if kind == "final":
                fade_start = n - int(0.9 * FPS)
                if i > fade_start:
                    img = img * (1 - (i - fade_start) / (n - fade_start))
            if calm:
                bar = int(OH * 0.075)
                img[:bar] = 0
                img[-bar:] = 0
            enc.write(self.post(img, i, gf, fx))

    def post(self, img, i, gf, fx):
        s = fx.get("rgb", 0)
        ca = s if s else 2
        out = img.copy()
        out[..., 0] = np.roll(img[..., 0], ca, axis=1)
        out[..., 2] = np.roll(img[..., 2], -ca, axis=1)
        if s:
            out[..., 0] = np.roll(out[..., 0], s // 3, axis=0)
        fl = fx.get("flash", 0)
        if fl:
            out = out * (1 - fl) + fl
        grain = rng.normal(0, 0.02, (OH // 2, OW // 2)).astype(np.float32)
        out = out + up(grain)[..., None]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


class Encoder:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "20", "-maxrate", "18M", "-bufsize", "36M", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        self.p.stdin.write(frame.tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def contact_sheet(src, plan, path):
    """Thumbnails of every picked moment so a human (or Claude) can sanity-check the plan."""
    tiles = []
    for c in plan["clips"]:
        if "src" not in c or c["kind"] == "quick" and len(tiles) > 30:
            continue
        fr = src.read(c["src"], 0.1, c["cx"])[0]
        th = cv2.resize(fr, (180, 320))
        cv2.putText(th, f'{c["kind"]} {c["src"]:.1f}s', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(th)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + 8]) for r in range(0, len(tiles), 8)]
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--music", required=True)
    ap.add_argument("--beats", required=True)
    ap.add_argument("--out", default="hype.mp4")
    ap.add_argument("--plan", help="re-render from an edited plan.json instead of auto-picking")
    ap.add_argument("--mode", default="auto", choices=["auto", "day", "night"])
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--seg", help="YOLOv8 segmentation weights (e.g. yolov8s-seg.pt) for exact hero masks")
    ap.add_argument("--joins", help="comma-separated times where source clips were joined")
    ap.add_argument("--only", type=int, help="render only this clip index (preview)")
    args = ap.parse_args()

    src = Source(args.source)
    beats = json.load(open(args.beats))
    log(f"source {src.w}x{src.h} @ {src.fps:.2f}fps, {src.dur:.1f}s")
    if args.plan:
        plan = json.load(open(args.plan))
    else:
        an = analyze(args.source, os.path.join(args.workdir, "analysis.json"))
        joins = [float(x) for x in args.joins.split(",")] if args.joins else []
        plan = build_plan(an, beats, src.dur, args.mode, joins)
        for c in plan["clips"]:
            if c["kind"] in ("impact", "final"):
                c["src"] = round(refine_contact(args.source, c["src"]), 3)
        plan["impacts"] = [c["src"] for c in plan["clips"] if c["kind"] == "impact"]
        json.dump(plan, open(os.path.join(args.workdir, "plan.json"), "w"), indent=1)
        contact_sheet(src, plan, os.path.join(args.workdir, "picks.jpg"))
    if args.mode != "auto":
        plan["mode"] = args.mode
    log(f"mode={plan['mode']} impacts={[round(x, 1) for x in plan['impacts']]} clips={len(plan['clips'])}")

    silent = os.path.join(args.workdir, "video_only.mp4")
    enc = Encoder(silent)
    r = Renderer(src, plan, beats, args.seg)
    for k, clip in enumerate(plan["clips"]):
        if args.only is not None and k != args.only:
            continue
        log(f"clip {k + 1}/{len(plan['clips'])} {clip['kind']} @ {clip['start']:.2f}s")
        r.render_clip(clip, enc)
    enc.close()
    # music only: the game audio is never mapped
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", silent, "-i", args.music,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-shortest", "-movflags", "+faststart", args.out], check=True)
    log(f"done -> {args.out} ({enc.n} frames)")


if __name__ == "__main__":
    main()
