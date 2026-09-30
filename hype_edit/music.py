"""Synthesize an original 150 BPM hardstyle track so the edit never gets muted.

Everything here is generated from scratch (no samples), so there is no
copyright claim possible. Writes track.wav and track.json (beat grid + drop).

Structure (1 bar = 4 beats = 1.6 s):
  bars 0-5   buildup: dark pad, heartbeat thumps, snare roll + riser
  bar  5     last beat silent (the gap right before the drop)
  bars 6-21  drop: distorted kick every beat, reverse bass, supersaw lead
  bar  22    final hit + tail
"""
import json
import sys

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
BPM = 150
BEAT = 60 / BPM
BAR = 4 * BEAT
BUILD_BARS = 6
DROP_BARS = 16
TAIL = 3.2  # makes the whole edit exactly 40.0 s
TOTAL = (BUILD_BARS + DROP_BARS + 1) * BAR + TAIL
DROP_T = BUILD_BARS * BAR
END_HIT_T = (BUILD_BARS + DROP_BARS) * BAR

rng = np.random.default_rng(928)


def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += gain * sig[: j - i]


def lowpass(x, fc, order=4):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def highpass(x, fc, order=4):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def saw(freq, t, phase=0.0):
    p = (freq * t + phase) % 1.0
    return 2 * p - 1


def supersaw(freq, t, voices=7, detune=0.012):
    out = np.zeros_like(t)
    for k in range(voices):
        d = 1 + detune * (k - voices // 2) / (voices // 2)
        out += saw(freq * d, t, rng.random())
    return out / voices


def hz(note):
    """MIDI note number to Hz."""
    return 440.0 * 2 ** ((note - 69) / 12)


def reverb(x, decay=1.8, mix=0.25):
    n = int(decay * SR)
    ir = rng.standard_normal(n) * np.exp(-6 * np.arange(n) / n)
    ir = lowpass(ir, 6000)
    wet = fftconvolve(x, ir)[: len(x)]
    wet /= np.max(np.abs(wet)) + 1e-9
    return (1 - mix) * x + mix * wet * np.max(np.abs(x))


# ---------------------------------------------------------------- instruments
def hard_kick(dur=0.38):
    t = t_axis(dur)
    f = 52 + 900 * np.exp(-t * 45)  # pitch sweep down
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 3.2)
    click = rng.standard_normal(len(t)) * np.exp(-t * 400) * 0.6
    k = np.tanh(6.0 * (body + click))  # the hardstyle distortion
    k = lowpass(k, 7000)
    k *= np.minimum(1, (dur - t) * 60)  # declick tail
    return k


def reverse_bass(freq, dur=BEAT * 0.6):
    t = t_axis(dur)
    env = (t / dur) ** 1.6 * np.minimum(1, (dur - t) * 80)
    x = np.tanh(3 * (saw(freq, t) + 0.5 * np.sin(2 * np.pi * freq * 2 * t)))
    return lowpass(x, 900) * env


def clap(dur=0.25):
    t = t_axis(dur)
    n = rng.standard_normal(len(t))
    env = np.zeros_like(t)
    for off in (0, 0.011, 0.022):
        env += np.where(t >= off, np.exp(-(t - off) * 60), 0)
    return highpass(n * env, 900)


def hat(dur=0.06):
    t = t_axis(dur)
    return highpass(rng.standard_normal(len(t)), 8000) * np.exp(-t * 80)


def snare(dur=0.18):
    t = t_axis(dur)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    noise = highpass(rng.standard_normal(len(t)), 1500) * np.exp(-t * 22)
    return 0.5 * tone + noise


def thump(dur=0.5):
    t = t_axis(dur)
    f = 45 + 80 * np.exp(-t * 20)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)


def impact(dur=2.4):
    t = t_axis(dur)
    f = 35 + 300 * np.exp(-t * 18)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    crash = highpass(rng.standard_normal(len(t)), 3000) * np.exp(-t * 2.2)
    return np.tanh(3 * boom) + 0.35 * crash


# ------------------------------------------------------------------- arrange
def build():
    n = int(TOTAL * SR)
    drums = np.zeros(n)
    bass = np.zeros(n)
    music = np.zeros(n)
    fx = np.zeros(n)

    # Am - F - C - G (roots, octave 3) — dark epic minor loop, one chord per bar
    prog = [57, 53, 48, 55]
    chord_shapes = {57: [0, 3, 7], 53: [0, 4, 7], 48: [0, 4, 7], 55: [0, 4, 7]}

    # buildup pad
    t_build = t_axis(DROP_T)
    pad = np.zeros_like(t_build)
    for b in range(BUILD_BARS):
        root = prog[b % 4]
        seg = t_axis(BAR)
        chord = sum(supersaw(hz(root - 12 + iv), seg, 5, 0.008) for iv in chord_shapes[root])
        env = np.minimum(1, seg / 0.3) * np.minimum(1, (BAR - seg) / 0.05)
        i = int(b * BAR * SR)
        pad[i : i + len(seg)] += chord * env
    pad = lowpass(pad, 1400) * np.linspace(0.25, 0.7, len(pad))
    music[: len(pad)] += pad

    # heartbeat thumps (beats 1 and 1-and-a-half) through the buildup
    for b in range(BUILD_BARS - 1):
        for beat in (0, 2):
            place(drums, thump(), b * BAR + beat * BEAT, 0.9)
            place(drums, thump(), b * BAR + beat * BEAT + 0.18, 0.5)

    # snare roll accelerating over the last 2 build bars, riser over 3 bars
    roll_start = (BUILD_BARS - 2) * BAR
    roll_end = DROP_T - BEAT
    tt = roll_start
    while tt < roll_end:
        prog_frac = (tt - roll_start) / (roll_end - roll_start)
        place(drums, snare(), tt, 0.25 + 0.6 * prog_frac)
        tt += BEAT / (1 if prog_frac < 0.25 else 2 if prog_frac < 0.5 else 4 if prog_frac < 0.75 else 8)
    rs = t_axis(3 * BAR - BEAT)
    riser = highpass(rng.standard_normal(len(rs)), 400) * (rs / rs[-1]) ** 2
    riser += 0.4 * saw(200 * 2 ** (3 * rs / rs[-1]), rs) * (rs / rs[-1]) ** 2
    place(fx, riser, DROP_T - 3 * BAR, 0.35)

    # drop
    kick = hard_kick()
    sidechain = np.ones(n)
    duck_len = int(0.22 * SR)
    duck = 0.15 + 0.85 * (np.arange(duck_len) / duck_len) ** 0.6
    lead_notes = [69, 72, 76, 74, 72, 71, 72, 67]  # 8th-note hook in A minor
    for b in range(DROP_BARS):
        bar_t = DROP_T + b * BAR
        root = prog[b % 4]
        for beat in range(4):
            bt = bar_t + beat * BEAT
            place(drums, kick, bt, 1.0)
            i = int(bt * SR)
            sidechain[i : i + duck_len] = np.minimum(sidechain[i : i + duck_len], duck[: max(0, min(duck_len, n - i))])
            place(bass, reverse_bass(hz(root - 24)), bt + BEAT * 0.4, 0.55)
            place(drums, hat(), bt + BEAT / 2, 0.25)
            if beat in (1, 3):
                place(drums, clap(), bt, 0.55)
        # supersaw lead: 8 eighth notes per bar, transposed with the chord
        shift = {57: 0, 53: -4, 48: -9, 55: -2}[root]
        for k, note in enumerate(lead_notes):
            nt = bar_t + k * BEAT / 2
            seg = t_axis(BEAT / 2 * 0.95)
            env = np.minimum(1, seg / 0.005) * np.exp(-seg * 3)
            place(music, supersaw(hz(note + shift), seg) * env, nt, 0.42)
        # chord stabs under the lead
        seg = t_axis(BAR)
        chord = sum(supersaw(hz(root - 12 + iv), seg, 5) for iv in chord_shapes[root])
        place(music, lowpass(chord, 2500) * 0.35, bar_t)
        if b % 4 == 3:  # crash every 4 bars
            place(fx, impact(1.2)[: int(1.2 * SR)] * 0.5, bar_t + BAR, 0.5)
    place(fx, impact(1.6), DROP_T, 0.9)

    # final hit
    place(drums, kick, END_HIT_T, 1.0)
    place(fx, impact(), END_HIT_T, 1.0)

    music = reverb(music * sidechain, 1.6, 0.3)
    bass = bass * sidechain
    mix = 0.95 * drums + 0.8 * bass + 0.55 * music + 0.6 * fx
    mix = highpass(mix, 28, 2)
    mix = np.tanh(1.6 * mix / (np.percentile(np.abs(mix), 99.5) + 1e-9))  # master glue/limit
    mix *= 0.93 / np.max(np.abs(mix))
    fade = int(1.5 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    return mix


def main(out_wav="track.wav", out_json="track.json"):
    from scipy.io import wavfile

    mix = build()
    stereo = np.stack([mix, np.roll(mix, 17)], axis=1)  # tiny width
    wavfile.write(out_wav, SR, (stereo * 32767).astype(np.int16))
    beats = [round(i * BEAT, 4) for i in range(int(TOTAL / BEAT))]
    json.dump(
        {"bpm": BPM, "beat": BEAT, "bar": BAR, "drop": DROP_T, "end_hit": END_HIT_T,
         "duration": TOTAL, "beats": beats},
        open(out_json, "w"), indent=1,
    )
    print(f"wrote {out_wav} ({TOTAL:.1f}s, drop at {DROP_T:.1f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:])
