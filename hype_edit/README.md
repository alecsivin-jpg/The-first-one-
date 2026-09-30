# 9.28 hype edit

Vertical (1080x1920, 30fps) over-the-top hype edit built from one long game clip.

```
pip install numpy scipy opencv-python-headless imageio-ffmpeg pillow
python music.py track.wav track.json          # original 150 BPM hardstyle, drop at 9.6s
python hype.py GAME.mov --music track.wav --beats track.json --out hype_928.mp4 --workdir work
```

- `music.py` synthesizes the soundtrack from scratch (no samples) so Instagram/TikTok can't mute it.
- `hype.py` scans the game for big hits and calm moments, writes `work/plan.json` and a
  `work/picks.jpg` contact sheet, then renders: orange/teal grade, white jerseys lit and on fire,
  opponents crushed dark and cold, embers, and on every hit a 0.3x ramp into contact / 2x out,
  115% zoom punch, white flash frame, shake, RGB split, glowing ground cracks, shockwave and debris.
- Edit `work/plan.json` (swap timestamps) and re-run with `--plan work/plan.json` to change shots.
- `--mode day|night` overrides the automatic day/night lighting pick.
- Game audio is never used; music only.
