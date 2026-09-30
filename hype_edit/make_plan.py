"""Hand-picked plan (the clips are vertical phone video, so cx is ignored: full frame is used) for the 9/28 flag football footage (5 phone clips joined into game.mp4).

Moments were chosen by eye from a 2 fps contact sheet of the whole game; cx is the
horizontal centre of the action (0 = left edge, 1 = right edge of the landscape frame).
"""
import json
import sys

beats = json.load(open(sys.argv[1]))
B, bar, drop, end_hit = beats["beat"], beats["bar"], beats["drop"], beats["end_hit"]

builds = [(14.0, 0.50), (44.6, 0.62), (57.0, 0.74)]  # wide stadium, sunset tree, hero in stance
# impacts in drop order: build up to the big play (31.0 = diving flag pull)
impacts = [(23.5, 0.6), (9.8, 0.4), (11.6, 0.68), (50.4, 0.45), (61.9, 0.6), (68.4, 0.36),
           (73.4, 0.35), (30.45, 0.7)]
quicks = [(1.4, .5), (4.9, .5), (5.6, .45), (8.9, .5), (9.6, .5), (10.5, .5), (11.3, .75), (12.2, .6),
          (19.0, .45), (21.0, .5), (23.5, .6), (25.5, .5), (27.4, .45), (29.5, .5), (30.3, .55),
          (31.7, .55), (32.3, .5), (32.0, .5), (38.5, .5), (40.6, .6), (47.6, .6), (49.5, .5),
          (50.8, .4), (51.2, .55), (52.6, .45), (53.3, .5), (54.0, .45), (55.6, .6), (56.4, .72),
          (58.6, .65), (60.2, .62), (61.5, .6), (65.0, .45), (66.4, .5), (67.5, .45), (69.2, .35),
          (71.0, .45), (73.1, .3), (73.7, .35), (74.4, .4), (60.8, .6), (30.6, .6), (53.6, .5),
          (9.9, .4), (85.2, .8), (86.6, .8), (88.0, .75), (51.0, .45)]

clips = []
for k, (t, cx) in enumerate(builds):
    clips.append({"kind": "build", "start": k * 2 * bar, "dur": 2 * bar, "src": t, "speed": 0.5, "cx": cx})
clips[-1]["dur"] -= B
clips.append({"kind": "title", "start": drop - B, "dur": B})  # black silent beat

qi = 0
for g, (it, icx) in enumerate(impacts):
    gt = drop + g * 8 * B
    clips.append({"kind": "impact", "start": gt, "dur": 3 * B, "src": it, "cx": icx, "contact": 2 * B})
    cur = gt + 3 * B
    for nb in ([1, 1, 0.5, 0.5, 1, 1] if g % 2 == 0 else [1, 0.5, 0.5, 1, 1, 1]):
        t, cx = quicks[qi % len(quicks)]
        qi += 1
        clips.append({"kind": "quick", "start": cur, "dur": nb * B, "src": t,
                      "speed": 1.25 if nb < 1 else 1.0, "cx": cx})
        cur += nb * B
clips.append({"kind": "final", "start": end_hit, "dur": beats["duration"] - end_hit, "src": 32.25, "cx": 0.5})
json.dump({"mode": "day", "impacts": [i[0] for i in impacts], "builds": [b[0] for b in builds],
           "clips": clips}, open(sys.argv[2], "w"), indent=1)
print(len(clips), "clips")
