#!/usr/bin/env python3
import time

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path as SysPath

# --- Load path points from CSV ---
# CSV should have columns: x, y
CSVPATH = SysPath(__file__).parent.parent / "pp_utils" / "skidpad_path.csv"
df = pd.read_csv(CSVPATH)
xs = df["x"].to_numpy()
ys = df["y"].to_numpy()

# --- Matplotlib setup ---
plt.ion()
fig, ax = plt.subplots()
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Path points (moving trail of last 5)")

xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()
pad = 1.0
if xmax - xmin < 1e-3:
    xmax = xmin + 1.0
if ymax - ymin < 1e-3:
    ymax = ymin + 1.0
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Line for last 5 points
line, = ax.plot([], [], "-o", markersize=4)

# --- Plot points one by one with trail of last 5 ---
trail_x = []
trail_y = []
trail_len = 5

for x, y in zip(xs, ys):
    trail_x.append(x)
    trail_y.append(y)

    # Keep only last 5
    if len(trail_x) > trail_len:
        trail_x = trail_x[-trail_len:]
        trail_y = trail_y[-trail_len:]

    line.set_data(trail_x, trail_y)

    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
