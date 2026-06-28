#!/usr/bin/env python3
"""
plot_mppi_trace.py — visualise an MPPI CSV trace (from MPPI_CSV=...).

Usage:  python3 plot_mppi_trace.py /tmp/mppi_trace.csv
Saves:  <csv>.png  (and shows it if a display is available)

Panels:
  1. Car XY path (coloured by speed) — where did the car actually go?
  2. Speed: odom vs v_plan vs vr_eff (target)  — accel/speed behaviour
  3. Steer command over time
  4. CTE over time
  5. Heading error over time
  6. Cost min/max + n_eff  — is MPPI discriminating between samples?
  7. Rollout time (ms)  — real-time check
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mppi_trace.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    print("empty csv"); sys.exit(1)

def col(name): return np.array([float(r[name]) for r in rows])

t   = col("time"); t = t - t[0]
x   = col("x");      y    = col("y")
odom= col("odom_spd"); vpl = col("v_plan"); vr = col("vr_eff")
steer = col("steer"); accel = col("accel"); lon0 = col("lon0")
cte = col("cte");    herr = col("heading_err")
cmin= col("cost_min"); cmax = col("cost_max"); neff = col("n_eff")
a0  = col("a0_std"); roll = col("roll_ms")

fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. XY path coloured by speed
ax = fig.add_subplot(gs[:, 0])
sc = ax.scatter(x, y, c=odom, cmap="viridis", s=18)
ax.plot(x, y, "-", lw=0.5, alpha=0.4, color="gray")
ax.scatter([x[0]], [y[0]], c="lime", s=120, marker="o", label="start", zorder=5, edgecolor="k")
ax.scatter([x[-1]], [y[-1]], c="red", s=120, marker="X", label="end", zorder=5, edgecolor="k")
ax.set_title("Car path (colour = speed)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3)
fig.colorbar(sc, ax=ax, label="speed m/s", shrink=0.5)

def line(gpos, title, series, ylab):
    a = fig.add_subplot(gpos)
    for lab, dat, st in series:
        a.plot(t, dat, st, label=lab, lw=1.3)
    a.set_title(title); a.set_xlabel("t (s)"); a.set_ylabel(ylab)
    a.legend(fontsize=8); a.grid(alpha=0.3)
    return a

line(gs[0,1], "Speed", [("odom",odom,"-b"),("v_plan",vpl,"--g"),("vr_eff target",vr,":r")], "m/s")
line(gs[1,1], "Steer command", [("steer",steer,"-m")], "rad")
line(gs[2,1], "Accel / lon0", [("accel",accel,"-b"),("lon0",lon0,"--r")], "")
line(gs[0,2], "Cross-track error", [("CTE",cte,"-c")], "m")
line(gs[1,2], "Heading error", [("hd_err",herr,"-")], "rad")
a = line(gs[2,2], "Cost spread + n_eff", [("cost_min",cmin,"-g"),("cost_max",cmax,"-r")], "cost")
a2 = a.twinx(); a2.plot(t, neff, ":k", lw=1, label="n_eff"); a2.set_ylabel("n_eff"); a2.legend(fontsize=7, loc="upper right")

out = path.rsplit(".",1)[0] + ".png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
print(f"\n=== SUMMARY ({len(rows)} cycles) ===")
print(f"roll_ms: mean={roll[1:].mean():.0f} max={roll[1:].max():.0f}  (excl. first warmup={roll[0]:.0f})")
print(f"speed:   max={odom.max():.2f} m/s")
print(f"CTE:     start={cte[0]:+.2f}  end={cte[-1]:+.2f}  max|{np.abs(cte).max():.2f}|")
print(f"steer:   range [{steer.min():+.3f}, {steer.max():+.3f}]")
print(f"a0_std:  mean={a0.mean():.3f}  (sample diversity; >0 = samples differ)")
print(f"n_eff:   mean={neff.mean():.0f}/1024  (low=peaked, ~1024=flat/degenerate)")
print(f"path travelled: ({x[0]:.1f},{y[0]:.1f}) -> ({x[-1]:.1f},{y[-1]:.1f})  dist={np.hypot(x[-1]-x[0],y[-1]-y[0]):.1f}m")
