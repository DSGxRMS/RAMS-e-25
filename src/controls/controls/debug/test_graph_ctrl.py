"""Verify CUDA-graph MPPIController: speed + steer correctness."""
import sys, time, math
sys.path.insert(0, "/mnt/d/RAMS-e-25/src/controls")
import torch
from controls.mppi_controller import MPPIController

m = MPPIController()
path = [(float(i), 0.0) for i in range(15)]   # straight path along +x

def warm_and_time(start_y, label):
    m.update_path(path)
    m.cur = 0; m.nom.zero_(); m.I_spd = 0.0
    m._pose_window.clear(); m.prev_yaw = None; m.yaw_rate = 0.0
    # warmup (first call captures the graph)
    for _ in range(8):
        s, a, info = m.compute(0.0, start_y, 0.0, 1.5, 0.05)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    s, a, info = m.compute(0.0, start_y, 0.0, 1.5, 0.05)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = (time.perf_counter() - t0) * 1000
    print(f"{label}: steer={s:+.4f}rad ({math.degrees(s):+.1f}deg) "
          f"CTE={info['cte']:+.2f} accel={a:+.2f} | {dt:.1f}ms")
    return s, dt

print("\n--- timing + correctness ---")
sL, dt = warm_and_time(2.0,  "Off LEFT  (y=+2)")    # north of path -> steer right (neg)
sR, _  = warm_and_time(-2.0, "Off RIGHT (y=-2)")    # south of path -> steer left (pos)
s0, _  = warm_and_time(0.0,  "On path   (y= 0)")

ok_speed = dt < 50
ok_dir   = sL < -0.02 and sR > 0.02 and abs(s0) < 0.15
print(f"\nSPEED: {'PASS' if ok_speed else 'FAIL'} ({dt:.1f}ms)   "
      f"STEER DIR: {'PASS' if ok_dir else 'FAIL'} "
      f"(L={sL:+.3f} R={sR:+.3f} mid={s0:+.3f})")
