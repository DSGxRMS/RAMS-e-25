"""
bench_cudagraph.py — does CUDA-graph capture kill the WSL2 per-kernel-launch
overhead that made the K=1024,T=200 Neural ODE rollout take 3.9 s?

Compares three rollout strategies, all on GPU, all K=1024, T=200, RK4:
  1. plain  : the current per-step Python loop (baseline, ~3.9 s expected)
  2. graphed: capture the whole T-step rollout as ONE CUDA graph, replay it
  3. (sanity) graphed result must equal plain result (same math)

Run inside WSL:
  cd /mnt/d/RAMS-e-25 && source install/setup.bash && python3 src/controls/controls/bench_cudagraph.py
"""
import sys, time, math, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/mnt/d/RAMS-e-25/src/controls")
import torch
import torch.nn as nn
from controls.mppi_controller import _GB, CKPT_PATH, VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I

assert torch.cuda.is_available(), "CUDA not available in WSL"
dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- load model ----
ck = torch.load(str(CKPT_PATH), map_location=dev, weights_only=False)
model = _GB(ck).to(dev).eval()

K = int(os.environ.get("BENCH_K", 1024))
T = int(os.environ.get("BENCH_T", 200))
DT_EFF = 0.02
L_WB = 1.535
STEER_MAX = 0.349
dt_t = torch.tensor(DT_EFF, dtype=torch.float32, device=dev)

# ---- static buffers for graph capture (must be fixed addresses) ----
# state (K,6), control (K,2), per-step steer (K,T), ref (T,4): refE,refN,nE,nN
A_buf    = torch.zeros(K, T, device=dev)            # sampled steer sequences
refE_buf = torch.zeros(T, device=dev)
refN_buf = torch.zeros(T, device=dev)
nE_buf   = torch.zeros(T, device=dev)
nN_buf   = torch.zeros(T, device=dev)
x0_buf   = torch.zeros(6, device=dev)               # initial state (broadcast to K)
p0_buf   = torch.zeros(2, device=dev)               # initial position
cost_buf = torch.zeros(K, device=dev)               # output

W_CT, W_HEAD, CT_SAT = 2.0, 0.5, 3.0

@torch.no_grad()
def rollout(write_cost):
    """One full T-step rollout. Writes final per-sample cost into write_cost (K,)."""
    xk = x0_buf.unsqueeze(0).expand(K, 6).contiguous()   # (K,6)
    pe = p0_buf[0].expand(K).contiguous()
    pn = p0_buf[1].expand(K).contiguous()
    cost = torch.zeros(K, device=dev)
    for t in range(T):
        vx = xk[:, VX_I]
        smax = (STEER_MAX - (STEER_MAX - 0.14) * (vx - 13.0) / 5.0).clamp(0.14, STEER_MAX)
        steer = A_buf[:, t].clamp(-smax, smax)
        # kinematic yaw injection
        yr_kin = vx * (steer / L_WB)
        xk = xk.clone()
        xk[:, YR_I] = yr_kin
        u = torch.stack([steer, torch.zeros(K, device=dev)], dim=1)   # (K,2)
        # world vel before
        sn, cs = xk[:, SIN_I], xk[:, COS_I]
        Ve1 = xk[:, VX_I]*cs - xk[:, VY_I]*sn
        Vn1 = xk[:, VX_I]*sn + xk[:, VY_I]*cs
        xk = model.step(xk, u, dt_t, euler=False)        # RK4
        sn2, cs2 = xk[:, SIN_I], xk[:, COS_I]
        Ve2 = xk[:, VX_I]*cs2 - xk[:, VY_I]*sn2
        Vn2 = xk[:, VX_I]*sn2 + xk[:, VY_I]*cs2
        pe = pe + 0.5*(Ve1+Ve2)*DT_EFF
        pn = pn + 0.5*(Vn1+Vn2)*DT_EFF
        perp = (pe - refE_buf[t]) * nE_buf[t] + (pn - refN_buf[t]) * nN_buf[t]
        ap = perp.abs()
        ctc = ap.clamp(max=CT_SAT)**2 + 2*CT_SAT*(ap - CT_SAT).clamp(min=0.0)
        carTh = torch.atan2(sn2, cs2)
        bear = torch.atan2(refN_buf[t] - pn, refE_buf[t] - pe)
        dh = torch.atan2(torch.sin(carTh - bear), torch.cos(carTh - bear))
        cost = cost + W_CT*ctc + W_HEAD*dh*dh
    write_cost.copy_(cost / T)

# ---- fill buffers with a representative scenario ----
torch.manual_seed(0)
A_buf.normal_(0, 0.10)
x0_buf.copy_(torch.tensor([4.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=dev))
refE_buf.copy_(torch.linspace(2, 10, T, device=dev))
refN_buf.zero_()
nE_buf.zero_(); nN_buf.fill_(1.0)

# ============ 1. PLAIN ============
torch.cuda.synchronize()
for _ in range(2):  # warmup
    rollout(cost_buf)
torch.cuda.synchronize()
t0 = time.perf_counter()
rollout(cost_buf)
torch.cuda.synchronize()
plain_ms = (time.perf_counter() - t0) * 1000
plain_cost = cost_buf.clone()
print(f"\n[1] PLAIN per-step loop : {plain_ms:.1f} ms")
torch.cuda.empty_cache()
print(f"    mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============ 2. CUDA GRAPH ============
# capture
g = torch.cuda.CUDAGraph()
# warmup in a side stream (required before capture)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        rollout(cost_buf)
torch.cuda.current_stream().wait_stream(s)
try:
    with torch.cuda.graph(g):
        rollout(cost_buf)
    # replay timing
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) * 1000
    graph_cost = cost_buf.clone()
    print(f"[2] CUDA graph replay   : {graph_ms:.1f} ms   ({plain_ms/graph_ms:.0f}x faster)")
    # correctness
    max_diff = (plain_cost - graph_cost).abs().max().item()
    print(f"[3] max cost diff plain-vs-graph: {max_diff:.3e}  (should be ~0)")
    print(f"\nVERDICT: {'PASS <50ms' if graph_ms < 50 else 'still slow'}  graph={graph_ms:.1f}ms")
except Exception as e:
    print(f"[2] CUDA graph FAILED: {type(e).__name__}: {e}")
    print("    (may be the dynamic torch.stack / .clone in loop — fixable)")
