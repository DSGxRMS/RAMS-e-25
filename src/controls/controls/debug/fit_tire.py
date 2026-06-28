"""
Fit EUFS tire (B,C,D,E) to reproduce the IITRMS dataset's LINEAR tire:
  Cf = Cr = 61012 N/rad per axle,  peak friction mu = 1.7.

EUFS lateral force model (per wheel, eufs dynamic_bicycle.cpp / _getFy):
  Fy_wheel = Fz_axle * D * sin( C * atan( B*(1-E)*alpha + E*atan(B*alpha) ) )
where the dynamics use FyF_tot = 2 * Fy_wheel (axle = 2 wheels), and
  Fz_axle_front = 0.5 * w_front * Fz_total          (eufs _getDownForceFront)
  Fz_total      = m*g + c_down*v^2                  (normal force)

Small-angle slope of the AXLE force (2 wheels):
  dFy_axle/dalpha |_0 = 2 * Fz_axle * D * B * C
Set this equal to the dataset Cf and solve for B.

We evaluate Fz at the STATIC nominal (v=0): Fz_total = m*g.
"""
import math

m       = 287.0
g       = 9.81
w_front = 0.47
Cf_target = 61012.0        # dataset per-axle cornering stiffness (N/rad)
mu      = 1.7              # dataset peak friction

# EUFS lateral shape constants (standard Pacejka lateral)
C = 1.3                    # shape factor (typical lateral)
D = mu                     # peak factor = friction coefficient
E = -1.0                   # curvature (mild)

# static normal force and front axle vertical load (per the EUFS split)
Fz_total = m * g
Fz_axle_front = 0.5 * w_front * Fz_total      # matches _getDownForceFront
Fz_axle_rear  = 0.5 * (1 - w_front) * Fz_total

# axle slope at alpha=0 is 2 * Fz_axle * D * B * C  -> solve B
# front and rear have different Fz; the dataset used equal Cf=Cr, but EUFS uses
# ONE tire block for both axles. We fit B so the FRONT axle matches Cf_target,
# then report what the rear axle stiffness comes out to (informational).
B = Cf_target / (2.0 * Fz_axle_front * D * C)

# verify
def fy_axle(Fz_axle, alpha):
    fy_wheel = Fz_axle * D * math.sin(C * math.atan(B*(1-E)*alpha + E*math.atan(B*alpha)))
    return 2.0 * fy_wheel

# numeric slope check
da = 1e-4
slope_front = (fy_axle(Fz_axle_front, da) - fy_axle(Fz_axle_front, -da)) / (2*da)
slope_rear  = (fy_axle(Fz_axle_rear,  da) - fy_axle(Fz_axle_rear,  -da)) / (2*da)
peak_front  = max(fy_axle(Fz_axle_front, a) for a in [x*0.005 for x in range(0,80)])

print(f"Fz_total (static)   = {Fz_total:.1f} N")
print(f"Fz_axle_front       = {Fz_axle_front:.1f} N   Fz_axle_rear = {Fz_axle_rear:.1f} N")
print(f"")
print(f"FITTED TIRE COEFFICIENTS:")
print(f"  B = {B:.4f}")
print(f"  C = {C}")
print(f"  D = {D}")
print(f"  E = {E}")
print(f"")
print(f"VERIFY:")
print(f"  front axle slope @0  = {slope_front:8.1f} N/rad  (target {Cf_target})")
print(f"  rear  axle slope @0  = {slope_rear:8.1f} N/rad  (rear has more load -> higher)")
print(f"  front axle peak Fy   = {peak_front:8.1f} N   (= mu*Fz_axle*2 = {mu*Fz_axle_front*2:.1f})")
print(f"")
print(f"NOTE: EUFS uses one tire block for both axles, so rear stiffness scales")
print(f"with its higher load (53% vs 47%). Dataset used equal Cf=Cr; this is a")
print(f"minor mismatch baked into the EUFS model structure, held stable by the sim.")
