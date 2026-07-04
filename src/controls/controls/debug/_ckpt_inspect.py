"""One-off: is the trained NN actually loaded into the controller's model?
Replicates the controller's _Func/_GB EXACTLY and checks whether best_model_600.pt
contains NN weights and whether _GB(ck) actually uses them."""
import torch, torch.nn as nn, math

CKPT = "/mnt/d/RAMS-e-25/src/controls/controls/fs_model/best_model_600.pt"
VX_I, VY_I, YR_I, SIN_I, COS_I, SLIP_I = 0, 1, 2, 3, 4, 5

class _Func(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU(),
            nn.Linear(128,128), nn.SiLU(), nn.Linear(128,4))
    def forward(self, x, u, xm, xs, um, us):
        inp = torch.cat([(x-xm)/xs, (u-um)/us], dim=-1)
        g = self.net(inp); yr=x[...,YR_I]; ys=x[...,SIN_I]; yc=x[...,COS_I]
        return torch.stack([g[...,0]*xs[VX_I], g[...,1]*xs[VY_I], g[...,2]*xs[YR_I],
                            yc*yr, -ys*yr, g[...,3]*xs[SLIP_I]], dim=-1)

def _rk4(f,x,u,h,*s):
    k1=f(x,u,*s); k2=f(x+0.5*h*k1,u,*s); k3=f(x+0.5*h*k2,u,*s); k4=f(x+h*k3,u,*s)
    return x+(h/6.0)*(k1+2*k2+2*k3+k4)

class _GB(nn.Module):
    def __init__(self, ck):
        super().__init__()
        self.func=_Func(); self.nr=int(ck["num_rk4_steps"])
        for n in ("x_mean","x_std","u_mean","u_std","x_min","x_max"):
            self.register_buffer(n, torch.as_tensor(ck[n], dtype=torch.float32))
    def step(self,x,u,dt):
        h=dt/self.nr
        for _ in range(self.nr):
            x=_rk4(self.func,x,u,h,self.x_mean,self.x_std,self.u_mean,self.u_std)
        a=x[...,SIN_I]; b=x[...,COS_I]; nrm=torch.sqrt(a*a+b*b).clamp_min(1e-6)
        x=x.clone(); x[...,SIN_I]=a/nrm; x[...,COS_I]=b/nrm
        return torch.clamp(x,self.x_min,self.x_max)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("=== checkpoint type:", type(ck).__name__)
if isinstance(ck, dict):
    print("=== top-level keys:")
    for k,v in ck.items():
        if isinstance(v, torch.Tensor): print(f"   {k:18s} Tensor{tuple(v.shape)}")
        elif isinstance(v, dict):       print(f"   {k:18s} dict keys={list(v.items().__iter__().__next__()[0] if False else list(v.keys()))[:10]}")
        else:                           print(f"   {k:18s} {type(v).__name__} = {v}")

# does the checkpoint contain NN weights anywhere?
def find_net_weights(d, path=""):
    found={}
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v, torch.Tensor) and ("net" in str(k) or "weight" in str(k) or "bias" in str(k)):
                found[path+str(k)]=tuple(v.shape)
            elif isinstance(v, dict):
                found.update(find_net_weights(v, path+str(k)+"."))
    return found
w = find_net_weights(ck)
print("=== NN-weight-looking tensors in checkpoint:", len(w))
for k,s in list(w.items())[:12]: print("   ", k, s)

# build the model the SAME way the controller does
m = _GB(ck); m.eval()
ctrl_w = m.func.net[0].weight.detach().clone()
print("=== controller's loaded net[0].weight  mean=%.5f std=%.5f" % (ctrl_w.mean(), ctrl_w.std()))

# if the checkpoint HAS a state_dict, load it into a 2nd model and compare
loaded=False
for key in ("model_state_dict","state_dict","model","net","func"):
    if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
        try:
            m2=_GB(ck);
            sd={kk.replace("func.",""):vv for kk,vv in ck[key].items()}
            # try a few prefixes
            try: m2.func.load_state_dict(ck[key]); loaded=True; src=key
            except Exception:
                try: m2.func.load_state_dict(sd); loaded=True; src=key+"(stripped)"
                except Exception: pass
            if loaded:
                tw=m2.func.net[0].weight
                print(f"=== checkpoint['{src}'] net[0].weight mean=%.5f std=%.5f" % (tw.mean(), tw.std()))
                print("=== SAME as controller's?", torch.allclose(ctrl_w, tw))
        except Exception as e: print("load attempt err:", e)

# steering sensitivity of the controller's loaded model
dt=torch.tensor(0.005)
def roll(steer):
    x=torch.tensor([[4.0,0,0,-1.0,0.0,0.0]]); pe=pn=0.0
    for _ in range(200):
        S=x[0,3].item();C=x[0,4].item();vx=x[0,0].item();vy=x[0,1].item()
        Ve1=-(vx*S+vy*C);Vn1=(vx*C-vy*S)
        x=m.step(x,torch.tensor([[steer,0.0]]),dt)
        S=x[0,3].item();C=x[0,4].item();vx=x[0,0].item();vy=x[0,1].item()
        Ve2=-(vx*S+vy*C);Vn2=(vx*C-vy*S)
        pe+=0.5*(Ve1+Ve2)*0.005; pn+=0.5*(Vn1+Vn2)*0.005
    th=math.degrees(math.atan2(x[0,4].item(),-x[0,3].item()))
    return pe,pn,th
for s in (+0.20,-0.20):
    pe,pn,th=roll(s)
    print(f"=== steer={s:+.2f}: end pos E={pe:6.2f} N={pn:6.2f}  heading={th:+7.1f} deg  (left turn => +heading,+N)")
