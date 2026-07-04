"""Verify: loading model_state_dict makes the rollout respond to steering."""
import torch, torch.nn as nn, math
CKPT="/mnt/d/RAMS-e-25/src/controls/controls/fs_model/best_model_600.pt"
VX_I,VY_I,YR_I,SIN_I,COS_I,SLIP_I=0,1,2,3,4,5
class _Func(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(8,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),
                            nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4))
    def forward(s,x,u,xm,xs,um,us):
        g=s.net(torch.cat([(x-xm)/xs,(u-um)/us],dim=-1))
        yr=x[...,YR_I];ys=x[...,SIN_I];yc=x[...,COS_I]
        return torch.stack([g[...,0]*xs[VX_I],g[...,1]*xs[VY_I],g[...,2]*xs[YR_I],
                            yc*yr,-ys*yr,g[...,3]*xs[SLIP_I]],dim=-1)
def _rk4(f,x,u,h,*s):
    k1=f(x,u,*s);k2=f(x+0.5*h*k1,u,*s);k3=f(x+0.5*h*k2,u,*s);k4=f(x+h*k3,u,*s)
    return x+(h/6.0)*(k1+2*k2+2*k3+k4)
class _GB(nn.Module):
    def __init__(s,ck):
        super().__init__()
        s.func=_Func();s.nr=int(ck["num_rk4_steps"])
        for n in ("x_mean","x_std","u_mean","u_std","x_min","x_max"):
            s.register_buffer(n,torch.as_tensor(ck[n],dtype=torch.float32))
    def step(s,x,u,dt):
        h=dt/s.nr
        for _ in range(s.nr): x=_rk4(s.func,x,u,h,s.x_mean,s.x_std,s.u_mean,s.u_std)
        a=x[...,SIN_I];b=x[...,COS_I];nrm=torch.sqrt(a*a+b*b).clamp_min(1e-6)
        x=x.clone();x[...,SIN_I]=a/nrm;x[...,COS_I]=b/nrm
        return torch.clamp(x,s.x_min,s.x_max)
ck=torch.load(CKPT,map_location="cpu",weights_only=False)
m=_GB(ck)
res=m.load_state_dict(ck["model_state_dict"], strict=False)
print("missing keys:", res.missing_keys)
print("unexpected keys:", res.unexpected_keys)
print("net[0].weight std AFTER load = %.5f (was ~0.20 random)" % m.func.net[0].weight.std())
m.eval()
dt=torch.tensor(0.005)
def roll(steer):
    x=torch.tensor([[4.0,0,0,-1.0,0.0,0.0]]);pe=pn=0.0
    for _ in range(200):
        S=x[0,3].item();C=x[0,4].item();vx=x[0,0].item();vy=x[0,1].item()
        Ve1=-(vx*S+vy*C);Vn1=(vx*C-vy*S)
        x=m.step(x,torch.tensor([[steer,0.0]]),dt)
        S=x[0,3].item();C=x[0,4].item();vx=x[0,0].item();vy=x[0,1].item()
        Ve2=-(vx*S+vy*C);Vn2=(vx*C-vy*S)
        pe+=0.5*(Ve1+Ve2)*0.005;pn+=0.5*(Vn1+Vn2)*0.005
    return pe,pn,math.degrees(math.atan2(x[0,4].item(),-x[0,3].item())),x[0,2].item()
for s in (+0.20,-0.20):
    pe,pn,th,yr=roll(s)
    print(f"steer={s:+.2f}: end E={pe:6.2f} N={pn:6.2f} heading={th:+7.1f}deg yawRate={yr:+.3f}  (+steer should => +heading,+N=left)")
