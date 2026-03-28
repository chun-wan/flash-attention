import torch, math, time, sys
sys.path.insert(0, "/workspace/fa4_rocm")
device = "cuda:0"

from triton_kernels.flash_fwd_triton_v3 import flash_attn_triton_v3_func as fa4v3
from triton_kernels.flash_fwd_triton_v2 import flash_attn_triton_v2_func as fa4v2
from aiter import flash_attn_func as aiter_fa
from aiter import fmha_v3_fwd as _fmha_v3

def v3asm(q,k,v,causal=False,softmax_scale=None):
    return _fmha_v3(q,k,v,0.0,softmax_scale or 1.0/math.sqrt(q.shape[-1]),causal,-1,-1,False,False,0)[0]
def sdpa(q,k,v,causal=False,softmax_scale=None):
    sc=softmax_scale or 1.0/math.sqrt(q.shape[-1])
    qt=q.transpose(1,2);kt=k.transpose(1,2);vt=v.transpose(1,2)
    hq,hk=q.shape[2],k.shape[2]
    if hq!=hk:kt=kt.repeat_interleave(hq//hk,dim=1);vt=vt.repeat_interleave(hq//hk,dim=1)
    return torch.nn.functional.scaled_dot_product_attention(qt,kt,vt,is_causal=causal,scale=sc).transpose(1,2)

W,I = 20, 100
def bench(fn,q,k,v,c):
    sc=1.0/math.sqrt(q.shape[-1])
    torch.cuda.synchronize()
    for _ in range(W): fn(q,k,v,causal=c,softmax_scale=sc)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(I): fn(q,k,v,causal=c,softmax_scale=sc)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/I*1e6

shapes = [
    (2,2048,2048,32,32,128,True, "b2 s2048 causal"),
    (2,4096,4096,32,32,128,True, "b2 s4096 causal"),
    (1,8192,8192,32,32,128,True, "b1 s8192 causal"),
    (4,1024,1024,32,32,128,True, "b4 s1024 causal"),
    (2,2048,2048,32,8,128,True,  "b2 gqa4 causal"),
    (2,2048,2048,32,32,128,False,"b2 noncausal"),
]

print("=" * 100)
print("FINAL: MI325X ROCm 7.2 Native BF16 Forward (fa4_rocm_dev container)")
print("=" * 100)
print(f"{'Shape':<22s}| {'FA4v2':>7s}| {'FA4v3':>7s}| {'FA2':>7s}| {'FA3v3':>7s}| {'SDPA':>7s}| {'v3/FA3':>7s}|")
print("-" * 75)

for b,sq,sk,hq,hk,hd,c,name in shapes:
    torch.manual_seed(42)
    q=torch.randn(b,sq,hq,hd,dtype=torch.bfloat16,device=device)
    k=torch.randn(b,sk,hk,hd,dtype=torch.bfloat16,device=device)
    v=torch.randn(b,sk,hk,hd,dtype=torch.bfloat16,device=device)
    cf=0.5 if c else 1.0; fl=4*b*sq*sk*hq*hd*cf
    r={}
    for bn,fn in [("v2",fa4v2),("v3",fa4v3),("fa2",aiter_fa),("v3a",v3asm),("sdpa",sdpa)]:
        try:
            lat=bench(fn,q,k,v,c); r[bn]=fl/(lat*1e-6)/1e12
        except: r[bn]=0
    def f(x): return f"{r.get(x,0):>5.0f}TF" if r.get(x,0)>0 else f"{'ERR':>7s}"
    ratio=f"{r['v3']/r['v3a']:.0%}" if r.get('v3',0)>0 and r.get('v3a',0)>0 else 'N/A'
    print(f"{name:<22s}| {f('v2'):>7s}| {f('v3'):>7s}| {f('fa2'):>7s}| {f('v3a'):>7s}| {f('sdpa'):>7s}| {ratio:>7s}|")

print("=" * 100)
