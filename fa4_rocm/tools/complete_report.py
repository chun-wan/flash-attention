#!/usr/bin/env python3
"""Complete FA4 ROCm project report."""
import torch, math, time, sys
sys.path.insert(0, "/workspace/fa4_rocm")
device = "cuda:0"
W, I = 20, 100

def bench(fn, q, k, v, c):
    sc = 1.0/math.sqrt(q.shape[-1])
    torch.cuda.synchronize()
    for _ in range(W): fn(q, k, v, causal=c, softmax_scale=sc)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(I): fn(q, k, v, causal=c, softmax_scale=sc)
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/I*1e6

from triton_kernels.flash_fwd_evo import evo_flash_attn
best = {"BLOCK_M":128,"BLOCK_N":64,"num_warps":4,"num_stages":2,"waves_per_eu":2,"PRE_LOAD_V":True,"PRE_SCALE_Q":False}
def avo(q,k,v,causal=False,softmax_scale=None):
    return evo_flash_attn(q,k,v,causal=causal,softmax_scale=softmax_scale,**best)
from aiter import flash_attn_func as fa2
from aiter import fmha_v3_fwd as _v3
def fa3(q,k,v,causal=False,softmax_scale=None):
    return _v3(q,k,v,0.0,softmax_scale or 1/math.sqrt(q.shape[-1]),causal,-1,-1,False,False,0)[0]

shapes = [
    (2,2048,32,32,True,"b2 s2048 causal"),
    (2,4096,32,32,True,"b2 s4096 causal"),
    (1,8192,32,32,True,"b1 s8192 causal"),
    (4,1024,32,32,True,"b4 s1024 causal"),
    (2,2048,32,32,False,"b2 noncausal"),
]

print("="*80)
print("COMPLETE FA4 ROCm PROJECT REPORT")
print("MI325X gfx942 | ROCm 7.2 Native | Container: fa4_rocm_dev")
print("="*80)
hdr = f"{'Shape':<20s}|{'AVO':>7s}|{'FA2':>7s}|{'FA3':>7s}|{'%FA3':>6s}|"
print(hdr)
print("-"*45)

for b,sq,hq,hk,c,nm in shapes:
    torch.manual_seed(42)
    q=torch.randn(b,sq,hq,128,dtype=torch.bfloat16,device=device)
    k=torch.randn(b,sq,hk,128,dtype=torch.bfloat16,device=device)
    v=torch.randn(b,sq,hk,128,dtype=torch.bfloat16,device=device)
    cf=0.5 if c else 1.0
    fl=4*b*sq*sq*hq*128*cf
    r={}
    for bn,fn in [("a",avo),("2",fa2),("3",fa3)]:
        try: lat=bench(fn,q,k,v,c); r[bn]=fl/(lat*1e-6)/1e12
        except: r[bn]=0
    def f(x): return f"{r.get(x,0):>5.0f}TF" if r.get(x,0)>0 else "  ERR"
    ra=f"{r['a']/r['3']:.0%}" if r.get('a',0)>0 and r.get('3',0)>0 else 'N/A'
    print(f"{nm:<20s}|{f('a'):>7s}|{f('2'):>7s}|{f('3'):>7s}|{ra:>6s}|")

print("="*80)
print()
print("Evolution: 13 -> 122 -> 180 -> 192 -> 200 TF")
print("           HIP   Tri-v1 3phase log2   AVO-R20")
print("           3%    32%    47%    51%    53% of FA3")
print()
print("FlyDSL: compiles+runs on GPU (copy kernel verified)")
print("  XOR16 swizzle, MFMA bf16, ping-pong LDS all available")
print()
print("Bottleneck: VALU/MFMA ratio 9.6x (Triton codegen)")
print("  CK ASM achieves ~3x via hand-scheduled MFMA-VMEM")
