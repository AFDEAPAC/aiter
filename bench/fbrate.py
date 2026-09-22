import sys; sys.path.insert(0,"/topk/bench")
import torch, aiter, select_ab_sweep as S
k=2048
def fb(m,n,seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x=torch.randn(m,n,dtype=torch.float32,device="cuda")
    idx=torch.empty((m,k),dtype=torch.int32,device="cuda")
    rs=torch.zeros(m,dtype=torch.int32,device="cuda"); re=torch.full((m,),n,dtype=torch.int32,device="cuda")
    w=aiter.topk_sampled_workspace_size(m,n,k); ws=torch.empty(w,dtype=torch.uint8,device="cuda")
    aiter.top_k_per_row_prefill_sampled(x,rs,re,idx,None,m,n,1,k=k,workspace=ws)
    torch.cuda.synchronize()
    c=ws[w-256:w-252].view(torch.int32).item()
    ref=torch.sort(torch.topk(x,k,dim=1).values,dim=1,descending=True).values
    got=torch.sort(x.gather(1,idx.clamp_min(0).long()),dim=1,descending=True).values
    return c,(ref-got).abs().max().item()
print("fallback rows over 8 seeds, gaussian:")
print("%6s %9s %14s %8s"%("M","N","fb rows/total","max err"))
tot=0
for m in (16,32,64,128,4096):
    for n in (131072,262144,524288,1048576):
        s=0; e=0.0
        for sd in range(8):
            c,err=fb(m,n,sd); s+=c; e=max(e,err)
        tot+=s
        print("%6d %9d %14s %8g"%(m,n,"%d/%d"%(s,m*8),e), flush=True)
print("TOTAL fallback rows:",tot)
print()
print("cost of the bigger S (no-fallback cells):")
for m,n in ((16,524288),(32,524288),(64,524288),(128,1048576),(1024,524288),(4096,524288),(4096,1048576),(4096,131072)):
    torch.manual_seed(1); torch.cuda.manual_seed_all(1)
    x=torch.randn(m,n,dtype=torch.float32,device="cuda")
    idx=torch.empty((m,k),dtype=torch.int32,device="cuda")
    rs=torch.zeros(m,dtype=torch.int32,device="cuda"); re=torch.full((m,),n,dtype=torch.int32,device="cuda")
    w=aiter.topk_sampled_workspace_size(m,n,k); ws=torch.empty(w,dtype=torch.uint8,device="cuda")
    fn=lambda: aiter.top_k_per_row_prefill_sampled(x,rs,re,idx,None,m,n,1,k=k,workspace=ws)
    for _ in range(3): fn()
    torch.cuda.synchronize()
    c=S._counts_per_iter(fn); us,_=S._trace_time(fn,c,iters=S._profiled_iters(fn,1,c))
    print("  m=%-5d n=%-8d %9.2f us"%(m,n,us), flush=True)
