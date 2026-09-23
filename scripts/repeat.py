import torch, aiter

print("AITER:", aiter.__file__, flush=True)


def call(x, k, **kw):
    r = aiter.topk_select(x, k, **kw)
    if isinstance(r, (tuple, list)):
        r = [t for t in r if t is not None][-1]
    return r


print("%6s %9s %14s %14s %14s" % ("M", "N", "same order", "same set", "same values"))
for M, N in ((8, 131072), (8, 131075), (64, 262147), (512, 131072)):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    a = call(x, 2048).clone()
    b = call(x, 2048).clone()
    order = bool(torch.equal(a, b))
    sa = a.to(torch.int64).sort(dim=1).values
    sb = b.to(torch.int64).sort(dim=1).values
    same_set = bool(torch.equal(sa, sb))
    va = torch.gather(x, 1, a.to(torch.int64)).sort(dim=1).values
    vb = torch.gather(x, 1, b.to(torch.int64)).sort(dim=1).values
    same_val = bool(torch.equal(va, vb))
    print("%6d %9d %14s %14s %14s" % (M, N, order, same_set, same_val), flush=True)

print()
print("and with deterministic=True, where order IS part of the contract:")
for M, N in ((8, 131075), (512, 131072)):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    try:
        a = call(x, 2048, deterministic=True).clone()
        b = call(x, 2048, deterministic=True).clone()
        print("   m=%-5d n=%-9d same order: %s" % (M, N, bool(torch.equal(a, b))), flush=True)
    except Exception as exc:
        print("   m=%-5d n=%-9d deterministic=True raised %s" % (M, N, type(exc).__name__), flush=True)

print()
print("and with sorted=True:")
for M, N in ((8, 131075),):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    try:
        a = call(x, 2048, sorted=True).clone()
        b = call(x, 2048, sorted=True).clone()
        print("   m=%-5d n=%-9d same order: %s" % (M, N, bool(torch.equal(a, b))), flush=True)
    except Exception as exc:
        print("   m=%-5d n=%-9d sorted=True raised %s" % (M, N, type(exc).__name__), flush=True)
