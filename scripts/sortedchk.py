import aiter
import torch

print("AITER:", aiter.__file__, flush=True)


def call(x, k, **kw):
    r = aiter.topk_select(x, k, **kw)
    return r


for M, N in ((8, 131072), (8, 131075), (64, 262144)):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ra = call(x, 2048, sorted=True, return_value=True)
    rb = call(x, 2048, sorted=True, return_value=True)

    # unpack: whatever shape the wrapper returns, find the int32 and the float
    def split(r):
        ts = [t for t in (r if isinstance(r, (tuple, list)) else [r]) if t is not None]
        i = [t for t in ts if t.dtype is torch.int32]
        v = [t for t in ts if t.dtype is torch.float32]
        return (i[0] if i else None), (v[0] if v else None)

    ia, va = split(ra)
    ib, vb = split(rb)
    ia, ib = ia.clone(), ib.clone()
    same_i = bool(torch.equal(ia, ib))
    gva = torch.gather(x, 1, ia.to(torch.int64))
    gvb = torch.gather(x, 1, ib.to(torch.int64))
    same_v_seq = bool(torch.equal(gva, gvb))
    is_desc = bool((gva[:, :-1] >= gva[:, 1:]).all())
    ref = torch.topk(x.float(), 2048, dim=1).values
    matches_ref = bool(torch.equal(gva, ref))
    ndiff = int((ia != ib).sum())
    # where they differ, do the VALUES at those positions tie?
    tie_only = True
    if not same_i:
        d = (ia != ib).nonzero()
        for r_, c_ in d[:200].tolist():
            if float(gva[r_, c_]) != float(gvb[r_, c_]):
                tie_only = False
                break
    print(
        "m=%-4d n=%-8d idx_equal=%-5s val_seq_equal=%-5s descending=%-5s "
        "equals_torch=%-5s ndiff=%-6d diffs_are_ties=%s"
        % (M, N, same_i, same_v_seq, is_desc, matches_ref, ndiff, tie_only),
        flush=True,
    )
