p = "/aiter/csrc/kernels/topk_per_row_sampled_kernels.cu"
with open(p) as fh:
    s = fh.read()
n = 0
for a, b in [
    ("topk_fused_impl<true, true>", "topk_fused_impl<false, true>"),
    ("topk_fused_impl<true, false>", "topk_fused_impl<false, false>"),
    ("topk_small_n<true, true>", "topk_small_n<false, true>"),
    ("topk_small_n<true, false>", "topk_small_n<false, false>"),
]:
    n += s.count(a)
    s = s.replace(a, b)
with open(p, "w") as fh:
    fh.write(s)
print(f"forced non-ragged at {n} call sites")
