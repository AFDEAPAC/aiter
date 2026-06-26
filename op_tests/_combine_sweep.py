import csv

def load(p):
    d = {}
    for l in open(p):
        parts = l.strip().split(",")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        b, c, v = int(parts[0]), int(parts[1]), parts[2]
        try:
            d[(b, c)] = float(v)
        except Exception:
            d[(b, c)] = None  # ERR
    return d

hk = load("/tmp/sweep_hk.csv")
asm = load("/tmp/sweep_asm.csv")
BS = [1, 16, 32, 64, 128, 256]
CTXS = [1200, 3200, 5200, 8000, 16384, 32768, 65536]
LAB = {1200: "1200", 3200: "3200", 5200: "5200", 8000: "8K", 16384: "16K", 32768: "32K", 65536: "64K"}

print("=== HK/asm ratio (qh64 qlen1 bf16; <1.0 = HK faster than asm) ===")
print("ctx\\B " + "".join("%7d" % b for b in BS))
for c in CTXS:
    cells = []
    for b in BS:
        h, a = hk.get((b, c)), asm.get((b, c))
        cells.append("%7.2f" % (h / a) if (h and a) else "%7s" % "-")
    print("%5s " % LAB[c] + "".join(cells))

print("\n=== raw us  HK | asm  at key cells ===")
for (b, c) in [(16, 16384), (32, 16384), (64, 16384), (32, 8000), (64, 5200),
               (128, 3200), (256, 5200), (64, 65536), (256, 16384)]:
    print("B%-3d ctx%-5s: HK %-9s asm %-9s" % (b, LAB[c], hk.get((b, c)), asm.get((b, c))))
