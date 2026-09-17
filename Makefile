HIPCC ?= hipcc
HIPFLAGS = -O3 -std=c++17 --offload-arch=gfx950 -Icsrc
LDFLAGS =

# Header dependencies come from the compiler, not from a hand-written list.
# They used to be listed by hand as `csrc/topk_common.hip.hpp` only, so editing
# csrc/topk_shape.hip.hpp or csrc/topk_generalize.hip.hpp did NOT trigger a
# rebuild and `make && ./benchmark_topk` silently measured the previous binary.
DEPDIR = .deps
DEPFLAGS = -MMD -MP -MF $(DEPDIR)/$(@F).d

.PHONY: all clean

all: benchmark_topk bw_kernel floor_bench

$(DEPDIR):
	@mkdir -p $(DEPDIR)

benchmark_topk: benchmark_topk.hip.cpp | $(DEPDIR)
	$(HIPCC) $(HIPFLAGS) $(DEPFLAGS) -o $@ benchmark_topk.hip.cpp $(LDFLAGS)

bw_kernel: scripts/bw_kernel.hip | $(DEPDIR)
	$(HIPCC) $(HIPFLAGS) $(DEPFLAGS) -o $@ scripts/bw_kernel.hip $(LDFLAGS)

floor_bench: scripts/floor_bench.hip | $(DEPDIR)
	$(HIPCC) $(HIPFLAGS) $(DEPFLAGS) -o $@ scripts/floor_bench.hip $(LDFLAGS)

clean:
	rm -rf benchmark_topk bw_kernel floor_bench score.json $(DEPDIR)

-include $(DEPDIR)/benchmark_topk.d
-include $(DEPDIR)/bw_kernel.d
-include $(DEPDIR)/floor_bench.d
