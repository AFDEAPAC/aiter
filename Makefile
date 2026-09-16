HIPCC ?= hipcc
HIPFLAGS = -O3 -std=c++17 --offload-arch=gfx950 -Icsrc
LDFLAGS =

.PHONY: all clean bw_kernel

all: benchmark_topk bw_kernel

benchmark_topk: benchmark_topk.hip.cpp csrc/topk_common.hip.hpp
	$(HIPCC) $(HIPFLAGS) -o $@ benchmark_topk.hip.cpp $(LDFLAGS)

bw_kernel: scripts/bw_kernel.hip
	$(HIPCC) $(HIPFLAGS) -o $@ scripts/bw_kernel.hip $(LDFLAGS)

clean:
	rm -f benchmark_topk bw_kernel score.json
