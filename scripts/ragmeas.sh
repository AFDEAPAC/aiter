#!/bin/bash
cd /topk
run() {
  rm -rf /tmp/krm
  FORCE_RAGGED=$1 rocprofv3 --kernel-trace --output-format csv -d /tmp/krm -- \
    python3 -u /topk/scripts/ragmeas.py >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
ev=[]
for f in glob.glob('/tmp/krm/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_' in r['Kernel_Name']:
            ev.append((int(r['Start_Timestamp']),(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3))
ev.sort()
# 8 shapes x 28 calls x 3 kernels
per=len(ev)//8
for i in range(8):
    seg=[d for _,d in ev[i*per:(i+1)*per]]
    # drop the 8 warmup calls (24 kernels)
    seg=seg[24:]
    print('%9.2f'%(sum(seg)/(len(seg)/3)), end=' ')
print()
"
}
echo "per-call three-kernel sum, through aiter.topk_select"
printf "%-26s %9s %9s %9s %9s %9s %9s %9s %9s\n" "" "m2048" "m1024" "m512" "m256" "m512" "m256" "m4096" "m128"
printf "%-26s %9s %9s %9s %9s %9s %9s %9s %9s\n" "" "131072" "131072" "131072" "131072" "262144" "262144" "131072" "1048576"
printf "%-26s %s" "ragged (caller gave end)" "$(run 1)"
printf "%-26s %s" "plain  (new default)     " "$(run 0)"
