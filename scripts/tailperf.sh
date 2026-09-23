#!/bin/bash
cd /topk
one() {
  rm -rf /tmp/tp
  M=$1 rocprofv3 --kernel-trace --output-format csv -d /tmp/tp -- \
    python3 -u scripts/tailperf.py >/dev/null 2>&1
  python3 -c "
import csv,glob,collections
ev=[]
for f in glob.glob('/tmp/tp/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_' in r['Kernel_Name']:
            ev.append((int(r['Start_Timestamp']),(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3))
ev.sort()
Ns=[131072,131073,131074,131075,131076,262144,262145,262147]
per=len(ev)//len(Ns)
out=[]
for i in range(len(Ns)):
    seg=[d for _,d in ev[i*per:(i+1)*per]][24:]   # drop 8 warmup calls x 3 kernels
    out.append('%8.2f'%(sum(seg)/(len(seg)/3)))
print(' '.join(out))
"
}
echo "per-call three-kernel sum, us. The tail costs nothing when N%4 == 0 because"
echo "ncols is then zero and the whole block is predicated off."
printf "%6s" M; for N in 131072 131073 131074 131075 131076 262144 262145 262147; do printf " %8d" $N; done; echo
for M in 512 256 64; do printf "%6d %s\n" $M "$(one $M)"; done
