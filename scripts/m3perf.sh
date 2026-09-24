#!/bin/bash
cd /topk
one() {
  rm -rf /tmp/m3p
  M=$1 FORCE_RAGGED=$2 rocprofv3 --kernel-trace --output-format csv -d /tmp/m3p -- \
    python3 -u scripts/m3perf.py >/dev/null 2>&1
  python3 -c "
import csv,glob
ev=[]
for f in glob.glob('/tmp/m3p/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_' in r['Kernel_Name']:
            ev.append((int(r['Start_Timestamp']),(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3))
ev.sort()
n=8; per=len(ev)//n
print(' '.join('%9.2f'%(sum([d for _,d in ev[i*per:(i+1)*per]][24:])/((per-24)/3)) for i in range(n)))
"
}
echo "per-call three-kernel sum, us. The 390-cell grid stops at +2, so N%4==3 has"
echo "never been measured. Each pow2 is followed by its +3."
printf "%6s %-7s" M path; for N in 131072 131075 262144 262147 524288 524291 1048576 1048579; do printf " %9d" $N; done; echo
for M in 512 256 64; do
  printf "%6d %-7s %s\n" $M "ragged" "$(one $M 1)"
  printf "%6d %-7s %s\n" $M "plain"  "$(one $M 0)"
done
