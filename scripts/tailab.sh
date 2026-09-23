#!/bin/bash
cd /topk
one() {
  rm -rf /tmp/tab
  M=$1 FORCE_RAGGED=$2 rocprofv3 --kernel-trace --output-format csv -d /tmp/tab -- \
    python3 -u scripts/tailab.py >/dev/null 2>&1
  python3 -c "
import csv,glob
ev=[]
for f in glob.glob('/tmp/tab/**/*kernel_trace.csv',recursive=True):
    for r in csv.DictReader(open(f)):
        if 'phase_' in r['Kernel_Name']:
            ev.append((int(r['Start_Timestamp']),(int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e3))
ev.sort()
Ns=[131072,131073,131075,262144,262147,524291]
per=len(ev)//len(Ns)
print(' '.join('%8.2f'%(sum([d for _,d in ev[i*per:(i+1)*per]][24:])/((per-24)/3)) for i in range(len(Ns))))
"
}
echo "per-call three-kernel sum, us. The plain path now covers every width, so the"
echo "question is whether it is still faster than the bounds-checking one there."
printf "%6s %-8s" M path; for N in 131072 131073 131075 262144 262147 524291; do printf " %8d" $N; done; echo
for M in 512 256; do
  printf "%6d %-8s %s\n" $M "ragged" "$(one $M 1)"
  printf "%6d %-8s %s\n" $M "plain" "$(one $M 0)"
done
