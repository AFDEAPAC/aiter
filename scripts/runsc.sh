#!/bin/bash
cd /topk
echo "### $1 ###"
python3 -u scripts/sortedchk.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn|NUMA"
