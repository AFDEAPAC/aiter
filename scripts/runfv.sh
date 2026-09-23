#!/bin/bash
cd /topk
STAGE=${STAGE:-all} python3 -u scripts/fullverify.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn|NUMA"
echo "EXIT=${PIPESTATUS[0]}"
