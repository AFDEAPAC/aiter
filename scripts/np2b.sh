#!/bin/bash
cd /topk
echo "### $1 ###"
M=8 python3 -u scripts/nonpow2b.py 2>&1 | grep -vE "^\[aiter\]|WARNING|warn" | tail -28
