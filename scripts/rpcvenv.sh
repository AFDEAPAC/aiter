#!/bin/bash
cd /topk
python3 -m venv --system-site-packages /tmp/rpcenv 2>&1 | tail -1
/tmp/rpcenv/bin/pip -q install -r /opt/rocm-10.0/libexec/rocprofiler-compute/requirements.txt 2>&1 | tail -3
echo "--- analyze ---"
PYTHONPATH= /tmp/rpcenv/bin/python /opt/rocm-10.0/libexec/rocprofiler-compute/rocprof-compute analyze \
  -p /topk/workloads/pa8/MI355 -b 2 2>&1 | tail -40
