#!/usr/bin/env python3
"""Emit .evo/hw.json for the running AMD GPU machine.

This is the canonical query_hw.py. It is referenced by:
- skills/kernel-optimization/hardware-query.md (documentation)
- skills/agentic-kernel-evolution/scripts/evo_init.py (copies this file
  into a freshly initialized evolution repo as scripts/query_hw.py)

If you change this file, you only need to change it here. Both the docs
and the scaffolder pick up the change automatically.

Usage:
    python query_hw.py > .evo/hw.json

The JSON is a run-start snapshot. Do not regenerate it mid-run unless you
fork to a new evo/<name>-vN branch (see hardware-query.md §When to re-query).

Requires: rocminfo, rocm-smi, hipconfig (all part of ROCm). No Python deps.
"""

import datetime
import json
import re
import socket
import subprocess


# Per-SKU AMD-published peak numbers. Only SKU-specific constants live here;
# everything else (haircuts, ridge points, etc.) is derived.
DATASHEET = {
    "MI300X": {"hbm_bw_tbps": 5.30, "bf16_tflops": 1307.4, "fp8_tflops": 2614.9, "fp32_tflops": 163.4},
    "MI300A": {"hbm_bw_tbps": 5.30, "bf16_tflops":  980.6, "fp8_tflops": 1961.2, "fp32_tflops": 122.6},
    "MI308X": {"hbm_bw_tbps": 5.30, "bf16_tflops":  612.0, "fp8_tflops": 1224.0, "fp32_tflops":  76.5},
    "MI325X": {"hbm_bw_tbps": 6.00, "bf16_tflops": 1307.4, "fp8_tflops": 2614.9, "fp32_tflops": 163.4},
    "MI355X": {"hbm_bw_tbps": 8.00, "bf16_tflops": 2500.0, "fp8_tflops": 5000.0, "fp32_tflops": 156.2},
}

# Fallback when rocminfo Marketing Name is generic ("AMD Radeon Graphics") —
# look up the PCI chip ID instead. Source: aiter's chip_info.py for the MI308
# set (verified upstream); other AMD-published mappings can be appended here
# with a citation in source_notes.
CHIP_ID_TO_SKU = {
    0x74A2: "MI308X",
    0x74A8: "MI308X",
    0x74B6: "MI308X",
    0x74BC: "MI308X",
}

XCD_FAMILIES = ("MI300", "MI308", "MI325", "MI355")


def run(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return ""


def search_int(pattern, text, default=0, flags=0):
    match = re.search(pattern, text, flags)
    return int(match.group(1)) if match else default


def search_str(pattern, text, default="unknown", flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else default


def split_agents(text):
    """Split rocminfo output into per-agent blocks.

    rocminfo lists every HSA Agent (CPU sockets first, then GPUs). We need
    the first GPU; using a single-shot regex on the full text would silently
    capture the CPU's `Marketing Name`, `Compute Unit`, and `SIMDs per CU`,
    which produces nonsense (Intel SKU, 0 SIMDs) downstream.
    """
    parts = re.split(r"\n(?=Agent\s+\d+\s*\n)", text)
    return [chunk for chunk in parts if "Device Type:" in chunk]


def first_gpu_agent(text):
    """Return the first agent block whose Device Type is GPU, or '' if none."""
    for chunk in split_agents(text):
        if re.search(r"Device Type:\s+GPU", chunk):
            return chunk
    return ""


def parse_rocminfo(text):
    gpu = first_gpu_agent(text)
    if not gpu:
        return {
            "gfx": "unknown",
            "sku": "no-gpu-detected",
            "cu_count": 0,
            "simd_per_cu": 0,
            "wavefront_size": 64,
            "workgroup_max": 1024,
            "chip_id": "",
            "sclk_mhz_boost": 0,
        }
    return {
        "gfx": search_str(r"Name:\s+(gfx\w+)", gpu),
        "sku": search_str(r"Marketing Name:\s+(.*)", gpu),
        "cu_count": search_int(r"Compute Unit:\s+(\d+)", gpu),
        "simd_per_cu": search_int(r"SIMDs per CU\s*:\s*(\d+)", gpu, 4, re.IGNORECASE),
        "wavefront_size": search_int(r"Wavefront Size:\s+(\d+)", gpu, 64),
        "workgroup_max": search_int(r"Workgroup Max Size:\s+(\d+)", gpu, 1024),
        "chip_id": search_str(r"Chip ID:\s+(\w+)", gpu, ""),
        "sclk_mhz_boost": search_int(r"Max Clock Freq\. \(MHz\):\s+(\d+)", gpu),
    }


def parse_cache(text):
    return {
        "lds_per_cu_kb": search_int(
            r"Local Memory Size.*?:\s+(\d+)\s*KB", text, 64, re.DOTALL | re.IGNORECASE
        ),
        "l1_per_cu_kb": search_int(r"L1:\s+(\d+)\s*KB", text, 32),
        "l2_total_mb": search_int(r"L2:\s+(\d+)\s*KB", text, 256 * 1024) // 1024,
    }


def memory_info():
    cap_gb = 0
    mclk_mhz = 0
    try:
        data = json.loads(run("rocm-smi --showmeminfo vram --json"))
        first = next(iter(data.values()))
        cap_gb = round(int(first.get("VRAM Total Memory (B)", 0)) / 1024**3, 1)
    except Exception:
        pass
    try:
        data = json.loads(run("rocm-smi -m --json"))
        first = next(iter(data.values()))
        match = re.search(r"(\d+)", first.get("mclk clock level", ""))
        mclk_mhz = int(match.group(1)) if match else 0
    except Exception:
        pass
    return cap_gb, mclk_mhz


def lookup_datasheet(sku):
    for name, values in DATASHEET.items():
        if name.lower() in sku.lower():
            return name, values
    return None, None


def parse_chip_id(value):
    """Return int chip id from rocminfo's `12345(0xABCD)` style string, or 0."""
    if not value:
        return 0
    match = re.search(r"0x([0-9a-fA-F]+)", value)
    if match:
        return int(match.group(1), 16)
    try:
        return int(value)
    except ValueError:
        return 0


def resolve_sku(marketing_name, chip_id_text):
    """Map (marketing_name, chip_id) → canonical SKU key in DATASHEET, or None.

    Marketing-name path covers boxes where rocminfo reports e.g.
    'AMD Instinct MI300X'. Chip-id path covers boxes that only report the
    generic 'AMD Radeon Graphics' (MI308 boards do this in practice).
    """
    name, values = lookup_datasheet(marketing_name)
    if values is not None:
        return name, values, "marketing-name"
    cid = parse_chip_id(chip_id_text)
    if cid in CHIP_ID_TO_SKU:
        sku = CHIP_ID_TO_SKU[cid]
        return sku, DATASHEET.get(sku), f"chip-id 0x{cid:x}"
    return None, None, None


def main():
    rocminfo = run("rocminfo")
    base = parse_rocminfo(rocminfo)
    cache = parse_cache(rocminfo)
    cap_gb, mclk_mhz = memory_info()
    sku_key, ds, sku_source = resolve_sku(base.get("sku", ""), base.get("chip_id", ""))
    family_match = sku_key or base.get("sku", "")

    out = {
        "sku": sku_key or base["sku"],
        "marketing_name": base["sku"],
        "gfx": base["gfx"],
        "cu_count": base["cu_count"],
        "xcd_count": 8 if any(x in family_match for x in XCD_FAMILIES) else 0,
        "simd_per_cu": base["simd_per_cu"],
        "wavefront_size": base["wavefront_size"],
        "workgroup_max": base["workgroup_max"],
        "chip_id": base["chip_id"],
        "vgpr_per_simd": 512,
        "sgpr_per_simd": 800,
        "lds_per_cu_kb": cache["lds_per_cu_kb"],
        "l1_per_cu_kb": cache["l1_per_cu_kb"],
        "l2_total_mb": cache["l2_total_mb"],
        "l2_per_xcd_mb": cache["l2_total_mb"] // max(1, 8),
        "hbm_capacity_gb": cap_gb,
        "sclk_mhz_boost": base["sclk_mhz_boost"],
        "mclk_mhz": mclk_mhz,
    }

    if ds:
        out["peak"] = {
            "hbm_bw_tbps": ds["hbm_bw_tbps"],
            "bf16_mfma_tflops": ds["bf16_tflops"],
            "fp16_mfma_tflops": ds["bf16_tflops"],
            "fp8_mfma_tflops": ds["fp8_tflops"],
            "fp32_mfma_tflops": ds["fp32_tflops"],
        }
        memory_haircut = 0.80
        compute_haircut = 0.85
        out["haircut"] = {"memory": memory_haircut, "compute": compute_haircut}
        peak = out["peak"]
        out["deliverable"] = {
            "hbm_bw_tbps": round(peak["hbm_bw_tbps"] * memory_haircut, 2),
            "bf16_mfma_tflops": round(peak["bf16_mfma_tflops"] * compute_haircut, 1),
            "fp16_mfma_tflops": round(peak["fp16_mfma_tflops"] * compute_haircut, 1),
            "fp8_mfma_tflops": round(peak["fp8_mfma_tflops"] * compute_haircut, 1),
            "fp32_mfma_tflops": round(peak["fp32_mfma_tflops"] * compute_haircut, 1),
        }
        deliverable = out["deliverable"]
        bw = deliverable["hbm_bw_tbps"] * 1e12
        out["ridge_point_flops_per_byte"] = {
            "bf16": round(deliverable["bf16_mfma_tflops"] * 1e12 / bw, 1),
            "fp16": round(deliverable["fp16_mfma_tflops"] * 1e12 / bw, 1),
            "fp8": round(deliverable["fp8_mfma_tflops"] * 1e12 / bw, 1),
            "fp32": round(deliverable["fp32_mfma_tflops"] * 1e12 / bw, 1),
        }
    else:
        out["peak"] = {
            "_warning": (
                "SKU not in DATASHEET; marketing_name "
                f"{base.get('sku', '')!r} chip_id {base.get('chip_id', '')!r}. "
                "Fill peak/deliverable manually or extend DATASHEET / "
                "CHIP_ID_TO_SKU in query_hw.py."
            )
        }
        out["haircut"] = {"memory": 0.80, "compute": 0.85}

    out["captured_at"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out["captured_on"] = socket.gethostname()
    out["rocm_version"] = run("hipconfig --version").strip() or "unknown"
    notes = (
        "rocminfo (first GPU agent) + rocm-smi + DATASHEET lookup. "
        "Haircut defaults 0.80/0.85; override after first saturating benchmark."
    )
    if sku_source:
        notes += f" SKU resolved via {sku_source}."
    out["source_notes"] = notes
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
