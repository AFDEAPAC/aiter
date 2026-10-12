"""Per-row fp8 quantization of the a16wmix stage-2 intermediate, several rows per program.

Same contract and arithmetic as aiter's dynamic_per_token_scaled_quant for DTYPE_O = fp8 on gfx942
(quant_kernels.cu data_to_per_row_scale + scaled_quant_vgpr_impl, aiter_opus_plus.h fp32_to_fp8_scaled_x2):
  scale = absmax(row) * float(1/240);  inv = v_rcp_f32(scale);  q = cvt_fp8(med3(x * inv, -240, 240))
Rows at or past num_rows[0] are left unwritten, as there. That kernel launches one block per row, which at
prefill (~159k rows of 384 values) runs at ~0.5 TB/s; this one does BLOCK_R rows per program. A zero row gets
scale 0 and q 0 (aiter's would produce rcp(0) * 0 = NaN).
"""
import torch
import triton
import triton.language as tl

_INV_FP8_MAX = float(torch.tensor(1.0 / 240.0, dtype=torch.float32))


@triton.jit
def _rowquant_fp8_kernel(x_ptr, q_ptr, s_ptr, nrows_ptr, COLS: tl.constexpr, BLOCK_R: tl.constexpr,
                         BLOCK_C: tl.constexpr, INV_MAX: tl.constexpr):
    n = tl.load(nrows_ptr)
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    rmask = rows < n
    m = rmask[:, None] & (cols < COLS)[None, :]
    offs = rows.to(tl.int64)[:, None] * COLS + cols[None, :]
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    scale = tl.max(tl.abs(x), axis=1) * INV_MAX
    inv = tl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [scale], dtype=tl.float32, is_pure=True, pack=1)
    inv = tl.where(scale > 0, inv, 0.0)
    y = tl.minimum(tl.maximum(x * inv[:, None], -240.0), 240.0)
    tl.store(q_ptr + offs, y.to(tl.float8e4b8), mask=m)
    tl.store(s_ptr + rows, scale, mask=rmask)


def rowquant_fp8(q: torch.Tensor, x: torch.Tensor, scale: torch.Tensor, num_rows: torch.Tensor, live_rows: int,
                 block_r: int = 16) -> None:
    """q[:n] = fp8(x[:n] / scale[:n]), scale[:n, 0] = absmax(x[:n]) / 240 with n = num_rows[0] <= live_rows."""
    cols = x.shape[-1]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and q.is_contiguous() and scale.is_contiguous()
    grid = (triton.cdiv(live_rows, block_r),)
    _rowquant_fp8_kernel[grid](x, q, scale, num_rows, COLS=cols, BLOCK_R=block_r,
                               BLOCK_C=triton.next_power_of_2(cols), INV_MAX=_INV_FP8_MAX, num_warps=4)
