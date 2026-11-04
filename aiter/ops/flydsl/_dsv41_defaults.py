# SPDX-License-Identifier: MIT
"""Architecture-conditional defaults for the DeepSeek-V4.1 a16wmix switches.

These paths are validated on gfx942 (CDNA3 / MI300X / MI325X) and left off
elsewhere, so the switches default on there and off everywhere else instead of
requiring a deployment to export them. Mirrors the contract sglang's
``srt/environ.py`` uses for ``_default_hip``: resolved lazily, cached, and any
failure resolves to off rather than raising at import time.

Setting the environment variable always wins, so a single switch can still be
turned off for an A/B.
"""

import functools


@functools.lru_cache(maxsize=1)
def _is_gfx942() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx942"
    except Exception:
        return False


def dsv41_on(value: str = "1", off: str = "0") -> str:
    """``value`` on gfx942, ``off`` elsewhere, as the string env defaults use."""
    return value if _is_gfx942() else off
