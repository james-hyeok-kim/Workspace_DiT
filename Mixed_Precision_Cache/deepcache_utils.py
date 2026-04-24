"""
deepcache_utils.py — Minimal version for Mixed_Precision_Cache.

Provides:
  - DeepCacheState : shared state with step_idx (required by eval_utils.py)
  - install_step_aware_quant : monkey-patch transformer.forward to track step_idx

The full DeepCache caching logic is not included here (Phase 6 caching
combination is a separate session). Only step tracking is needed.
"""

import types

import torch


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DeepCacheState:
    """Minimal state object. eval_utils.py calls reset() and next_image()."""

    def __init__(self):
        self.step_idx: int = 0
        self._image_idx: int = 0

    def reset(self):
        self.step_idx = 0

    def next_image(self):
        self._image_idx += 1


# ---------------------------------------------------------------------------
# install_step_aware_quant
# ---------------------------------------------------------------------------

def install_step_aware_quant(transformer, state: DeepCacheState) -> DeepCacheState:
    """
    Monkey-patch transformer.forward to increment state.step_idx once per denoising step.

    StepAwareMixedLinear layers read state.step_idx - 1 at forward time to determine
    the current (0-indexed) step, then pick W3 or W4 from bit_schedule.

    Must be called AFTER apply_step_aware_mixed_quantization.
    """
    orig_forward = transformer.forward.__func__

    def _step_counter(self, *args, **kwargs):
        state.step_idx += 1
        return orig_forward(self, *args, **kwargs)

    transformer.forward = types.MethodType(_step_counter, transformer)
    return state


def uninstall_step_aware_quant(transformer, state: DeepCacheState):
    """Remove the step_counter monkey-patch (restores original transformer.forward)."""
    if hasattr(transformer.forward, "__func__"):
        return  # already unpatched
    # Reconstruct original: __func__ is the wrapped closure; retrieve via __wrapped__ if set
    # Simpler: re-load from class
    transformer.forward = types.MethodType(
        type(transformer).forward, transformer
    )
