"""
tome_patch.py
PixArt BasicTransformerBlock에 Token Merging을 monkey-patch로 설치.

diffusers 소스를 수정하지 않고 types.MethodType으로 block.forward를 교체.

참고:
  BasicTransformerBlock.forward:
    diffusers/models/attention.py (lines 960-1079)
  PixArt norm_type == "ada_norm_single" 전용 구현.
"""

import types
from typing import Any

import torch
from torch import Tensor

from tome_core import bipartite_soft_matching, compute_r


# ---------------------------------------------------------------------------
# Patched block forward factory
# ---------------------------------------------------------------------------

def _make_tome_forward(merge_ratio: float):
    """
    merge_ratio를 capture하여 BasicTransformerBlock용 patched forward 반환.
    PixArt의 norm_type == "ada_norm_single" 에 특화.
    """

    def tome_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        class_labels: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        # --- PixArt는 norm_type == "ada_norm_single" ---
        batch_size = hidden_states.shape[0]
        N = hidden_states.shape[1]

        # 0. AdaLN modulation 계산
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 1. Norm + modulate
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # 2. Token Merging — norm_hidden_states만 merge (residual hidden_states는 절대 merge 안 함)
        r = compute_r(N, merge_ratio)
        if r > 0:
            merge_fn, unmerge_fn = bipartite_soft_matching(norm_hidden_states, r)
            norm_hs_merged = merge_fn(norm_hidden_states)   # (B, N-r, C)
        else:
            norm_hs_merged = norm_hidden_states
            merge_fn = unmerge_fn = None

        # 3. Self-Attention (줄어든 시퀀스)
        attn_output = self.attn1(
            norm_hs_merged,
            encoder_hidden_states=None,   # self-attn
            attention_mask=None,          # PixArt self-attn: mask 없음
            **cross_attention_kwargs,
        )

        # 4. Residual add 전에 unmerge → hidden_states는 항상 full N 유지
        if r > 0:
            attn_output = unmerge_fn(attn_output)           # (B, N-r, C) → (B, N, C)

        attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states         # residual은 항상 full N

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 5. Cross-Attention (PixArt ada_norm_single: norm2 미적용)
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 6. FFN
        norm_hidden_states_ff = self.norm2(hidden_states)
        norm_hidden_states_ff = norm_hidden_states_ff * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states_ff)
        ff_output = gate_mlp * ff_output
        hidden_states = ff_output + hidden_states

        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    return tome_forward


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_TOME_MARKER = "_tome_patched"


def install_tome(
    transformer,
    merge_ratio: float,
    block_start: int = 0,
    block_end: int = 28,
) -> dict:
    """
    transformer.transformer_blocks[block_start:block_end]에 ToMe patch 설치.

    Args:
        transformer  : PixArtTransformer2DModel 인스턴스
        merge_ratio  : 0.0~0.5, 각 block에서 merge할 토큰 비율
        block_start  : 첫 번째 patch 적용 block 인덱스 (포함)
        block_end    : 마지막 patch 적용 block 인덱스 (미포함)

    Returns:
        config dict (로깅용)
    """
    n_blocks = len(transformer.transformer_blocks)
    block_end = min(block_end, n_blocks)

    patched_fn = _make_tome_forward(merge_ratio)
    for i in range(block_start, block_end):
        block = transformer.transformer_blocks[i]
        block.forward = types.MethodType(patched_fn, block)
        setattr(block, _TOME_MARKER, True)

    n_patched = block_end - block_start
    return {
        "merge_ratio": merge_ratio,
        "block_start": block_start,
        "block_end":   block_end,
        "n_patched":   n_patched,
        "total_blocks": n_blocks,
    }


def uninstall_tome(transformer):
    """
    모든 block의 monkey-patch를 제거 (diffusers 원본 class forward 복원).
    인스턴스 forward를 삭제하면 class method로 자동 폴백됨.
    """
    for block in transformer.transformer_blocks:
        if getattr(block, _TOME_MARKER, False):
            try:
                del block.forward
            except AttributeError:
                pass
            try:
                delattr(block, _TOME_MARKER)
            except AttributeError:
                pass
