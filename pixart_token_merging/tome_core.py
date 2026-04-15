"""
tome_core.py
Token Merging (ToMe) 알고리즘 핵심 구현 — 최적화 버전.

핵심 개선사항 vs 초기 구현:
  - nonzero() CPU-GPU 동기화 완전 제거
  - x.clone() 전체 복사 제거 → A/B 파티션 서브텐서만 조작
  - 클로저 반환: 인덱스를 한 번만 계산, merge/unmerge 재사용
  - scatter 연산 크기를 N → nB (절반)로 축소

Reference: Bolya et al., "Token Merging: Your ViT But Faster" (ICLR 2023)
"""

from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Bipartite soft matching: merge / unmerge 클로저 쌍을 반환.

    인덱스 계산은 여기서 한 번만 수행 (no_grad, GPU only — nonzero 없음).

    Args:
        metric : (B, N, C)  유사도 계산 feature (norm_hidden_states 권장)
        r      : merge할 쌍 수 (≤ nA = ceil(N/2))

    Returns:
        merge_fn   : (B, N, C) → (B, N-r, C)
        unmerge_fn : (B, N-r, C) → (B, N, C)
    """
    B, N, C = metric.shape
    nA = (N + 1) // 2   # A 파티션 크기 (짝수 인덱스, ceil)
    nB = N // 2         # B 파티션 크기 (홀수 인덱스)
    r = min(r, nA)

    with torch.no_grad():
        # A = 짝수 인덱스, B = 홀수 인덱스 (partition by stride slicing — no copy)
        a = F.normalize(metric[:, ::2], dim=-1)    # (B, nA, C)
        b = F.normalize(metric[:, 1::2], dim=-1)   # (B, nB, C)

        # A→B 코사인 유사도
        scores = torch.bmm(a, b.transpose(1, 2))   # (B, nA, nB)

        # 각 A 토큰의 최선 B 매칭
        node_max, node_idx = scores.max(dim=-1)    # (B, nA)

        # 유사도 내림차순으로 A 정렬 → 상위 r개 선택
        edge_idx = node_max.argsort(dim=-1, descending=True).unsqueeze(-1)  # (B, nA, 1)

        src_idx = edge_idx[:, :r]          # (B, r, 1) — merge될 A 로컬 인덱스
        unm_idx = edge_idx[:, r:]          # (B, nA-r, 1) — 유지될 A 로컬 인덱스
        dst_idx = node_idx.unsqueeze(-1).gather(1, src_idx)  # (B, r, 1) — B 로컬 인덱스

    # -----------------------------------------------------------------------
    # 클로저 변수: src_idx, unm_idx, dst_idx, B, r, nA, nB, N
    # -----------------------------------------------------------------------

    def merge_fn(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """
        (B, N, C) → (B, N-r, C)

        출력 레이아웃: [unmerged-A | B-partition]
          - 앞 (nA-r) 토큰: merge되지 않은 A 파티션 토큰
          - 뒤 nB 토큰    : B 파티션 토큰 (일부는 A 값이 평균됨)
        """
        _C = x.shape[-1]
        _unm_len = unm_idx.shape[1]   # nA - r

        # A 서브텐서에서 gather (clone 없음)
        a_part = x[:, ::2]   # view (B, nA, C) — no copy
        src = a_part.gather(1, src_idx.expand(B, r, _C))           # (B, r, C)
        unm = a_part.gather(1, unm_idx.expand(B, _unm_len, _C))    # (B, nA-r, C)

        # B 서브텐서 복사 (scatter_add_ in-place 필요)
        dst = x[:, 1::2].clone()    # (B, nB, C)

        if mode == "mean":
            # 카운트: (B, nB, 1) — 브로드캐스트로 C 축 불필요
            cnt = torch.ones(B, nB, 1, device=x.device, dtype=x.dtype)
            cnt.scatter_add_(
                1, dst_idx,
                torch.ones(B, r, 1, device=x.device, dtype=x.dtype)
            )
            dst.scatter_add_(1, dst_idx.expand(B, r, _C), src)
            dst /= cnt          # broadcast over C
        else:
            dst.scatter_add_(1, dst_idx.expand(B, r, _C), src)

        return torch.cat([unm, dst], dim=1)   # (B, nA-r + nB, C)

    def unmerge_fn(x: torch.Tensor) -> torch.Tensor:
        """
        (B, N-r, C) → (B, N, C)

        merge_fn 출력의 레이아웃 [unm | dst] 를 원래 순서로 복원:
          - 홀수 위치(B): dst 그대로
          - 짝수 위치 중 unm: 원래 값 복원
          - 짝수 위치 중 src: 대응 dst 값 복제
        """
        _C = x.shape[-1]
        _unm_len = unm_idx.shape[1]

        unm = x[:, :_unm_len]          # (B, nA-r, C)
        dst = x[:, _unm_len:]          # (B, nB, C)

        # A 파티션 재구성 (nA 크기 버퍼)
        out_a = torch.empty(B, nA, _C, device=x.device, dtype=x.dtype)
        out_a.scatter_(1, unm_idx.expand(B, _unm_len, _C), unm)
        out_a.scatter_(1, src_idx.expand(B, r, _C),
                       dst.gather(1, dst_idx.expand(B, r, _C)))

        # 짝수/홀수 위치로 인터리빙
        out = torch.empty(B, N, _C, device=x.device, dtype=x.dtype)
        out[:, ::2]  = out_a
        out[:, 1::2] = dst
        return out

    return merge_fn, unmerge_fn


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_r(N: int, merge_ratio: float) -> int:
    """
    토큰 수 N과 merge ratio에서 실제 merge 할 쌍 수 r 계산.
    A 파티션 크기(ceil(N/2))를 넘지 않도록 클리핑.
    """
    n_a = (N + 1) // 2
    r = int(N * merge_ratio)
    r = min(r, n_a)
    r = max(r, 0)
    return r
