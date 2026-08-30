"""Evidence Sufficiency Checker — Signal B (Contribution mới).

Kiểm tra xem các Claims được LLM sinh ra có thực sự được hỗ trợ bởi
Context hay không, sử dụng NLI (DeBERTa — đã load sẵn, không tốn thêm VRAM).

SE_semantic = 0 chỉ chứng minh consistency, KHÔNG chứng minh correctness.
Evidence Sufficiency lấp đầy khoảng trống đó.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from uncertainty_rag.modality.base import ContextChunk, ModalityHandler
from uncertainty_rag.models.nli_model import NLIModel

logger = logging.getLogger(__name__)


@dataclass
class EvidenceProfile:
    """Kết quả kiểm tra Evidence Sufficiency."""

    evidence_ratio: float  # supported / total claims (0.0 → 1.0)
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    claim_evidence_map: dict = field(default_factory=dict)


class EvidenceChecker:
    """Kiểm tra Evidence Sufficiency bằng NLI.

    Với mỗi claim, quét tất cả chunk trong context:
      NLI(premise=chunk_text, hypothesis=claim) → entailment_score
    Nếu max(entailment_score) >= threshold → claim SUPPORTED.
    Nếu không → claim UNSUPPORTED → cần RETRIEVE thêm.
    """

    def __init__(
        self,
        nli_model: NLIModel,
        modality_handler: Optional[ModalityHandler] = None,
        support_threshold: float = 0.5,
    ) -> None:
        self.nli = nli_model
        self.handler = modality_handler
        self.support_threshold = support_threshold

    def _get_chunk_text(self, chunk: ContextChunk) -> str:
        """Lấy text representation của chunk (hỗ trợ mọi modality)."""
        if self.handler:
            return self.handler.get_chunk_text_repr(chunk)
        if isinstance(chunk.content, str):
            return chunk.content
        return str(chunk.content)

    def check(
        self,
        claims: list[str],
        chunks: list[ContextChunk],
    ) -> EvidenceProfile:
        """Kiểm tra xem mỗi claim có được hỗ trợ bởi ít nhất 1 chunk hay không."""
        if not claims:
            return EvidenceProfile(evidence_ratio=1.0)

        if not chunks:
            return EvidenceProfile(
                evidence_ratio=0.0,
                unsupported_claims=list(claims),
            )

        chunk_texts = [(chunk.id, self._get_chunk_text(chunk)) for chunk in chunks]

        supported = []
        unsupported = []
        claim_evidence_map = {}

        # Batch NLI: tạo tất cả cặp (chunk_text, claim) rồi chạy 1 lần
        all_pairs = []
        pair_indices = []

        for ci, claim in enumerate(claims):
            for chi, (chunk_id, chunk_text) in enumerate(chunk_texts):
                if chunk_text.strip():
                    all_pairs.append((chunk_text, claim))
                    pair_indices.append((ci, chi))

        if not all_pairs:
            return EvidenceProfile(
                evidence_ratio=0.0,
                unsupported_claims=list(claims),
            )

        all_scores = self.nli.predict_batch(all_pairs)

        claim_scores: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(claims))}
        for (ci, chi), (contra, neutral, entail) in zip(pair_indices, all_scores):
            claim_scores[ci].append((chi, entail))

        for ci, claim in enumerate(claims):
            scores = claim_scores.get(ci, [])
            if not scores:
                unsupported.append(claim)
                claim_evidence_map[claim] = ("NONE", 0.0)
                continue

            best_chi, best_score = max(scores, key=lambda x: x[1])
            best_chunk_id = chunk_texts[best_chi][0]

            if best_score >= self.support_threshold:
                supported.append(claim)
                claim_evidence_map[claim] = (best_chunk_id, round(best_score, 4))
                logger.info(
                    f"  [Evidence] SUPPORTED: \"{claim[:80]}\" "
                    f"← Chunk {best_chunk_id[:12]}.. (entailment={best_score:.4f})"
                )
            else:
                unsupported.append(claim)
                claim_evidence_map[claim] = (best_chunk_id, round(best_score, 4))
                logger.info(
                    f"  [Evidence] UNSUPPORTED: \"{claim[:80]}\" "
                    f"(best={best_score:.4f} < {self.support_threshold})"
                )

        total = len(claims)
        ratio = len(supported) / total if total > 0 else 1.0

        logger.info(
            f"  [Evidence Summary] {len(supported)}/{total} claims supported "
            f"→ evidence_ratio={ratio:.4f}"
        )

        return EvidenceProfile(
            evidence_ratio=ratio,
            supported_claims=supported,
            unsupported_claims=unsupported,
            claim_evidence_map=claim_evidence_map,
        )
