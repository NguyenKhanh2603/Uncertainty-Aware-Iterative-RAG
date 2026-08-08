"""Ablation study runner — systematically evaluates component contributions.

Ablation variants:
1. No Pruning: Skip Step 2 entirely
2. No Active Retrieval: Skip Step 3 entirely
3. No Uncertainty Decomposition: Use SE_total only, no aleatoric/epistemic split
4. Token-level only: Replace semantic clustering with raw token entropy (like FLARE)
5. Random Routing: Randomly decide prune vs. retrieve
6. Text-Only on Multimodal: Use linearized text only (no images/table structure)
7. Fixed M only: Disable adaptive M (compare with U4)
8. Fixed Thresholds only: Disable adaptive thresholds (compare with U2)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from uncertainty_rag.config import Config


class AblationVariant(str, Enum):
    """Available ablation configurations."""

    FULL = "full"  # Our complete method
    NO_PRUNE = "no_prune"
    NO_RETRIEVE = "no_retrieve"
    NO_DECOMPOSITION = "no_decomposition"
    TOKEN_LEVEL_ONLY = "token_level_only"
    RANDOM_ROUTING = "random_routing"
    TEXT_ONLY_MULTIMODAL = "text_only_multimodal"
    FIXED_M = "fixed_m"  # Disable adaptive M (U4 ablation)
    FIXED_THRESHOLDS = "fixed_thresholds"  # Disable adaptive thresholds (U2 ablation)


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""

    variant: AblationVariant
    description: str
    config_overrides: dict

    def apply(self, base_config: Config) -> Config:
        """Apply overrides to create the ablation config."""
        cfg = deepcopy(base_config)
        for path, value in self.config_overrides.items():
            parts = path.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        return cfg


# ── Pre-defined ablation variants ───────────────────────────────────────────────

ABLATION_VARIANTS = {
    AblationVariant.FULL: AblationConfig(
        variant=AblationVariant.FULL,
        description="Complete method with all components",
        config_overrides={},
    ),
    AblationVariant.NO_PRUNE: AblationConfig(
        variant=AblationVariant.NO_PRUNE,
        description="Skip pruning step — never prune context",
        config_overrides={"pruning.pre_filter_enabled": False},
    ),
    AblationVariant.NO_RETRIEVE: AblationConfig(
        variant=AblationVariant.NO_RETRIEVE,
        description="Skip retrieval step — no active evidence gathering",
        config_overrides={"retrieval.top_k": 0},
    ),
    AblationVariant.FIXED_M: AblationConfig(
        variant=AblationVariant.FIXED_M,
        description="U4 ablation: always use fixed M (no adaptive sampling)",
        config_overrides={"sampling.adaptive_M_enabled": False},
    ),
    AblationVariant.FIXED_THRESHOLDS: AblationConfig(
        variant=AblationVariant.FIXED_THRESHOLDS,
        description="U2 ablation: use fixed thresholds (no adaptive calibration)",
        config_overrides={"thresholds.mode": "fixed"},
    ),
}


def get_ablation_configs(base_config: Config) -> dict[str, Config]:
    """Generate all ablation configurations from a base config.

    Returns:
        Dict mapping variant name → modified Config.
    """
    configs = {}
    for variant, ablation in ABLATION_VARIANTS.items():
        configs[variant.value] = ablation.apply(base_config)
    return configs
