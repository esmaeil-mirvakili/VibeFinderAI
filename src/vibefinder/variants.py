"""Ablation variant structure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantConfig:
    name: str
    use_multi_step_reasoning: bool
    use_critic_revision: bool
    use_lyric_retriever: bool
