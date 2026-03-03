"""Entropy and probability helpers."""

from typing import Dict
import math


def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    """Normalize a sparse distribution over arbitrary string keys."""
    cleaned = {k: max(0.0, float(v)) for k, v in dist.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        if not cleaned:
            return {}
        u = 1.0 / len(cleaned)
        return {k: u for k in cleaned}
    return {k: cleaned[k] / total for k in cleaned}


def shannon_entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy in bits for a discrete distribution."""
    p = normalize(dist)
    return -sum(v * math.log2(v) for v in p.values() if v > 0.0)


# Backward-compatible aliases.
normalize_distribution_generic = normalize
shannon_entropy_bits_generic = shannon_entropy
