import bisect
import random
from typing import Dict, Tuple, List

# Cache for cumulative probability distributions: (alpha, n) -> List[float]
_ZIPF_CACHE: Dict[Tuple[float, int], List[float]] = {}


def _get_zipf_cdf(alpha: float, n: int) -> List[float]:
    key = (float(alpha), int(n))
    if key not in _ZIPF_CACHE:
        # Precompute CDF once
        harmonic = sum(1.0 / (i ** alpha) for i in range(1, n + 1))
        inv_harmonic = 1.0 / harmonic
        cum_prob = 0.0
        cdf = []
        for i in range(1, n + 1):
            cum_prob += inv_harmonic / (i ** alpha)
            cdf.append(cum_prob)
        # Ensure the last element is exactly 1.0
        cdf[-1] = 1.0
        _ZIPF_CACHE[key] = cdf
    return _ZIPF_CACHE[key]


def zipf_distribution(alpha: float, n: int) -> int:
    """Return a 1-based index drawn from Zipf(alpha, n)."""
    cdf = _get_zipf_cdf(alpha, n)
    rand = random.random()
    idx = bisect.bisect_left(cdf, rand)
    return min(idx + 1, n)
