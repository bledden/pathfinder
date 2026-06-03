"""Small self-contained utilities for the qLDPC decoder experiments."""
import math


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for k successes in n trials -> (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))
