import numpy as np


def lhs_sample(parameters: dict, n_samples: int, seed: int | None = None) -> dict:
    """
    Latin Hypercube Sampling (LHS) for independent parameters.

    parameters: dict like {"a":[min,max], "b":[min,max], ...}
        If min == max, that parameter is treated as a constant (no sampling).
    n_samples:  number of samples
    seed:       RNG seed (optional)

    Returns: dict of arrays length n_samples, plus "iteration" = 0..n_samples-1
    """
    rng = np.random.default_rng(seed)

    if not isinstance(parameters, dict) or len(parameters) == 0:
        raise ValueError("parameters must be a non-empty dict of {name: [min, max]}.")

    out = {"iteration": np.arange(n_samples, dtype=int)}
    cut = np.linspace(0.0, 1.0, n_samples + 1)

    for name, bounds in parameters.items():
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"Parameter '{name}' must be [min, max].")

        lo, hi = float(bounds[0]), float(bounds[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Parameter '{name}' requires finite bounds.")

        if hi < lo:
            raise ValueError(f"Parameter '{name}' requires max >= min (got {lo} > {hi}).")

        if hi == lo:
            out[name] = np.full(n_samples, lo, dtype=float)
            continue

        u = rng.random(n_samples)
        points_01 = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points_01)

        out[name] = lo + points_01 * (hi - lo)

    return out

