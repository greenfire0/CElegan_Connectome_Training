# n.py
import numpy as np
from Worm_Env.weight_dict import dict as weights_dict

def flatten_weights(d):
    vals = []
    for sub in d.values():
        vals.extend(sub.values())
    return np.asarray(vals, dtype=float)

def percentile_rank(subset: np.ndarray, x: float) -> float:
    """Empirical percentile rank P(X ≤ x) within subset."""
    if subset.size == 0:
        return float("nan")
    return (np.sum(subset <= x) / subset.size) * 100.0

def main():
    w = flatten_weights(weights_dict)
    pos = w[w > 0]
    neg = w[w < 0]

    p_20_in_pos   = percentile_rank(pos, 20.0)
    p_m20_in_neg  = percentile_rank(neg, -20.0)

    # Optional: where does zero sit within positives/negatives?
    p_0_in_pos    = percentile_rank(pos, 0.0)   # should be ~0th
    p_0_in_neg    = percentile_rank(neg, 0.0)   # should be ~100th

    print(f"20 is at the {p_20_in_pos:.2f}th percentile within positive weights")
    print(f"-20 is at the {p_m20_in_neg:.2f}th percentile within negative weights")
    print(f"0 is at the {p_0_in_pos:.2f}th percentile within positive weights")
    print(f"0 is at the {p_0_in_neg:.2f}th percentile within negative weights")
    print(min(neg),max(pos))

if __name__ == "__main__":
    main()
