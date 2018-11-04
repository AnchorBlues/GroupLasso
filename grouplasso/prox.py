from numba import jit, f8, i1
import numpy as np


@jit(f8[:](f8[:], f8, i1[:]))
def _prox(coef, thresh, group_ids):
    """
    Proximal operator.
    Group sparsity case: apply group sparsity operator
    """
    result = np.zeros_like(coef).astype(np.float64)
    unique_group_ids = np.unique(group_ids)
    for group_id in unique_group_ids:
        target_idx = group_ids == group_id
        target_coef = coef[target_idx]
        group_norm = np.linalg.norm(target_coef, 2)
        multiplier = max(0, 1 - thresh / group_norm)
        result[target_idx] = multiplier * target_coef

    return result
