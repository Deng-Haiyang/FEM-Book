"""Boundary-condition treatment and linear-equation solution."""
from __future__ import annotations

import numpy as np

from model import Model


def matrix_rank_info(K: np.ndarray, tol: float = 1.0e-10) -> tuple[int, bool, float | None]:
    """Return numerical rank, singular flag and condition number when available."""
    rank = int(np.linalg.matrix_rank(K, tol=tol))
    singular = rank < K.shape[0]
    cond = None if singular else float(np.linalg.cond(K))
    return rank, singular, cond


def solve_by_reduction(model: Model, K: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve Kd=f by partitioning known and unknown displacement dofs.

    Partition notation follows the homework statement:
    d_E is the known displacement vector; d_F is the unknown displacement vector.
    K_FF d_F = f_F - K_FE d_E.
    """
    neq = model.neq
    all_dof = np.arange(neq, dtype=int)
    fixed = np.asarray(model.fixed_dof, dtype=int)
    fixed_values = np.asarray(model.fixed_value, dtype=float)

    if len(np.unique(fixed)) != len(fixed):
        raise ValueError("Duplicate fixed dofs are not allowed.")
    if np.any(fixed < 0) or np.any(fixed >= neq):
        raise ValueError("fixed_dof contains invalid dof numbers.")

    free = np.setdiff1d(all_dof, fixed)
    d = np.zeros(neq, dtype=float)
    d[fixed] = fixed_values

    K_FF = K[np.ix_(free, free)]
    K_FE = K[np.ix_(free, fixed)]
    rhs = f[free] - K_FE @ fixed_values

    d[free] = np.linalg.solve(K_FF, rhs)
    reactions = K @ d - f

    info = {
        "free_dof_0_based": free,
        "fixed_dof_0_based": fixed,
        "K_FF": K_FF,
        "rhs": rhs,
        "rank_K_FF": int(np.linalg.matrix_rank(K_FF)),
        "singular_K_FF": bool(np.linalg.matrix_rank(K_FF) < K_FF.shape[0]),
        "condition_K_FF": float(np.linalg.cond(K_FF)),
    }
    return d, reactions, info
