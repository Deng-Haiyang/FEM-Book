"""Direct stiffness assembly routines."""
from __future__ import annotations

import numpy as np

from model import Model, build_lm
from element import element_stiffness


def assemble_global_stiffness(model: Model) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Assemble K using K[LM[a,e], LM[b,e]] += Ke[a,b]."""
    K = np.zeros((model.neq, model.neq), dtype=float)
    LM = build_lm(model)
    element_matrices: list[np.ndarray] = []

    for e in range(model.nel):
        Ke = element_stiffness(model, e)
        element_matrices.append(Ke)
        for a in range(Ke.shape[0]):
            A = LM[a, e]
            for b in range(Ke.shape[1]):
                B = LM[b, e]
                K[A, B] += Ke[a, b]

    return K, LM, element_matrices


def assemble_force_vector(model: Model) -> np.ndarray:
    """Build the global force vector from force_dof and force_value."""
    f = np.zeros(model.neq, dtype=float)
    for dof, value in zip(model.force_dof, model.force_value):
        if dof < 0 or dof >= model.neq:
            raise ValueError(f"force_dof contains invalid dof number {dof + 1}.")
        f[dof] += value
    return f
