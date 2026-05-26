"""Post-processing and formatted output helpers."""
from __future__ import annotations

import numpy as np

from model import Model
from element import element_geometry, element_stress_force, ElementResult


def extract_element_displacement(model: Model, LM: np.ndarray, e: int, d: np.ndarray) -> np.ndarray:
    return d[LM[:, e]]


def postprocess_elements(model: Model, LM: np.ndarray, d: np.ndarray) -> list[ElementResult]:
    results: list[ElementResult] = []
    for e in range(model.nel):
        d_e = extract_element_displacement(model, LM, e, d)
        results.append(element_stress_force(model, e, d_e))
    return results


def sparsity_ratio(K: np.ndarray, tol: float = 1.0e-12) -> float:
    return 1.0 - np.count_nonzero(np.abs(K) > tol) / K.size


def matrix_to_string(A: np.ndarray, precision: int = 6) -> str:
    return np.array2string(A, precision=precision, suppress_small=True, floatmode="fixed")


def vector_to_numbered_lines(v: np.ndarray, prefix: str) -> str:
    return "\n".join(f"{prefix}{i + 1} = {value:.6f}" for i, value in enumerate(v))
