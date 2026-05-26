"""Element-level stiffness, strain, stress and axial-force calculations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import Model


@dataclass
class ElementResult:
    element_id: int
    nodes: tuple[int, int]
    length: float
    direction_cosines: np.ndarray
    stress: float
    axial_force: float
    local_displacement: np.ndarray


def element_geometry(model: Model, e: int) -> tuple[float, np.ndarray]:
    """Return element length and direction cosines."""
    n1, n2 = model.ien[e]
    vec = model.coords[n2] - model.coords[n1]
    length = float(np.linalg.norm(vec))
    if length <= 0.0:
        raise ValueError(f"Element {e + 1} has zero length.")
    direction = vec / length
    return length, direction


def element_stiffness(model: Model, e: int) -> np.ndarray:
    """Compute global-coordinate stiffness matrix of a 1D/2D/3D truss element."""
    length, direction = element_geometry(model, e)
    k_axial = model.E[e] * model.area[e] / length

    if model.nsd == 1:
        # One-dimensional axial bar element.
        return k_axial * np.array([[1.0, -1.0], [-1.0, 1.0]])

    # General truss form in global coordinates: k = EA/L * [[l l^T, -l l^T], [-l l^T, l l^T]]
    ll = np.outer(direction, direction)
    return k_axial * np.block([[ll, -ll], [-ll, ll]])


def element_stress_force(model: Model, e: int, d_e: np.ndarray) -> ElementResult:
    """Compute element stress and axial force using the global displacement vector of one element."""
    length, direction = element_geometry(model, e)
    B = np.concatenate((-direction, direction)) / length
    strain = float(B @ d_e)
    stress = model.E[e] * strain
    axial_force = stress * model.area[e]
    nodes_1_based = tuple((model.ien[e] + 1).tolist())
    return ElementResult(e + 1, nodes_1_based, length, direction, stress, axial_force, d_e.copy())
