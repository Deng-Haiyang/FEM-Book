"""Model input and degree-of-freedom utilities for truss/bar finite elements."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Model:
    title: str
    nsd: int
    ndof: int
    nnp: int
    nel: int
    nen: int
    E: np.ndarray
    area: np.ndarray
    coords: np.ndarray
    ien: np.ndarray
    fixed_dof: np.ndarray
    fixed_value: np.ndarray
    force_dof: np.ndarray
    force_value: np.ndarray

    @property
    def neq(self) -> int:
        return self.nnp * self.ndof


def load_model(path: str | Path) -> Model:
    """Load a JSON model file.

    Notes
    -----
    JSON node numbers and dof numbers are allowed to be 1-based, as required by
    the homework statement. They are converted to 0-based indices internally.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    nsd = int(data["nsd"])
    ndof = int(data.get("ndof", nsd))
    nnp = int(data["nnp"])
    nel = int(data["nel"])
    nen = int(data.get("nen", 2))

    if nen != 2:
        raise ValueError("This program supports two-node bar/truss elements only.")
    if nsd not in (1, 2, 3):
        raise ValueError("nsd must be 1, 2, or 3.")
    if ndof != nsd:
        raise ValueError("For axial bar/truss elements, ndof should equal nsd.")

    x = np.asarray(data["x"], dtype=float)
    if nsd == 1:
        coords = x.reshape(nnp, 1)
    elif nsd == 2:
        y = np.asarray(data["y"], dtype=float)
        coords = np.column_stack((x, y))
    else:
        y = np.asarray(data["y"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
        coords = np.column_stack((x, y, z))

    ien = np.asarray(data["IEN"], dtype=int) - 1
    if ien.shape != (nel, nen):
        raise ValueError(f"IEN should have shape ({nel}, {nen}), got {ien.shape}.")
    if ien.min() < 0 or ien.max() >= nnp:
        raise ValueError("IEN contains node numbers outside the valid range.")

    E = np.asarray(data["E"], dtype=float)
    area = np.asarray(data.get("CArea", data.get("A")), dtype=float)
    if len(E) != nel or len(area) != nel:
        raise ValueError("E and CArea/A must contain one value per element.")

    fixed_dof = np.asarray(data.get("fixed_dof", []), dtype=int) - 1
    fixed_value = np.asarray(data.get("fixed_value", []), dtype=float)
    force_dof = np.asarray(data.get("force_dof", []), dtype=int) - 1
    force_value = np.asarray(data.get("force_value", []), dtype=float)

    if len(fixed_dof) != len(fixed_value):
        raise ValueError("fixed_dof and fixed_value must have the same length.")
    if len(force_dof) != len(force_value):
        raise ValueError("force_dof and force_value must have the same length.")

    title = str(data.get("Title", path.stem))
    return Model(title, nsd, ndof, nnp, nel, nen, E, area, coords, ien,
                 fixed_dof, fixed_value, force_dof, force_value)


def global_dof(node_id_zero_based: int, local_dof: int, ndof: int) -> int:
    """Return the 0-based global dof number for a node and local dof."""
    return node_id_zero_based * ndof + local_dof


def build_lm(model: Model) -> np.ndarray:
    """Build the location matrix LM with shape (nen*ndof, nel)."""
    lm = np.zeros((model.nen * model.ndof, model.nel), dtype=int)
    for e, nodes in enumerate(model.ien):
        col: List[int] = []
        for node in nodes:
            for a in range(model.ndof):
                col.append(global_dof(int(node), a, model.ndof))
        lm[:, e] = col
    return lm
