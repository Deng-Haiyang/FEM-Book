"""Finite element program for global stiffness assembly and bar/truss solution.

Usage:
    python main.py ../examples/example_1d_bar.json
    python main.py ../examples/example_2d_truss.json --output ../results/example_2d_output.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from model import load_model
from assembly import assemble_global_stiffness, assemble_force_vector
from solver import matrix_rank_info, solve_by_reduction
from postprocess import postprocess_elements, matrix_to_string, sparsity_ratio


def analyze(model_path: str | Path) -> str:
    model = load_model(model_path)
    K, LM, element_matrices = assemble_global_stiffness(model)
    f = assemble_force_vector(model)

    rank_K, singular_K, cond_K = matrix_rank_info(K)
    d, reactions, solve_info = solve_by_reduction(model, K, f)
    element_results = postprocess_elements(model, LM, d)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Title: {model.title}")
    lines.append(f"nsd={model.nsd}, ndof={model.ndof}, nnp={model.nnp}, nel={model.nel}, neq={model.neq}")
    lines.append("")

    lines.append("Location matrix LM (shown in 1-based dof numbers, rows=element dofs, columns=elements):")
    lines.append(matrix_to_string(LM + 1, precision=0))
    lines.append("")

    for i, Ke in enumerate(element_matrices, start=1):
        lines.append(f"Element stiffness matrix Ke[{i}]:")
        lines.append(matrix_to_string(Ke))
        lines.append("")

    lines.append("Global stiffness matrix K:")
    lines.append(matrix_to_string(K))
    lines.append("")
    lines.append(f"K is symmetric: {np.allclose(K, K.T)}")
    lines.append(f"Rank(K) before boundary conditions: {rank_K}/{K.shape[0]}")
    lines.append(f"K is singular before boundary conditions: {singular_K}")
    if cond_K is not None:
        lines.append(f"Condition number of K before boundary conditions: {cond_K:.6e}")
    lines.append(f"Sparsity ratio of K: {sparsity_ratio(K):.6f}")
    lines.append(f"Diagonal entries are non-negative: {np.all(np.diag(K) >= -1.0e-12)}")
    lines.append("")

    lines.append("Global force vector f:")
    lines.append(matrix_to_string(f.reshape(-1, 1)))
    lines.append("")
    lines.append("Unknown/free dofs F (1-based): " + np.array2string(solve_info["free_dof_0_based"] + 1))
    lines.append("Known/fixed dofs E (1-based): " + np.array2string(solve_info["fixed_dof_0_based"] + 1))
    lines.append("Reduced stiffness matrix K_FF:")
    lines.append(matrix_to_string(solve_info["K_FF"]))
    lines.append(f"Rank(K_FF) after boundary conditions: {solve_info['rank_K_FF']}/{solve_info['K_FF'].shape[0]}")
    lines.append(f"K_FF is singular after boundary conditions: {solve_info['singular_K_FF']}")
    lines.append(f"Condition number of K_FF: {solve_info['condition_K_FF']:.6e}")
    lines.append("")

    lines.append("Solved displacement vector d:")
    for i, value in enumerate(d, start=1):
        lines.append(f"d{i} = {value:.6f}")
    lines.append("")

    lines.append("Reaction vector r = K d - f:")
    for i, value in enumerate(reactions, start=1):
        lines.append(f"r{i} = {value:.6f}")
    lines.append("")
    lines.append("Reactions on fixed dofs only:")
    for dof in model.fixed_dof:
        lines.append(f"dof {dof + 1}: reaction = {reactions[dof]:.6f}")
    lines.append("")

    lines.append("Element post-processing:")
    for r in element_results:
        dir_text = ", ".join(f"{x:.6f}" for x in r.direction_cosines)
        de_text = ", ".join(f"{x:.6f}" for x in r.local_displacement)
        lines.append(
            f"Element {r.element_id} nodes {r.nodes}: L={r.length:.6f}, "
            f"direction=({dir_text}), de=[{de_text}], "
            f"stress={r.stress:.6f}, axial_force={r.axial_force:.6f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Global stiffness assembly and truss/bar solver")
    parser.add_argument("model", help="Path to a JSON model file")
    parser.add_argument("--output", "-o", help="Optional output text file")
    args = parser.parse_args()

    text = analyze(args.model)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
