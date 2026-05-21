#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三维杆单元 / 空间桁架单元刚度矩阵、应力计算与整体桁架求解程序

功能：
1. 根据两节点坐标计算三维杆单元长度 L 和方向余弦 (cx, cy, cz)
2. 建立全局坐标系下 6×6 单元刚度矩阵 Ke
3. 根据单元节点位移 de 计算轴向应变 epsilon、应力 sigma 和轴力 N
4. 自动完成 PDF 作业要求中的两个验证算例
5. 检查刚度矩阵对称性、半正定性、奇异性和刚体平移位移特征
6. 验证刚度矩阵第 j 列的物理意义
7. 附加题：多个三维杆单元组装整体刚度矩阵，并求解简单空间桁架结构

单位建议统一采用：N, m, Pa
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# 基础工具函数
# =============================================================================

def _as_vector(name: str, value, size: int) -> np.ndarray:
    """
    将输入转换为指定长度的一维浮点数组，并检查维度。
    """
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} 必须包含 {size} 个数值，目前输入长度为 {array.size}。")
    return array


def _check_positive(name: str, value: float) -> float:
    """
    检查材料参数或截面参数是否为正。
    """
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，目前输入为 {value}。")
    return value


def node_dof(node_id: int, direction: str | int) -> int:
    """
    返回某节点某方向对应的整体自由度编号。

    节点编号采用 Python 习惯：从 0 开始。
    direction 可输入：'x'/'u'/0，'y'/'v'/1，'z'/'w'/2。

    例如：
    node_dof(3, 'z') 表示第 4 个节点的 z 向自由度。
    """
    if isinstance(direction, str):
        direction = direction.lower()
        direction_map = {
            "x": 0, "u": 0,
            "y": 1, "v": 1,
            "z": 2, "w": 2,
        }
        if direction not in direction_map:
            raise ValueError("direction 只能取 'x'/'u', 'y'/'v', 'z'/'w' 或 0/1/2。")
        direction_id = direction_map[direction]
    else:
        direction_id = int(direction)
        if direction_id not in (0, 1, 2):
            raise ValueError("direction 编号只能为 0、1、2。")

    return 3 * int(node_id) + direction_id


def element_dofs(i: int, j: int) -> list[int]:
    """
    返回三维杆单元两个节点对应的 6 个整体自由度编号。
    """
    return [
        node_dof(i, 0), node_dof(i, 1), node_dof(i, 2),
        node_dof(j, 0), node_dof(j, 1), node_dof(j, 2),
    ]


# =============================================================================
# 三维杆单元刚度、应力、内力计算
# =============================================================================

def truss3d_element_stiffness(x1, x2, E, A, tol: float = 1.0e-12):
    """
    计算三维杆单元在全局坐标系下的单元刚度矩阵。

    Parameters
    ----------
    x1, x2 : array_like
        两个节点坐标，例如 [x, y, z]。
    E : float
        弹性模量，单位 Pa。
    A : float
        截面积，单位 m^2。
    tol : float
        判断退化单元的长度容差。

    Returns
    -------
    L : float
        单元长度。
    direction_cosines : ndarray, shape (3,)
        方向余弦 [cx, cy, cz]。
    Ke : ndarray, shape (6, 6)
        三维杆单元全局刚度矩阵。
    """
    x1 = _as_vector("x1", x1, 3)
    x2 = _as_vector("x2", x2, 3)
    E = _check_positive("E", E)
    A = _check_positive("A", A)

    dx = x2 - x1
    L = float(np.linalg.norm(dx))

    if L <= tol:
        raise ValueError("退化单元错误：两个节点重合或距离过小，不能继续计算刚度矩阵。")

    c = dx / L

    # 方向余弦外积矩阵：C = [cx, cy, cz]^T [cx, cy, cz]
    C = np.outer(c, c)

    # 三维杆单元全局刚度矩阵：
    # Ke = EA/L * [[ C, -C],
    #              [-C,  C]]
    Ke = (E * A / L) * np.block([[C, -C],
                                 [-C, C]])

    return L, c, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据单元节点位移计算三维杆单元轴向应变、应力和轴力。

    Parameters
    ----------
    x1, x2 : array_like
        两个节点坐标，例如 [x, y, z]。
    E : float
        弹性模量，单位 Pa。
    A : float
        截面积，单位 m^2。
    de : array_like
        单元节点位移列阵 [u1, v1, w1, u2, v2, w2]，单位 m。

    Returns
    -------
    epsilon : float
        轴向应变。
    sigma : float
        轴向应力，单位 Pa。
    N : float
        轴力，单位 N。拉力为正，压力为负。
    """
    de = _as_vector("de", de, 6)
    L, c, _ = truss3d_element_stiffness(x1, x2, E, A)

    # 三维杆单元应变-位移矩阵：epsilon = B de
    # B = 1/L * [-cx, -cy, -cz, cx, cy, cz]
    B = np.r_[-c, c] / L

    epsilon = float(B @ de)
    sigma = float(E * epsilon)
    N = float(sigma * A)

    return epsilon, sigma, N


def element_internal_force(x1, x2, E, A, de) -> np.ndarray:
    """
    计算单元节点内力列阵 Fe = Ke de。
    """
    de = _as_vector("de", de, 6)
    _, _, Ke = truss3d_element_stiffness(x1, x2, E, A)
    return Ke @ de


# =============================================================================
# 单元刚度矩阵性质检查
# =============================================================================

def check_stiffness_properties(Ke: np.ndarray, tol: float = 1.0e-7) -> dict:
    """
    检查刚度矩阵的基本性质：对称性、特征值非负性、奇异性和秩。
    """
    Ke = np.asarray(Ke, dtype=float)
    if Ke.shape != (6, 6):
        raise ValueError("Ke 必须是 6×6 矩阵。")

    eigenvalues = np.linalg.eigvalsh(Ke)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))

    return {
        "is_symmetric": bool(np.allclose(Ke, Ke.T, atol=tol, rtol=1.0e-10)),
        "eigenvalues": eigenvalues,
        "is_positive_semidefinite": bool(np.min(eigenvalues) >= -tol * scale),
        "rank": int(np.linalg.matrix_rank(Ke, tol=tol * scale)),
        "is_singular": bool(np.linalg.matrix_rank(Ke, tol=tol * scale) < 6),
    }


def rigid_translation_check(Ke: np.ndarray, translation=(1.0, 1.0, 1.0),
                            tol: float = 1.0e-7) -> dict:
    """
    检查刚体平移是否产生内力。

    对自由杆单元，若两个节点发生完全相同的平移，
    de = [tx, ty, tz, tx, ty, tz]^T，
    则杆长不变，应变为 0，理论上 Fe = Ke de = 0。
    """
    t = _as_vector("translation", translation, 3)
    de = np.r_[t, t]
    Fe = Ke @ de
    return {
        "rigid_de": de,
        "internal_force": Fe,
        "is_zero_force": bool(np.allclose(Fe, np.zeros(6), atol=tol, rtol=0.0)),
    }


def column_physical_meaning_check(Ke: np.ndarray, dof_index: int = 0) -> dict:
    """
    验证刚度矩阵第 j 列的物理意义。

    令第 j 个自由度位移为 1，其他自由度位移为 0，
    则 Fe = Ke de 正好等于 Ke 的第 j 列。
    这说明 kij 表示第 j 个自由度发生单位位移时，
    第 i 个自由度方向上产生的节点力。
    """
    if not 0 <= dof_index < 6:
        raise ValueError("dof_index 必须在 0 到 5 之间。")

    unit_de = np.zeros(6)
    unit_de[dof_index] = 1.0
    Fe = Ke @ unit_de

    return {
        "dof_index": dof_index,
        "unit_de": unit_de,
        "force_vector": Fe,
        "stiffness_column": Ke[:, dof_index],
        "same_as_column": bool(np.allclose(Fe, Ke[:, dof_index])),
    }


# =============================================================================
# 附加题：空间桁架整体刚度矩阵组装与求解
# =============================================================================

def assemble_global_stiffness(nodes, elements) -> np.ndarray:
    """
    将多个三维杆单元组装成整体刚度矩阵 K。

    Parameters
    ----------
    nodes : array_like, shape (n_node, 3)
        所有节点坐标。节点编号从 0 开始。
    elements : list
        杆件列表，每个元素格式为 [i, j, E, A]。
        i, j 为单元两端节点编号；E 为弹性模量；A 为截面积。

    Returns
    -------
    K : ndarray, shape (3*n_node, 3*n_node)
        空间桁架整体刚度矩阵。
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes 必须是形如 (节点数, 3) 的二维数组。")

    n_node = nodes.shape[0]
    total_dof = 3 * n_node
    K = np.zeros((total_dof, total_dof), dtype=float)

    for elem_id, elem in enumerate(elements):
        if len(elem) != 4:
            raise ValueError("elements 中每根杆件必须写成 [i, j, E, A]。")

        i, j, E, A = elem
        i = int(i)
        j = int(j)

        if not (0 <= i < n_node and 0 <= j < n_node):
            raise ValueError(f"第 {elem_id + 1} 根杆件节点编号超出范围。")
        if i == j:
            raise ValueError(f"第 {elem_id + 1} 根杆件两端节点相同，属于退化单元。")

        _, _, Ke = truss3d_element_stiffness(nodes[i], nodes[j], E, A)
        dofs = element_dofs(i, j)

        # 按照自由度对应关系，将单元刚度矩阵 Ke 叠加到整体刚度矩阵 K 中
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += Ke[a, b]

    return K


def solve_truss3d(nodes, elements, loads, fixed_dofs) -> dict:
    """
    求解空间桁架整体节点位移、支反力和各杆件应力。

    Parameters
    ----------
    nodes : array_like, shape (n_node, 3)
        节点坐标。
    elements : list
        杆件列表，每根杆件格式为 [i, j, E, A]。
    loads : array_like
        整体节点外力列阵。
        可以输入长度为 3*n_node 的一维数组；
        也可以输入形如 (n_node, 3) 的二维数组。
    fixed_dofs : list[int]
        约束自由度编号，例如 node_dof(0, 'x')。

    Returns
    -------
    results : dict
        包含整体刚度矩阵、节点位移、支反力和单元应力结果。
    """
    nodes = np.asarray(nodes, dtype=float)
    n_node = nodes.shape[0]
    total_dof = 3 * n_node

    K = assemble_global_stiffness(nodes, elements)
    F = np.asarray(loads, dtype=float).reshape(-1)

    if F.size != total_dof:
        raise ValueError(f"loads 长度必须为 {total_dof}，目前为 {F.size}。")

    fixed_dofs = sorted(set(int(dof) for dof in fixed_dofs))
    for dof in fixed_dofs:
        if dof < 0 or dof >= total_dof:
            raise ValueError(f"约束自由度 {dof} 超出范围。")

    all_dofs = np.arange(total_dof)
    free_dofs = np.array([dof for dof in all_dofs if dof not in fixed_dofs], dtype=int)
    fixed_dofs_array = np.array(fixed_dofs, dtype=int)

    if free_dofs.size == 0:
        raise ValueError("所有自由度都被约束，无法求解未知位移。")

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F[free_dofs]

    if np.linalg.matrix_rank(Kff) < Kff.shape[0]:
        raise ValueError("约束不足或结构存在机构：自由自由度刚度矩阵 Kff 奇异，无法求解。")

    Uf = np.linalg.solve(Kff, Ff)

    U = np.zeros(total_dof, dtype=float)
    U[free_dofs] = Uf

    # 支反力：R = K U - F
    reactions = K @ U - F

    # 回代计算每根杆件的应变、应力和轴力
    element_results = []
    for elem_id, elem in enumerate(elements):
        i, j, E, A = elem
        i = int(i)
        j = int(j)
        dofs = element_dofs(i, j)
        de = U[dofs]
        L, c, _ = truss3d_element_stiffness(nodes[i], nodes[j], E, A)
        epsilon, sigma, N = truss3d_element_stress(nodes[i], nodes[j], E, A, de)

        element_results.append({
            "element_id": elem_id,
            "node_i": i,
            "node_j": j,
            "L": L,
            "direction_cosines": c,
            "epsilon": epsilon,
            "sigma": sigma,
            "N": N,
            "de": de,
        })

    return {
        "K": K,
        "F": F,
        "U": U,
        "reactions": reactions,
        "free_dofs": free_dofs,
        "fixed_dofs": fixed_dofs_array,
        "element_results": element_results,
    }


# =============================================================================
# 输出辅助函数
# =============================================================================

def print_matrix(name: str, matrix: np.ndarray):
    """
    以科学计数法输出矩阵，便于复制到报告中。
    """
    print(f"{name} =")
    print(np.array2string(
        np.asarray(matrix, dtype=float),
        formatter={"float_kind": lambda x: f"{x: .4e}"},
        max_line_width=140
    ))


def print_nodal_vector(name: str, vector: np.ndarray):
    """
    按节点形式输出整体位移或外力。
    """
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size % 3 != 0:
        raise ValueError("节点向量长度必须是 3 的倍数。")

    print(name)
    for node in range(vector.size // 3):
        ux, uy, uz = vector[3 * node: 3 * node + 3]
        print(f"  节点 {node}: [{ux: .6e}, {uy: .6e}, {uz: .6e}]")


# =============================================================================
# 任务 2、任务 3、任务 4：单个三维杆单元验证
# =============================================================================

def run_case(case_name: str, x1, x2, E, A, de):
    """
    运行单个验证算例并输出计算结果。
    """
    print("\n" + "=" * 80)
    print(case_name)
    print("=" * 80)

    L, c, Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    Fe = element_internal_force(x1, x2, E, A, de)

    print(f"节点 1 坐标 x1 = {np.asarray(x1, dtype=float)}")
    print(f"节点 2 坐标 x2 = {np.asarray(x2, dtype=float)}")
    print(f"E = {E:.4e} Pa")
    print(f"A = {A:.4e} m^2")
    print(f"L = {L:.6g} m")
    print(f"方向余弦 (cx, cy, cz) = ({c[0]:.6g}, {c[1]:.6g}, {c[2]:.6g})")
    print_matrix("Ke / N·m^-1", Ke)

    print(f"de = {np.asarray(de, dtype=float)} m")
    print(f"epsilon = {epsilon:.6e}")
    print(f"sigma   = {sigma:.6e} Pa = {sigma / 1.0e6:.6g} MPa")
    print(f"N       = {N:.6e} N")
    print_matrix("Fe = Ke de / N", Fe)

    prop = check_stiffness_properties(Ke)
    print("\n刚度矩阵性质检查：")
    print(f"是否对称：{prop['is_symmetric']}")
    print(f"是否半正定：{prop['is_positive_semidefinite']}")
    print(f"矩阵秩 rank(Ke)：{prop['rank']}")
    print(f"是否奇异：{prop['is_singular']}")
    print("特征值：")
    print(np.array2string(prop["eigenvalues"],
                          formatter={"float_kind": lambda x: f"{x: .4e}"}))

    rigid = rigid_translation_check(Ke, translation=(1.0, 1.0, 1.0))
    print("\n刚体平移检查：de = [1, 1, 1, 1, 1, 1]^T")
    print_matrix("Ke de / N", rigid["internal_force"])
    print(f"是否近似为零内力：{rigid['is_zero_force']}")

    column_check = column_physical_meaning_check(Ke, dof_index=0)
    print("\n刚度矩阵物理意义验证：取第 1 个自由度 u1 = 1，其余自由度为 0")
    print_matrix("Fe = Ke de / N", column_check["force_vector"])
    print_matrix("Ke 第 1 列", column_check["stiffness_column"])
    print(f"Fe 是否等于 Ke 第 1 列：{column_check['same_as_column']}")

    return {
        "L": L,
        "direction_cosines": c,
        "Ke": Ke,
        "epsilon": epsilon,
        "sigma": sigma,
        "N": N,
        "Fe": Fe,
        "properties": prop,
    }


def run_degenerate_element_test():
    """
    退化单元检查：两个节点重合时，程序应报错而不是继续计算。
    """
    print("\n" + "=" * 80)
    print("退化单元检查")
    print("=" * 80)
    try:
        truss3d_element_stiffness([0, 0, 0], [0, 0, 0], 200e9, 1.0e-4)
    except ValueError as error:
        print(f"成功捕获错误：{error}")


# =============================================================================
# 附加题验证：简单空间桁架结构
# =============================================================================

def run_extra_truss_example():
    """
    附加题示例：求解一个 4 节点、3 杆件的简单空间桁架。

    结构说明：
    - 节点 0、1、2 固定；
    - 节点 3 为受力节点；
    - 三根杆分别连接 0-3、1-3、2-3；
    - 在节点 3 施加三向集中力；
    - 程序求解节点 3 位移、支反力以及每根杆件的应力和轴力。
    """
    print("\n" + "=" * 80)
    print("附加题：简单空间桁架整体刚度矩阵组装与求解")
    print("=" * 80)

    E = 210e9       # Pa
    A = 1.0e-4      # m^2

    # 节点坐标，节点编号从 0 开始
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # 杆件信息：[节点 i, 节点 j, E, A]
    elements = [
        [0, 3, E, A],
        [1, 3, E, A],
        [2, 3, E, A],
    ]

    # 整体外力列阵，长度为 3 × 节点数
    # 这里在节点 3 施加 Fx=1000 N, Fy=500 N, Fz=-2000 N
    loads = np.zeros(3 * len(nodes))
    loads[node_dof(3, "x")] = 1000.0
    loads[node_dof(3, "y")] = 500.0
    loads[node_dof(3, "z")] = -2000.0

    # 约束：节点 0、1、2 的 x、y、z 三个方向位移均为 0
    fixed_dofs = []
    for node in [0, 1, 2]:
        fixed_dofs += [node_dof(node, "x"), node_dof(node, "y"), node_dof(node, "z")]

    results = solve_truss3d(nodes, elements, loads, fixed_dofs)

    print("节点坐标：")
    for i, coord in enumerate(nodes):
        print(f"  节点 {i}: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) m")

    print("\n杆件连接关系：")
    for elem_id, elem in enumerate(elements):
        print(f"  杆件 {elem_id}: 节点 {int(elem[0])} - 节点 {int(elem[1])}, E={elem[2]:.3e} Pa, A={elem[3]:.3e} m^2")

    print("\n约束自由度 fixed_dofs =", fixed_dofs)
    print_nodal_vector("\n外力 F / N：", results["F"])
    print_matrix("\n整体刚度矩阵 K / N·m^-1", results["K"])
    print_nodal_vector("\n求解得到的节点位移 U / m：", results["U"])
    print_nodal_vector("\n支反力 R = K U - F / N：", results["reactions"])

    print("\n杆件应变、应力和轴力结果：")
    for item in results["element_results"]:
        tension_state = "受拉" if item["N"] >= 0 else "受压"
        print(
            f"  杆件 {item['element_id']} "
            f"({item['node_i']} - {item['node_j']}): "
            f"L = {item['L']:.6e} m, "
            f"epsilon = {item['epsilon']:.6e}, "
            f"sigma = {item['sigma'] / 1.0e6:.6e} MPa, "
            f"N = {item['N']:.6e} N, "
            f"{tension_state}"
        )

    print("\n附加题说明：")
    print("  1. 整体刚度矩阵 K 由各单元 Ke 按自由度编号叠加得到。")
    print("  2. 约束自由度位移已知为 0，只对自由自由度建立 Kff Uf = Ff。")
    print("  3. 求得整体位移 U 后，再提取每根杆件的单元位移 de，回代计算应变、应力和轴力。")

    return results


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    PDF 作业要求中的两个验证算例 + 附加题整体桁架算例。
    """
    np.set_printoptions(precision=6, suppress=False)

    # 算例 1：沿 x 轴的一维杆单元
    # 期望：L=2 m, c=(1,0,0), epsilon=5e-4, sigma=100 MPa, N=1.0e4 N
    case1 = run_case(
        case_name="算例 1：沿 x 轴的一维杆单元",
        x1=[0, 0, 0],
        x2=[2, 0, 0],
        E=200e9,
        A=1.0e-4,
        de=[0, 0, 0, 1.0e-3, 0, 0],
    )

    assert np.isclose(case1["L"], 2.0)
    assert np.allclose(case1["direction_cosines"], [1.0, 0.0, 0.0])
    assert np.isclose(case1["epsilon"], 5.0e-4)
    assert np.isclose(case1["sigma"], 100e6)
    assert np.isclose(case1["N"], 1.0e4)

    # 算例 2：空间任意方向杆单元
    # 期望：L=3 m, c=(1/3,2/3,2/3), epsilon=1e-3, sigma=210 MPa, N=4.2e4 N
    case2 = run_case(
        case_name="算例 2：空间任意方向杆单元",
        x1=[0, 0, 0],
        x2=[1, 2, 2],
        E=210e9,
        A=2.0e-4,
        de=[0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3],
    )

    assert np.isclose(case2["L"], 3.0)
    assert np.allclose(case2["direction_cosines"], [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    assert np.isclose(case2["epsilon"], 1.0e-3)
    assert np.isclose(case2["sigma"], 210e6)
    assert np.isclose(case2["N"], 4.2e4)
    assert case2["properties"]["is_symmetric"]
    assert case2["properties"]["is_positive_semidefinite"]
    assert case2["properties"]["is_singular"]

    run_degenerate_element_test()

    # 附加题：整体空间桁架结构求解
    run_extra_truss_example()

    print("\n" + "=" * 80)
    print("全部验证算例与附加题算例运行完成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
