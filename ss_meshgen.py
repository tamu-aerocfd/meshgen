#!/usr/bin/env python3
"""Generate a smooth-seal mesh as ASCII UGRID with periodic theta connectivity."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MeshConfig:
    shaft_radius: float
    nx_landing: int
    x_landing: float
    x_seal: float
    nx_seal: int
    x1: float
    y_clearance: float
    ny_clearance: int
    r_stator: float
    ny_landing: int
    y1: float
    n_theta: int


@dataclass(frozen=True)
class UGrid:
    nodes: List[Tuple[float, float, float]]
    cells: List[Tuple[int, int, int, int, int, int, int, int]]
    boundary_quads: List[Tuple[int, int, int, int]]
    boundary_ids: List[int]


def load_config(path: Path) -> MeshConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    required = {
        "shaft_radius",
        "nx_landing",
        "x_landing",
        "x_seal",
        "nx_seal",
        "x1",
        "y_clearance",
        "ny_clearance",
        "r_stator",
        "ny_landing",
        "y1",
        "n_theta",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing))}")

    cfg = MeshConfig(
        shaft_radius=float(data["shaft_radius"]),
        nx_landing=int(data["nx_landing"]),
        x_landing=float(data["x_landing"]),
        x_seal=float(data["x_seal"]),
        nx_seal=int(data["nx_seal"]),
        x1=float(data["x1"]),
        y_clearance=float(data["y_clearance"]),
        ny_clearance=int(data["ny_clearance"]),
        r_stator=float(data["r_stator"]),
        ny_landing=int(data["ny_landing"]),
        y1=float(data["y1"]),
        n_theta=int(data["n_theta"]),
    )
    if cfg.ny_clearance % 2 != 0:
        raise ValueError("ny_clearance must be even for reflected spacing.")
    if cfg.ny_landing % 2 != 0:
        raise ValueError("ny_landing must be even for reflected spacing.")
    if cfg.nx_seal % 2 != 0:
        raise ValueError("nx_seal must be even for symmetric two-sided seal spacing.")
    if cfg.r_stator <= cfg.shaft_radius + cfg.y_clearance:
        raise ValueError("r_stator must be greater than shaft_radius + y_clearance.")
    if cfg.n_theta < 3:
        raise ValueError("n_theta must be at least 3.")
    return cfg


def geometric_series_length(d0: float, ratio: float, count: int) -> float:
    if count <= 0:
        return 0.0
    if ratio == 1.0:
        return d0 * count
    return d0 * (1.0 - ratio**count) / (1.0 - ratio)


def solve_growth_ratio(d0: float, length: float, count: int) -> float:
    if count <= 0:
        raise ValueError("Cell count must be positive.")
    if length <= 0.0:
        raise ValueError("Length must be positive.")
    if count == 1:
        return 1.0

    low, high = 1e-6, 10.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        value = geometric_series_length(d0, mid, count)
        if value < length:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def report_growth_ratio(label: str, ratio: float) -> None:
    print(f"Growth ratio for {label}: {ratio:.6f}")
    if ratio < 1.0 or ratio > 1.3:
        print(f"WARNING: growth ratio for {label} ({ratio:.6f}) is outside [1.0, 1.3].")


def build_spacing(d0: float, length: float, count: int, label: str) -> List[float]:
    ratio = solve_growth_ratio(d0, length, count)
    report_growth_ratio(label, ratio)
    return [d0 * ratio**i for i in range(count)]


def cumulative_coords(start: float, spacing: List[float]) -> List[float]:
    coords = [start]
    for d in spacing:
        coords.append(coords[-1] + d)
    return coords


def build_shrinking_spacing_toward_right(
    d0_right: float, length: float, count: int, label: str
) -> List[float]:
    grow_away_from_right = build_spacing(d0_right, length, count, label)
    return list(reversed(grow_away_from_right))


def build_symmetric_spacing_ends(
    d0_end: float, length: float, count: int, label: str
) -> List[float]:
    if count % 2 != 0:
        raise ValueError(f"{label} count must be even for symmetric two-sided spacing.")
    half = count // 2
    left_half = build_spacing(d0_end, length / 2.0, half, label)
    return left_half + list(reversed(left_half))


def build_reflected_radial_coords(
    r_min: float, r_max: float, count: int, y1: float, label: str
) -> List[float]:
    if count % 2 != 0:
        raise ValueError(f"{label} count must be even for reflected spacing.")
    half = count // 2
    half_spacing = build_spacing(y1, (r_max - r_min) / 2.0, half, label)
    spacing = half_spacing + list(reversed(half_spacing))
    return cumulative_coords(r_min, spacing)


def add_block_periodic_theta(
    nodes: List[Tuple[float, float, float]],
    node_index: Dict[Tuple[float, float, float], int],
    cells: List[Tuple[int, int, int, int, int, int, int, int]],
    x_coords: List[float],
    r_coords: List[float],
    n_theta: int,
) -> None:
    nx = len(x_coords) - 1
    ny = len(r_coords) - 1
    local: Dict[Tuple[int, int, int], int] = {}

    def register(coord: Tuple[float, float, float]) -> int:
        key = (round(coord[0], 12), round(coord[1], 12), round(coord[2], 12))
        if key in node_index:
            return node_index[key]
        idx = len(nodes)
        nodes.append(coord)
        node_index[key] = idx
        return idx

    for k in range(n_theta):
        theta = 2.0 * math.pi * k / n_theta
        ct = math.cos(theta)
        st = math.sin(theta)
        for j, r in enumerate(r_coords):
            for i, x in enumerate(x_coords):
                y = r * ct
                z = r * st
                local[(i, j, k)] = register((x, y, z))

    for k in range(n_theta):
        k2 = (k + 1) % n_theta
        for j in range(ny):
            for i in range(nx):
                n0 = local[(i, j, k)]
                n1 = local[(i + 1, j, k)]
                n2 = local[(i + 1, j + 1, k)]
                n3 = local[(i, j + 1, k)]
                n4 = local[(i, j, k2)]
                n5 = local[(i + 1, j, k2)]
                n6 = local[(i + 1, j + 1, k2)]
                n7 = local[(i, j + 1, k2)]
                cells.append((n0, n1, n2, n3, n4, n5, n6, n7))


def extract_boundary_quads(
    nodes: List[Tuple[float, float, float]],
    cells: List[Tuple[int, int, int, int, int, int, int, int]],
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    face_patterns = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for cell in cells:
        for p in face_patterns:
            face = tuple(cell[i] for i in p)
            key = tuple(sorted(face))
            face_map[key] = face
            counts[key] = counts.get(key, 0) + 1

    min_x = min(x for x, _, _ in nodes)
    max_x = max(x for x, _, _ in nodes)
    radii = [math.sqrt(y * y + z * z) for _, y, z in nodes]
    min_r = min(radii)
    max_r = max(radii)
    tol = 1e-8

    def classify(face: Tuple[int, int, int, int]) -> int:
        xs = [nodes[i][0] for i in face]
        rs = [math.sqrt(nodes[i][1] ** 2 + nodes[i][2] ** 2) for i in face]
        if all(abs(r - min_r) < tol for r in rs):
            return 1  # rotating wall (shaft)
        if all(abs(r - max_r) < tol for r in rs):
            return 2  # rotating wall (stator)
        if all(abs(x - min_x) < tol for x in xs):
            return 3  # inlet
        if all(abs(x - max_x) < tol for x in xs):
            return 4  # outlet
        return 0

    quads: List[Tuple[int, int, int, int]] = []
    ids: List[int] = []
    for key, count in counts.items():
        if count == 1:
            face = face_map[key]
            quads.append(face)
            ids.append(classify(face))
    return quads, ids


def build_mesh(config: MeshConfig) -> UGrid:
    r_shaft = config.shaft_radius
    r_clear_top = config.shaft_radius + config.y_clearance

    # Inlet landing: one-sided shrink toward x1 at right (near seal entrance)
    x_inlet_spacing = build_shrinking_spacing_toward_right(
        config.x1, config.x_landing, config.nx_landing, "inlet landing x (shrink to x1)"
    )
    x_inlet = cumulative_coords(0.0, x_inlet_spacing)

    # Seal: two-sided shrink to x1 at both ends, meeting at center
    x_seal_spacing = build_symmetric_spacing_ends(
        config.x1, config.x_seal, config.nx_seal, "seal x (symmetric ends)"
    )
    x_seal = cumulative_coords(config.x_landing, x_seal_spacing)

    # Outlet landing: one-sided growth from x1 at left (seal exit) to outlet
    x_outlet_spacing = build_spacing(
        config.x1, config.x_landing, config.nx_landing, "outlet landing x (grow from x1)"
    )
    x_outlet = cumulative_coords(config.x_landing + config.x_seal, x_outlet_spacing)

    # Clearance and outer landing both use reflected spacing with y1 at both sides.
    y_clear = build_reflected_radial_coords(
        r_shaft, r_clear_top, config.ny_clearance, config.y1, "clearance y (reflected)"
    )
    y_outer = build_reflected_radial_coords(
        r_clear_top, config.r_stator, config.ny_landing, config.y1, "outer landing y (reflected)"
    )

    nodes: List[Tuple[float, float, float]] = []
    node_index: Dict[Tuple[float, float, float], int] = {}
    cells: List[Tuple[int, int, int, int, int, int, int, int]] = []

    # 1) inlet clearance
    add_block_periodic_theta(nodes, node_index, cells, x_inlet, y_clear, config.n_theta)
    # 2) inlet outer landing
    add_block_periodic_theta(nodes, node_index, cells, x_inlet, y_outer, config.n_theta)
    # 3) seal clearance
    add_block_periodic_theta(nodes, node_index, cells, x_seal, y_clear, config.n_theta)
    # 4) outlet clearance
    add_block_periodic_theta(nodes, node_index, cells, x_outlet, y_clear, config.n_theta)
    # 5) outlet outer landing
    add_block_periodic_theta(nodes, node_index, cells, x_outlet, y_outer, config.n_theta)

    boundary_quads, boundary_ids = extract_boundary_quads(nodes, cells)
    return UGrid(nodes=nodes, cells=cells, boundary_quads=boundary_quads, boundary_ids=boundary_ids)


def iter_ugrid_lines(ugrid: UGrid) -> Iterable[str]:
    yield f"{len(ugrid.nodes)} 0 {len(ugrid.boundary_quads)} 0 0 0 {len(ugrid.cells)}\n"
    for x, y, z in ugrid.nodes:
        yield f"{x:.9f} {y:.9f} {z:.9f}\n"
    for quad in ugrid.boundary_quads:
        yield " ".join(str(i + 1) for i in quad) + "\n"
    for sid in ugrid.boundary_ids:
        yield f"{sid}\n"
    for cell in ugrid.cells:
        yield " ".join(str(i + 1) for i in cell) + "\n"


def write_ugrid(path: Path, ugrid: UGrid) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(iter_ugrid_lines(ugrid))


def write_mapbc(path: Path, boundary_ids: List[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sid in boundary_ids:
            handle.write(f"{sid}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate smooth-seal UGRID mesh.")
    parser.add_argument("input", type=Path, help="Path to smooth-seal JSON input")
    parser.add_argument("output", type=Path, help="Output UGRID path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.input)
    ugrid = build_mesh(config)
    write_ugrid(args.output, ugrid)
    write_mapbc(args.output.with_suffix(args.output.suffix + ".mapbc"), ugrid.boundary_ids)


if __name__ == "__main__":
    main()
