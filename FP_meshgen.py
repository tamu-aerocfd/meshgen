#!/usr/bin/env python3
"""Generate a 3D two-block flat-plate mesh in ASCII UGRID format."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MeshConfig:
    x_landing: float
    nx_landing: int
    x1: float
    x_plate: float
    nx_plate: int
    ny: int
    y1: float
    ly: float
    nz: int
    lz: float


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
        "x_landing",
        "nx_landing",
        "x1",
        "x_plate",
        "nx_plate",
        "ny",
        "y1",
        "ly",
        "nz",
        "lz",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing))}")

    return MeshConfig(
        x_landing=float(data["x_landing"]),
        nx_landing=int(data["nx_landing"]),
        x1=float(data["x1"]),
        x_plate=float(data["x_plate"]),
        nx_plate=int(data["nx_plate"]),
        ny=int(data["ny"]),
        y1=float(data["y1"]),
        ly=float(data["ly"]),
        nz=int(data["nz"]),
        lz=float(data["lz"]),
    )


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
        current = geometric_series_length(d0, mid, count)
        if current < length:
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


def build_blocks(config: MeshConfig) -> UGrid:
    x_plate_spacing = build_spacing(config.x1, config.x_plate, config.nx_plate, "plate x")
    x_plate_coords = cumulative_coords(0.0, x_plate_spacing)

    x_landing_spacing = build_spacing(config.x1, config.x_landing, config.nx_landing, "landing x")
    x_landing_coords = [0.0]
    for dx in x_landing_spacing:
        x_landing_coords.append(x_landing_coords[-1] - dx)
    x_landing_coords.reverse()

    y_spacing = build_spacing(config.y1, config.ly, config.ny, "y")
    y_coords = cumulative_coords(0.0, y_spacing)

    dz = config.lz / config.nz
    z_coords = [k * dz for k in range(config.nz + 1)]

    nodes: List[Tuple[float, float, float]] = []
    node_index: Dict[Tuple[float, float, float], int] = {}
    cells: List[Tuple[int, int, int, int, int, int, int, int]] = []

    def register_node(coord: Tuple[float, float, float]) -> int:
        if coord in node_index:
            return node_index[coord]
        idx = len(nodes)
        nodes.append(coord)
        node_index[coord] = idx
        return idx

    def add_block(x_coords: List[float]) -> None:
        nx = len(x_coords) - 1
        ny = len(y_coords) - 1
        nz = len(z_coords) - 1
        local: Dict[Tuple[int, int, int], int] = {}

        for k, z in enumerate(z_coords):
            for j, y in enumerate(y_coords):
                for i, x in enumerate(x_coords):
                    local[(i, j, k)] = register_node((x, y, z))

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n0 = local[(i, j, k)]
                    n1 = local[(i + 1, j, k)]
                    n2 = local[(i + 1, j + 1, k)]
                    n3 = local[(i, j + 1, k)]
                    n4 = local[(i, j, k + 1)]
                    n5 = local[(i + 1, j, k + 1)]
                    n6 = local[(i + 1, j + 1, k + 1)]
                    n7 = local[(i, j + 1, k + 1)]
                    cells.append((n0, n1, n2, n3, n4, n5, n6, n7))

    add_block(x_landing_coords)
    add_block(x_plate_coords)

    boundary_quads, boundary_ids = extract_boundary_quads(nodes, cells)
    return UGrid(nodes=nodes, cells=cells, boundary_quads=boundary_quads, boundary_ids=boundary_ids)


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
    min_y = min(y for _, y, _ in nodes)
    max_y = max(y for _, y, _ in nodes)
    min_z = min(z for _, _, z in nodes)
    max_z = max(z for _, _, z in nodes)
    tol = 1e-9

    def classify(face: Tuple[int, int, int, int]) -> int:
        xs = [nodes[i][0] for i in face]
        ys = [nodes[i][1] for i in face]
        zs = [nodes[i][2] for i in face]
        x_mean = sum(xs) / 4.0
        if all(abs(x - min_x) < tol for x in xs):
            return 1  # inlet
        if all(abs(x - max_x) < tol for x in xs):
            return 2  # outlet
        if all(abs(y - min_y) < tol for y in ys):
            return 3 if x_mean < 0.0 else 6  # landing inviscid / plate viscous
        if all(abs(y - max_y) < tol for y in ys):
            return 4  # far field
        if all(abs(z - min_z) < tol for z in zs) or all(abs(z - max_z) < tol for z in zs):
            return 5  # symmetry
        return 0

    quads: List[Tuple[int, int, int, int]] = []
    ids: List[int] = []
    for key, count in counts.items():
        if count == 1:
            face = face_map[key]
            quads.append(face)
            ids.append(classify(face))

    return quads, ids


def iter_ugrid_lines(ugrid: UGrid) -> Iterable[str]:
    yield f"{len(ugrid.nodes)} 0 {len(ugrid.boundary_quads)} 0 0 0 {len(ugrid.cells)}\n"
    for x, y, z in ugrid.nodes:
        yield f"{x:.6f} {y:.6f} {z:.6f}\n"
    for quad in ugrid.boundary_quads:
        yield " ".join(str(i + 1) for i in quad) + "\n"
    for bc in ugrid.boundary_ids:
        yield f"{bc}\n"
    for cell in ugrid.cells:
        yield " ".join(str(i + 1) for i in cell) + "\n"


def write_ugrid(path: Path, ugrid: UGrid) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(iter_ugrid_lines(ugrid))


def write_mapbc(path: Path, boundary_ids: List[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for bc in boundary_ids:
            handle.write(f"{bc}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a flat-plate UGRID mesh.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.input)
    ugrid = build_blocks(config)
    write_ugrid(args.output, ugrid)
    write_mapbc(args.output.with_suffix(args.output.suffix + ".mapbc"), ugrid.boundary_ids)


if __name__ == "__main__":
    main()
