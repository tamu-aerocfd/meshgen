#!/usr/bin/env python3
"""Boilerplate generator for a multi-block backward-facing step UGRID file.

Reads a JSON input file, performs basic math to compute node coordinates with
geometric growth spacing in x and y, and writes an ASCII UGRID file containing
boundary quads and hexahedral volume elements.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MeshConfig:
    landing_length: float
    nx_landing: int
    dx_corner: float
    step_height: float
    upper_height: float
    ny_upper: int
    ny_step: int
    delta_z: float
    nz: int
    step_length: float
    nx_step: int


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
        "landing_length",
        "nx_landing",
        "dx_corner",
        "step_height",
        "upper_height",
        "ny_upper",
        "ny_step",
        "delta_z",
        "nz",
        "step_length",
        "nx_step",
    }
    missing = required.difference(data)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required keys: {missing_str}")

    return MeshConfig(
        landing_length=float(data["landing_length"]),
        nx_landing=int(data["nx_landing"]),
        dx_corner=float(data["dx_corner"]),
        step_height=float(data["step_height"]),
        upper_height=float(data["upper_height"]),
        ny_upper=int(data["ny_upper"]),
        ny_step=int(data["ny_step"]),
        delta_z=float(data["delta_z"]),
        nz=int(data["nz"]),
        step_length=float(data["step_length"]),
        nx_step=int(data["nx_step"]),
    )


def geometric_series_length(dx0: float, ratio: float, count: int) -> float:
    if count <= 0:
        return 0.0
    if ratio == 1.0:
        return dx0 * count
    return dx0 * (1.0 - ratio**count) / (1.0 - ratio)


def solve_growth_ratio(dx0: float, length: float, count: int) -> float:
    if count <= 0:
        raise ValueError("Cell count must be positive.")
    if length <= 0.0:
        raise ValueError("Length must be positive.")
    if count == 1:
        return 1.0

    min_length = dx0
    max_length = geometric_series_length(dx0, 1.3, count)
    if not (min_length <= length <= max_length):
        raise ValueError(
            f"Requested length {length} is out of bounds for dx0={dx0} and count={count}."
        )

    low, high = 1e-6, 1.3
    for _ in range(80):
        mid = 0.5 * (low + high)
        current = geometric_series_length(dx0, mid, count)
        if current < length:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def report_growth_ratio(label: str, ratio: float) -> None:
    print(f"Growth ratio for {label}: {ratio:.6f}")
    if ratio < 1.0 or ratio > 1.3:
        print(
            f"WARNING: growth ratio for {label} ({ratio:.6f}) is outside [1.0, 1.3]."
        )


def build_spacing(dx0: float, length: float, count: int, label: str) -> List[float]:
    ratio = solve_growth_ratio(dx0, length, count)
    report_growth_ratio(label, ratio)
    return [dx0 * ratio**i for i in range(count)]


def cumulative_coords(start: float, spacings: List[float]) -> List[float]:
    coords = [start]
    for delta in spacings:
        coords.append(coords[-1] + delta)
    return coords


def symmetric_spacing(dx0: float, length: float, count: int, label: str) -> List[float]:
    if count % 2 != 0:
        raise ValueError(f"{label} count must be even for symmetric spacing.")
    half_count = count // 2
    half_length = length / 2.0
    half_spacing = build_spacing(dx0, half_length, half_count, label)
    return half_spacing + list(reversed(half_spacing))


def wall_to_center_coords(
    height: float, count: int, dx0: float, label: str, wall_at_positive: bool
) -> List[float]:
    spacing = build_spacing(dx0, height, count, label)
    coords_from_wall = cumulative_coords(0.0, spacing)
    if wall_at_positive:
        coords = [height - s for s in coords_from_wall]
        return list(reversed(coords))
    return [-height + s for s in coords_from_wall]


def build_blocks(config: MeshConfig) -> UGrid:

    x_step_coords = wall_to_center_coords(
        config.step_length,
        config.nx_step,
        config.dx_corner,
        "step x (wall to outlet)",
        wall_at_positive=False,
    )
    x_step_coords=[config.step_length+i for i in x_step_coords]
    #print(x_step_coords)
    x_landing_spacing = build_spacing(
        config.dx_corner, config.landing_length, config.nx_landing, "landing x"
    )
    x_landing_coords = [0.0]
    for delta in x_landing_spacing:
        x_landing_coords.append(x_landing_coords[-1] - delta)
    x_landing_coords.reverse()
    #print(x_landing_coords)
    y_upper = wall_to_center_coords(
        config.upper_height,
        config.ny_upper,
        config.dx_corner,
        "upper y (wall to center)",
        wall_at_positive=True,
    )
    y_upper = [config.upper_height-i for i in y_upper]
    
    

    y_step_spacing = symmetric_spacing(config.dx_corner, config.step_height, config.ny_step, "step y (symmetric)")
    y_lower = cumulative_coords(0.0,y_step_spacing)
    y_lower = [-i for i in y_lower]
    #print(y_lower)
    #print(y_upper)
    z_spacing = [config.delta_z for _ in range(config.nz)]
    z_coords = cumulative_coords(0.0, z_spacing)

    nodes: List[Tuple[float, float, float]] = []
    node_index: Dict[Tuple[float, float, float], int] = {}
    cells: List[Tuple[int, int, int, int, int, int, int, int]] = []

    def register_node(coord: Tuple[float, float, float]) -> int:
        if coord in node_index:
            return node_index[coord]
        node_index[coord] = len(nodes)
        nodes.append(coord)
        return node_index[coord]

    def add_block(x_coords: List[float], y_coords: List[float], z_coords: List[float]) -> None:
        nx = len(x_coords) - 1
        ny = len(y_coords) - 1
        nz = len(z_coords) - 1
        block_nodes: Dict[Tuple[int, int, int], int] = {}

        for k, z in enumerate(z_coords):
            for j, y in enumerate(y_coords):
                for i, x in enumerate(x_coords):
                    idx = register_node((x, y, z))
                    block_nodes[(i, j, k)] = idx

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n0 = block_nodes[(i, j, k)]
                    n1 = block_nodes[(i + 1, j, k)]
                    n2 = block_nodes[(i + 1, j + 1, k)]
                    n3 = block_nodes[(i, j + 1, k)]
                    n4 = block_nodes[(i, j, k + 1)]
                    n5 = block_nodes[(i + 1, j, k + 1)]
                    n6 = block_nodes[(i + 1, j + 1, k + 1)]
                    n7 = block_nodes[(i, j + 1, k + 1)]
                    cells.append((n0, n1, n2, n3, n4, n5, n6, n7))

    add_block(x_landing_coords, y_upper, z_coords)
    add_block(x_step_coords, y_upper, z_coords)
    add_block(x_step_coords, y_lower, z_coords)

    boundary_quads, boundary_ids = extract_boundary_quads(nodes, cells)
    return UGrid(
        nodes=nodes,
        cells=cells,
        boundary_quads=boundary_quads,
        boundary_ids=boundary_ids,
    )


def extract_boundary_quads(
    nodes: List[Tuple[float, float, float]],
    cells: List[Tuple[int, int, int, int, int, int, int, int]]
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}
    face_counts: Dict[Tuple[int, int, int, int], int] = {}

    face_indices = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    for cell in cells:
        for indices in face_indices:
            face = tuple(cell[i] for i in indices)
            face_key = tuple(sorted(face))
            face_map[face_key] = face
            face_counts[face_key] = face_counts.get(face_key, 0) + 1

    min_x = min(coord[0] for coord in nodes)
    max_x = max(coord[0] for coord in nodes)
    min_y = min(coord[1] for coord in nodes)
    max_y = max(coord[1] for coord in nodes)
    min_z = min(coord[2] for coord in nodes)
    max_z = max(coord[2] for coord in nodes)

    tolerance = 1e-9

    def classify_face(face: Tuple[int, int, int, int]) -> int:
        xs = [nodes[idx][0] for idx in face]
        ys = [nodes[idx][1] for idx in face]
        zs = [nodes[idx][2] for idx in face]
        if all(abs(x - min_x) < tolerance for x in xs):
            return 1  # inlet
        if all(abs(x - max_x) < tolerance for x in xs):
            return 2  # outlet
        if all(abs(y - min_y) < tolerance for y in ys):
            return 3  # bottom wall
        if all(abs(y - max_y) < tolerance for y in ys):
            return 4  # top far field
        if all(abs(z - min_z) < tolerance for z in zs) or all(
            abs(z - max_z) < tolerance for z in zs
        ):
            return 5  # symmetry sides
        return 3

    boundary_quads: List[Tuple[int, int, int, int]] = []
    boundary_ids: List[int] = []
    for face_key, count in face_counts.items():
        if count == 1:
            face = face_map[face_key]
            boundary_quads.append(face)
            boundary_ids.append(classify_face(face))

    return boundary_quads, boundary_ids


def iter_ugrid_lines(ugrid: UGrid) -> Iterable[str]:
    number_of_nodes = len(ugrid.nodes)
    number_of_surf_trias = 0
    number_of_surf_quads = len(ugrid.boundary_quads)
    number_of_vol_tets = 0
    number_of_vol_pents_5 = 0
    number_of_vol_pents_6 = 0
    number_of_vol_hexs = len(ugrid.cells)

    yield (
        f"{number_of_nodes} {number_of_surf_trias} {number_of_surf_quads} "
        f"{number_of_vol_tets} {number_of_vol_pents_5} {number_of_vol_pents_6} "
        f"{number_of_vol_hexs}\n"
    )
    for x, y, z in ugrid.nodes:
        yield f"{x:.6f} {y:.6f} {z:.6f}\n"
    for quad in ugrid.boundary_quads:
        yield " ".join(str(idx + 1) for idx in quad) + "\n"
    for surface_id in ugrid.boundary_ids:
        yield f"{surface_id}\n"
    for cell in ugrid.cells:
        yield " ".join(str(idx + 1) for idx in cell) + "\n"


def write_ugrid(path: Path, ugrid: UGrid) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(iter_ugrid_lines(ugrid))


def write_mapbc(path: Path, boundary_ids: List[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for surface_id in boundary_ids:
            handle.write(f"{surface_id}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-block backward-facing step UGRID file."
    )
    parser.add_argument("input", type=Path, help="Path to JSON mesh configuration.")
    parser.add_argument("output", type=Path, help="Output UGRID file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.input)
    ugrid = build_blocks(config)
    write_ugrid(args.output, ugrid)
    mapbc_path = args.output.with_suffix(args.output.suffix + ".mapbc")
    write_mapbc(mapbc_path, ugrid.boundary_ids)


if __name__ == "__main__":
    main()
