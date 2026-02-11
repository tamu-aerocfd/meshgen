#!/usr/bin/env python3
"""Convert an ASCII UGRID volume mesh to Tecplot ASCII format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class UGridVolume:
    nodes: List[Tuple[float, float, float]]
    hexes: List[Tuple[int, int, int, int, int, int, int, int]]


def _read_tokens(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().split()


def _take_int(tokens: List[str], idx: int) -> Tuple[int, int]:
    return int(tokens[idx]), idx + 1


def _take_float(tokens: List[str], idx: int) -> Tuple[float, int]:
    return float(tokens[idx]), idx + 1


def load_ugrid(path: Path) -> UGridVolume:
    tokens = _read_tokens(path)
    i = 0

    n_nodes, i = _take_int(tokens, i)
    n_trias, i = _take_int(tokens, i)
    n_quads, i = _take_int(tokens, i)
    n_tets, i = _take_int(tokens, i)
    n_pyr5, i = _take_int(tokens, i)
    n_pri6, i = _take_int(tokens, i)
    n_hex8, i = _take_int(tokens, i)

    if n_tets or n_pyr5 or n_pri6:
        raise ValueError(
            "Only all-hex UGRID files are supported for Tecplot conversion in this tool."
        )

    nodes: List[Tuple[float, float, float]] = []
    for _ in range(n_nodes):
        x, i = _take_float(tokens, i)
        y, i = _take_float(tokens, i)
        z, i = _take_float(tokens, i)
        nodes.append((x, y, z))

    i += 3 * n_trias
    i += 4 * n_quads
    i += n_trias + n_quads

    hexes: List[Tuple[int, int, int, int, int, int, int, int]] = []
    for _ in range(n_hex8):
        conn = []
        for _ in range(8):
            node_id, i = _take_int(tokens, i)
            conn.append(node_id)
        hexes.append(tuple(conn))

    return UGridVolume(nodes=nodes, hexes=hexes)


def write_tecplot(path: Path, mesh: UGridVolume) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write('TITLE = "UGRID Volume"\n')
        handle.write('VARIABLES = "X" "Y" "Z"\n')
        handle.write(
            f'ZONE T="Volume", N={len(mesh.nodes)}, E={len(mesh.hexes)}, '
            'DATAPACKING=POINT, ZONETYPE=FEBRICK\n'
        )

        for x, y, z in mesh.nodes:
            handle.write(f"{x:.12g} {y:.12g} {z:.12g}\n")

        for cell in mesh.hexes:
            handle.write(" ".join(str(nid) for nid in cell) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ASCII UGRID to Tecplot ASCII.")
    parser.add_argument("input", type=Path, help="Path to input ASCII .ugrid file")
    parser.add_argument("output", type=Path, help="Path to output Tecplot .dat file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = load_ugrid(args.input)
    write_tecplot(args.output, mesh)


if __name__ == "__main__":
    main()
