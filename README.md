# Meshgen UGRID Scripts

This repository now provides two mesh generators that write ASCII UGRID files and
companion `.mapbc` files.

## 1) Backward-facing step (BFS)

Use `BFS_meshgen.py` with the existing BFS JSON input format.

```bash
python3 BFS_meshgen.py sample_input.json bfs_output.ugrid
```

## 2) Flat plate (FP)

Use `FP_meshgen.py` with parameters:

- `x_landing`
- `nx_landing`
- `x1`
- `x_plate`
- `nx_plate`
- `ny`
- `y1`
- `ly`
- `nz`
- `lz`

```bash
python3 FP_meshgen.py fp_sample_input.json fp_output.ugrid
```

### Flat-plate spacing behavior

- Landing block spans `[-x_landing, 0]` and uses `x1` at `x=0`, growing toward
  `-x_landing`.
- Plate block spans `[0, x_plate]` and uses `x1` at `x=0`, growing toward
  `x_plate`.
- Both blocks share the same y-grid, starting at `y1` and growing upward to `ly`.

### Flat-plate BC IDs

- `1`: inlet (`x = min_x`)
- `2`: outlet (`x = max_x`)
- `3`: inviscid wall on landing (`y = 0`, `x < 0`)
- `4`: far field (`y = max_y`)
- `5`: symmetry sides (`z = min_z` and `z = max_z`)
- `6`: viscous wall on plate (`y = 0`, `x >= 0`)
