# Meshgen UGRID Boilerplate

This repository contains a Python script that reads a JSON mesh configuration,
performs basic coordinate math for a multi-block backward-facing step, and
writes a simple UGRID-like text file containing nodes and hex connectivity.

The mesh is constructed from three structured blocks:

- Inlet block (upper channel, upstream of the step)
- Upper block (upper channel, downstream of the step)
- Step block (lower channel, downstream of the step)

## Usage

```bash
python3 meshgen.py sample_input.json output.ugrid
```

## Input format

The input JSON file must include:

- `landing_length`: length of the inlet block upstream of the step
- `nx_landing`: number of cells in the inlet block (x direction)
- `dx_corner`: starting spacing at the step corner (x/y directions)
- `step_height`: height of the step (also used for the upper channel height)
- `ny_step`: number of cells across the step height
- `delta_z`: uniform spacing in the z direction
- `nz`: number of cells in the z direction
- `step_length`: length of the downstream blocks
- `nx_step`: number of cells in the downstream blocks (x direction)

Example:

```json
{
  "landing_length": 2.0,
  "nx_landing": 6,
  "dx_corner": 0.05,
  "step_height": 1.0,
  "ny_step": 10,
  "delta_z": 0.1,
  "nz": 5,
  "step_length": 4.0,
  "nx_step": 12
}
```

## Output format

The output is a simple text representation of a UGRID-like file with node
coordinates followed by hex cell connectivity. This is intended as a
boilerplate starting point for integrating with real UGRID/NetCDF workflows.
