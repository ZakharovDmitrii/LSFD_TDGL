# TDGL LSFD Solver

[![PyPI version](https://badge.fury.io/py/tdgl_LSFD_lib.svg)](https://pypi.org/project/tdgl_LSFD_lib/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TDGL LSFD Solver** is a Python library for simulating the dynamics of superconducting thin films within the time-dependent Ginzburg–Landau (TDGL) framework. Spatial derivatives are computed with the **Least-Squares Finite Difference (LSFD)** method on unstructured triangular meshes, which enables arbitrary 2D geometries and high-order accuracy. Linear and cubic **spin–orbit coupling (SOC)** terms are built in, opening the door to modeling the **superconducting diode effect (SDE)** and other non-reciprocal superconducting phenomena."

<p align="center">
  <img src="docs/images/four_vortex.png" width="45%" alt="Device and triangular mesh">
  <img src="docs/images/shifted_vortex.png" width="45%" alt="Vortex pattern |psi|^2">
</p>
<p align="center"><i>Left: Vortices on radial film: |\psi|, phase, \mu, supercurrent, normal current, div(Js) Right: Supercurrent distribution in case gamma = 0.1 .</i></p>

## Features

- **Unstructured meshes** — Delaunay triangulation of arbitrary 2D geometries (MeshPy), including holes and current terminals; Laplacian and Lloyd (CVT) smoothing with grading toward the boundary.
- **High-order LSFD operators** — derivatives up to 4th order on irregular stencils, with gauge-invariant link variables $U_{ij}=\exp(-i\,\mathbf{A}_i\cdot\mathbf{e}_{ij})$. There is point valued scheme in opposite to dual structure of previous TDGL solvers. 
- **Accurate boundary conditions** — Neumann boundary conditions enforced via *ghost* or *mirror* points, keeping stencils symmetric at the boundary.
- **Ferromagnetic coupling** — spin–orbit coupling linear term with coefficient $\eta$ and cubic term with coefficient $\gamma$ with an arbitrary polarization direction $\mathbf{s}$.
- **Adaptive dynamics** — adaptive time stepping and adaptive tolerance for the Poisson solver for the scalar potential $\mu$.
- **Conservation checks** — a finite-volume (FVM) integrator to verify the divergence theorem and local/global current conservation.
- **HDF5 I/O** — checkpoints with the ability to resume a simulation.
- **Visualization** — devices, meshes, Voronoi diagrams, fields and animations.

## Installation

### From PyPI (recommended)

```bash
pip install tdgl_LSFD_lib
```

### From source (development mode)

```bash
git clone https://github.com/ZakharovDmitrii/LSFD_TDGL.git
cd LSFD_TDGL
pip install -e .
```

## Project structure

```
LSFD_TDGL/
├── tdgl_LSFD_lib/
│   ├── device/            # Device geometry, polygons, mesh generation
│   ├── mesh/              # Mesh, TriMesh, DualMesh (Voronoi), mirror points
│   ├── operators/         # LSFD operators, FVM integrator, numba kernels
│   ├── external_fields/   # External fields: B, J, ferromagnet (eta, gamma, s)
│   ├── solver/            # TDGL solver, runner, adaptive options
│   └── post_processing/   # HDF5 loading, analysis and animations
├── examples/              # Runnable examples
├── docs/images/           # Images for this README
├── pyproject.toml
├── LICENSE
└── README.md
```

## How it works

1. **Mesh** — a constrained Delaunay triangulation of the film is generated and smoothed; a Voronoi dual mesh provides control volumes for finite-volume checks.
2. **LSFD** — at every vertex, derivatives up to 4th order are obtained by a weighted least-squares fit over the neighbor stencil; gauge invariance is enforced through link variables.
3. **Boundary** — mirror (or ghost) points reflected across the boundary symmetrize boundary stencils and impose the condition $(\mathbf{n}, \mathbf{D}\psi) = -i\eta(\mathbf{n},\mathbf{s})\psi$.
4. **Dynamics** — a semi-implicit adaptive Euler scheme advances $\psi$; the scalar potential $\mu$ is obtained from an iterative Poisson solve with adaptive tolerance.

## Dependencies

Required: `numpy`, `scipy`, `h5py`, `tqdm`, `shapely`, `matplotlib`, `meshpy`.
Optional: `numba` (acceleration), `opt_einsum` (optimized tensor contractions).

## Citation

If you use this library in your research, please cite both the paper in which the method was first formulated and the software itself:

```bibtex
@article{3nnt-b8xv,
  title = {Vortex structure and intervortex interaction in superconducting structures with intrinsic diode effect},
  author = {Putilov, A. V. and Zakharov, D. V. and Kudlis, A. and Mel'nikov, A. S. and Buzdin, A. I.},
  journal = {Phys. Rev. B},
  volume = {112},
  issue = {13},
  pages = {134507},
  numpages = {14},
  year = {2025},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/3nnt-b8xv},
  url = {https://link.aps.org/doi/10.1103/3nnt-b8xv}
}

@software{tdgl_lsfd_lib,
  author  = {Zakharov, Dmitrii},
  title   = {tdgl\_LSFD\_lib: time-dependent Ginzburg--Landau solver
             on unstructured meshes},
  year    = {2026},
  url     = {https://github.com/ZakharovDmitrii/LSFD_TDGL},
  version = {0.1.0b1},
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.й
