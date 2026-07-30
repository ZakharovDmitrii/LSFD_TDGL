"""
solve.py — High-level interface for running TDGL simulations.

This module provides the `solve()` function, which is the main entry point
for users of the tdgl_LSFD_lib library. It orchestrates the creation of
the solver, runner, and trackers, and returns a Solution object for
post-processing.

Architecture:
    - solve() is a thin facade that wires everything together.
    - TDGLSolver handles the physics (one step of TDGL dynamics).
    - TDGLRunner manages the time loop, HDF5 persistence, and trackers.
    - Trackers (ConservationTracker, PhysicalTracker, VortexTracker) are
      created automatically by Runner based on SolverOptions flags.

Usage:
    # >>> from tdgl_LSFD_lib import solve, Device, SolverOptions
    # >>> from tdgl_LSFD_lib.external_fields import ExternalFields
    # >>> from tdgl_LSFD_lib.operators import LSFD_operators
    # >>>
    # >>> device = Device(...)
    # >>> device.make_mesh()
    # >>> operators = LSFD_operators(device.mesh)
    # >>> external_fields = ExternalFields(...)
    # >>> options = SolverOptions(solve_time=100.0)
    # >>>
    # >>> solution = solve(device, operators, external_fields, options)
    # >>> solution.plot_energy()  # post-processing
"""
import logging
from typing import Optional

import numpy as np

from ..device.device import Device
from ..external_fields.external_fields import ExternalFields
from ..operators.operators import LSFD_operators
from ..post_processing.solution import Solution
from .dynamics_options import SolverOptions
from .runner import TDGLRunner
from .solver import TDGLSolver

logger = logging.getLogger(__name__)


def solve(
    device: Device,
    operators: LSFD_operators,
    external_fields: ExternalFields,
    options: SolverOptions,
    psi_init: Optional[np.ndarray] = None,
    mu_init: Optional[np.ndarray] = None,
    seed_solution: Optional[str] = None,
) -> Solution:
    """
    Run a TDGL simulation.

    This is the main entry point for the library. It sets up logging,
    creates the solver and runner, runs the simulation, and returns a
    Solution object pointing to the output HDF5 file.

    Trackers are created automatically by TDGLRunner based on SolverOptions:
        - track_conservation=True  → ConservationTracker (energy, fluxes, residual)
        - track_physical=True      → PhysicalTracker (mu, phase at probe points)
        - track_vortices=True      → VortexTracker (future)

    Args:
        device: Device geometry and mesh (mesh must be built before calling solve).
        operators: LSFD operators for differential operations on the mesh.
        external_fields: External fields (magnetic, transport, ferromagnetic).
        options: SolverOptions with simulation parameters (time, step, trackers, etc.).
        psi_init: Initial wave function. If None, uses psi=1 everywhere.
        mu_init: Initial scalar potential. If None, uses mu=0 everywhere.
        seed_solution: Path to HDF5 file with initial conditions (for continuation).

    Returns:
        Solution object wrapping the output HDF5 file. Provides methods for
        post-processing (plotting energy, reading fields, etc.).

    Raises:
        ValueError: If track_physical=True but device.probe_points is not set.
        ValueError: If any SolverOptions parameter is invalid.

    # Example:
    #     >>> # Basic usage with default trackers
    #     >>> options = SolverOptions(solve_time=100.0)
    #     >>> solution = solve(device, operators, external_fields, options)
    #
    #     >>> # Disable PhysicalTracker (no probe points)
    #     >>> options = SolverOptions(solve_time=100.0, track_physical=False)
    #     >>> solution = solve(device, operators, external_fields, options)
    #
    #     >>> # Continue from previous solution
    #     >>> options = SolverOptions(solve_time=200.0)
    #     >>> solution = solve(device, operators, external_fields, options,
    #     ...                  seed_solution="previous_output.h5")
    """
    # 1. Setup logging BEFORE creating any objects
    _setup_logging(options)

    # 2. Validate configuration early (fail fast with clear error)
    _validate_configuration(device, options)

    logger.info(
        f"Starting TDGL simulation: "
        f"solve_time={options.solve_time}, "
        f"time_scheme={options.time_scheme.value}, "
        f"run_mode={options.run_mode.value}"
    )
    logger.info(
        f"Trackers: "
        f"conservation={options.track_conservation}, "
        f"physical={options.track_physical}, "
        f"vortices={options.track_vortices}"
    )

    # 3. Create solver
    solver = TDGLSolver(
        device=device,
        operators=operators,
        external_fields=external_fields,
        options=options,
    )

    # 4. Initial conditions
    if psi_init is None:
        psi_init = np.ones(device.n_sites, dtype=np.complex128)
    if mu_init is None:
        mu_init = np.zeros(device.n_sites, dtype=np.float64)

    # 5. Create and run Runner
    # Note: Runner creates trackers automatically based on options flags
    runner = TDGLRunner(solver=solver, options=options)

    result_info = runner.run(
        psi_init=psi_init,
        mu_init=mu_init,
        seed_solution=seed_solution,
    )

    # 6. Return Solution object for post-processing
    return Solution(path=result_info['output_path'])


def _validate_configuration(device: Device, options: SolverOptions) -> None:
    """
    Validate that the configuration is consistent before starting simulation.

    Catches common misconfigurations early with clear error messages,
    before any expensive objects (solver, mesh operations) are created.

    Args:
        device: Device to validate.
        options: SolverOptions to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    # PhysicalTracker requires probe_points
    if options.track_physical:
        if device.probe_points is None or len(device.probe_points) == 0:
            raise ValueError(
                "track_physical=True but device.probe_points is not set. "
                "Either set device.probe_points (list of [x, y] coordinates) "
                "or set options.track_physical=False."
            )

    # VortexTracker validation (future)
    # if options.track_vortices:
    #     if options.max_vortices < 1:
    #         raise ValueError(...)


def _setup_logging(options: SolverOptions) -> None:
    """
    Configure logging based on SolverOptions.

    Sets up the root logger for the tdgl_LSFD_lib package with:
        - Console handler (always)
        - File handler (if options.log_file is set)
        - Log level from options.log_level

    This function is idempotent — safe to call multiple times.
    Old handlers are cleared before adding new ones.

    Args:
        options: SolverOptions with log_file and log_level.
    """
    # Message format
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # Map string level names to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(options.log_level.upper(), logging.INFO)

    # Create handlers
    handlers = [logging.StreamHandler()]  # Always log to console

    # Add file handler if log_file is specified
    if options.log_file is not None:
        handlers.append(logging.FileHandler(str(options.log_file), mode='w'))

    # Configure the library root logger (not Python's root, to avoid affecting other libs)
    root_logger = logging.getLogger('tdgl_LSFD_lib')
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Clear old handlers (for repeated runs)

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Prevent logs from propagating to Python's root logger (avoids duplication)
    root_logger.propagate = False