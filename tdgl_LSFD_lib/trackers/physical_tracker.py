"""
physical_tracker.py — Tracker for physical observables at probe points.

Monitors:
    - Scalar potential mu at specified probe points
    - Phase of order parameter psi at probe points (in units of π)

Probe points are fixed spatial locations defined in device.probe_points.
The number of probes is constant throughout the simulation.

Uses fixed-size numpy buffers in RAM, flushed to HDF5 periodically.

HDF5 output:
    time_series/
    ├── mu_probe          — shape (time, n_probes)
    ├── phase_probe       — shape (time, n_probes)
    ├── probe_points_coords  — shape (n_probes, 2), coordinates [x, y]
    └── probe_points_indices — shape (n_probes,), mesh site indices
"""
import logging
from typing import Dict, Any, Optional

import numpy as np

from .base_tracker import SimulationTracker
from ..solver.solver import StepResult, TDGLSolver
from ..device.device import Device

logger = logging.getLogger(__name__)


class PhysicalTracker(SimulationTracker):
    """
    Tracks mu and phase of psi at user-defined probe points.

    Args:
        device: Device object containing probe_points (list of [x, y] coordinates).
        buffer_size: Size of memory buffer (typically = save_every).

    Raises:
        ValueError: If device.probe_points is None or empty.
    """

    def __init__(
        self,
        device: Device,
        buffer_size: int = 100,
    ):
        self.device = device
        self.buffer_size = buffer_size

        # Validate probe points
        if device.probe_points is None or len(device.probe_points) == 0:
            raise ValueError(
                "PhysicalTracker requires device.probe_points to be set. "
                "Provide a list of [x, y] coordinates."
            )

        self.n_probes = len(device.probe_points)
        self._probe_indices: Optional[np.ndarray] = None  # Lazy init (need solver)

        # === BUFFERS (fixed-size numpy arrays, shape: buffer_size × n_probes) ===
        self._mu_buf = np.zeros((buffer_size, self.n_probes), dtype=np.float64)
        self._phase_buf = np.zeros((buffer_size, self.n_probes), dtype=np.float64)

        # Current buffer index
        self._idx = 0

        # HDF5 datasets created flag
        self._datasets_created = False

    def _resolve_probe_indices(self, solver: TDGLSolver) -> None:
        """Map probe point coordinates to nearest mesh site indices."""
        if self._probe_indices is not None:
            return

        self._probe_indices = np.array([
            solver.device.closest_site(xy)
            for xy in self.device.probe_points
        ], dtype=np.int64)

        logger.debug(
            f"PhysicalTracker: {self.n_probes} probe points resolved to mesh sites "
            f"{self._probe_indices}"
        )

    def on_step(
        self,
        step: int,
        t: float,
        dt: float,
        result: StepResult,
        solver: TDGLSolver,
    ) -> Dict[str, Any]:
        """Extract mu and phase at probe points and store in buffer."""
        self._resolve_probe_indices(solver)

        idx = self._idx

        # mu at probe points
        self._mu_buf[idx, :] = result.mu[self._probe_indices]

        # Phase of psi at probe points (in units of π)
        self._phase_buf[idx, :] = np.angle(result.psi[self._probe_indices]) / np.pi

        self._idx += 1

        return {
            "mu_probe": self._mu_buf[idx - 1, :],
            "phase_probe": self._phase_buf[idx - 1, :],
        }

    def flush_to_hdf5(self, hdf5_group) -> None:
        """Flush filled buffer to HDF5 file."""
        if self._idx == 0:
            return

        n = self._idx

        # Create datasets on first flush (resizable along time axis)
        if not self._datasets_created:
            # Time series datasets
            hdf5_group.create_dataset(
                "mu_probe", (0, self.n_probes),
                maxshape=(None, self.n_probes), dtype='f8',
            )
            hdf5_group.create_dataset(
                "phase_probe", (0, self.n_probes),
                maxshape=(None, self.n_probes), dtype='f8',
            )

            # Probe points metadata (fixed, not resizable)
            probe_coords = np.array(self.device.probe_points, dtype=np.float64)
            hdf5_group.create_dataset(
                "probe_points_coords", data=probe_coords, dtype='f8',
            )
            hdf5_group["probe_points_coords"].attrs["description"] = (
                "Coordinates [x, y] of each probe point. "
                "Column i corresponds to probe point index i."
            )

            # Probe site indices (will be filled after first on_step)
            if self._probe_indices is not None:
                hdf5_group.create_dataset(
                    "probe_points_indices", data=self._probe_indices, dtype='i8',
                )
                hdf5_group["probe_points_indices"].attrs["description"] = (
                    "Mesh site indices corresponding to each probe point. "
                    "Index i corresponds to probe point index i."
                )

            self._datasets_created = True
        elif self._probe_indices is not None and "probe_points_indices" not in hdf5_group:
            # Save probe indices if they were resolved after first flush
            hdf5_group.create_dataset(
                "probe_points_indices", data=self._probe_indices, dtype='i8',
            )
            hdf5_group["probe_points_indices"].attrs["description"] = (
                "Mesh site indices corresponding to each probe point. "
                "Index i corresponds to probe point index i."
            )

        # Resize and append time series
        current_size = hdf5_group["mu_probe"].shape[0]
        new_size = current_size + n

        hdf5_group["mu_probe"].resize(new_size, axis=0)
        hdf5_group["mu_probe"][current_size:new_size, :] = self._mu_buf[:n, :]

        hdf5_group["phase_probe"].resize(new_size, axis=0)
        hdf5_group["phase_probe"][current_size:new_size, :] = self._phase_buf[:n, :]

        # Reset buffer index
        self._idx = 0

        logger.debug(f"PhysicalTracker flushed {n} steps to HDF5")