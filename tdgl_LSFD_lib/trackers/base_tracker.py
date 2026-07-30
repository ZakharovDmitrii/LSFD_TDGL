"""
base_tracker.py — Abstract base class for simulation trackers.

This module defines the contract between TDGLRunner and tracker classes.

Architecture overview:
    - Solver computes spatial fields (psi, mu, currents) at each time step.
    - Trackers compute scalar quantities from these fields and build time series.
    - Runner is a coordinator: it runs the time loop and saves results to HDF5.

Storage strategy:
    - Each tracker stores scalar time series in fixed-size numpy buffers in RAM.
    - When a buffer is full (every `save_every` steps), it is flushed to HDF5.
    - This keeps RAM usage constant regardless of simulation length.
    - Final time-series plots are built in post-processing from the HDF5 file.

Tracker interface:
    A tracker must implement exactly two methods:
        1. on_step()    — compute scalars from current StepResult and store in buffer
        2. flush_to_hdf5() — write filled buffer to HDF5 file
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from ..solver.solver import StepResult, TDGLSolver


class SimulationTracker(ABC):
    """
    Abstract base class for all simulation trackers.

    A tracker is a modular component that extracts scalar physical quantities
    (energy, fluxes, probe values, etc.) from the solver's StepResult at each
    time step. These scalars are accumulated into time series and periodically
    flushed to an HDF5 file for permanent storage.

    Design principles:
        1. Separation of concerns:
           - Solver computes spatial fields (functions of coordinates r).
           - Trackers compute scalar quantities (functions of time t).
           - Runner coordinates the time loop and saves results.
        2. Memory efficiency:
           - Fixed-size buffers in RAM (size = save_every).
           - Periodic flush to HDF5 keeps memory usage constant.
        3. Flexibility:
           - Each tracker implements its own logic independently.
           - The number and type of scalars are determined by the tracker itself.
        4. Post-processing:
           - Final time-series plots are built from HDF5 after simulation ends.

    Subclasses MUST implement:
        - on_step(): Compute scalars from StepResult and store in internal buffer.
        - flush_to_hdf5(): Write the filled buffer to an HDF5 group.
    """

    @abstractmethod
    def on_step(
        self,
        step: int,
        t: float,
        dt: float,
        result: StepResult,
        solver: TDGLSolver,
    ) -> Dict[str, Any]:
        """
        Process one simulation step: compute scalars and store them in the buffer.

        This method is called by the Runner at EVERY time step. It should:
            1. Analyze the current StepResult (which contains all spatial fields).
            2. Compute the scalar quantities of interest (e.g., energy, fluxes).
            3. Store these scalars in the tracker's internal numpy buffer.
            4. Optionally return the computed values for immediate use by the Runner.

        Args:
            step (int):
                Current step number (0-indexed, increments each call).
            t (float):
                Current simulation time.
            dt (float):
                Current time step size (may change if adaptive stepping is used).
            result (StepResult):
                Container with all fields produced by the solver at this step:
                    - psi, psi_abs_sq, mu: main state variables
                    - supercurrent_x, supercurrent_y, div_Js: current fields
                    - normal_current: normal current at boundary
                    - psi_derivatives: all derivatives of psi, shape (N, 14)
                    - s_applied, Bz, eta, gamma: external field parameters
                    - poisson_residual, poisson_iterations: solver diagnostics
            solver (TDGLSolver):
                Reference to the solver instance. Provides access to operators,
                mesh, device, and other solver internals if needed.

        Returns:
            Dict[str, Any]:
                Dictionary mapping scalar names to their current values.
                Keys should be descriptive strings (e.g., "energy_voronoi").
                Values can be scalars (float) or arrays (np.ndarray).
                The Runner may use these values for logging or other purposes.

        Performance notes:
            - This method is called at EVERY step, so it must be fast.
            - Avoid unnecessary memory allocations inside this method.
            - Use vectorized NumPy operations where possible.
            - Cache expensive computations (e.g., boundary indices) in __init__.

        Example:
            >>> def on_step(self, step, t, dt, result, solver):
            ...     energy = self._compute_energy(result)
            ...     self._buffer[self._idx] = energy
            ...     self._idx += 1
            ...     return {"energy": energy}
        """
        pass

    @abstractmethod
    def flush_to_hdf5(self, hdf5_group) -> None:
        """
        Flush the filled buffer to an HDF5 file.

        This method is called by the Runner when:
            1. The buffer is full (every `save_every` steps).
            2. The simulation ends (final flush of remaining data).

        It should:
            1. Create HDF5 datasets if they do not yet exist (use maxshape=(None,)
               for resizable dimensions).
            2. Resize the datasets to accommodate the new data.
            3. Append the filled portion of the buffer to the datasets.
            4. Reset the internal buffer index to 0.

        Args:
            hdf5_group:
                An h5py.Group object where the tracker should write its data.
                Typically this is a dedicated group like "time_series".
                The tracker is responsible for creating its own datasets
                within this group (e.g., "energy", "flux_edges", etc.).

        Notes:
            - Use dtype='f8' for float64, 'i8' for int64.
            - Flush only the filled portion of the buffer (not the entire buffer_size).
            - This method should be idempotent (safe to call multiple times).
            - After flushing, the buffer index must be reset to 0.

        Example:
            >>> def flush_to_hdf5(self, hdf5_group):
            ...     if self._idx == 0:
            ...         return
            ...     if "energy" not in hdf5_group:
            ...         hdf5_group.create_dataset(
            ...             "energy", (0,), maxshape=(None,), dtype='f8'
            ...         )
            ...     current_size = hdf5_group["energy"].shape[0]
            ...     new_size = current_size + self._idx
            ...     hdf5_group["energy"].resize(new_size, axis=0)
            ...     hdf5_group["energy"][current_size:new_size] = self._buffer[:self._idx]
            ...     self._idx = 0
        """
        pass