"""
runner.py — Simulation loop, HDF5 persistence, and monitoring.

Architecture:
    - TDGLRunner orchestrates the simulation loop.
    - Solver computes spatial fields (psi, mu, currents) at each step.
    - Trackers extract scalar observables and store them in fixed-size buffers.
    - Runner stores time/step/dt/poisson_tolerance in its own buffer.
    - Every `save_every` steps (and at the end), Runner flushes:
        * its own buffer (time, step, dt, poisson_tolerance) to HDF5
        * each tracker's buffer to HDF5
        * the current spatial fields (psi, mu, currents) to HDF5

HDF5 layout:
    output.h5
    ├── mesh/                    — mesh geometry (once)
    ├── A_for_constant_Bz        — fixed field (once)
    ├── data/                    — spatial snapshots every save_every steps
    │   ├── 0/
    │   │   ├── psi, mu, supercurrent_x, ...
    │   │   └── attrs: step, time, dt
    │   └── 1/
    └── time_series/             — scalar time series (every step)
        ├── time, step, dt, poisson_tolerance   (from Runner)
        ├── energy_voronoi, flux_edges, ...     (from ConservationTracker)
        ├── mu_probe, phase_probe               (from PhysicalTracker)
        ├── probe_points_coords, probe_points_indices  (from PhysicalTracker)
"""
import itertools
import logging
import os
import tempfile
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm import TqdmWarning, tqdm

from .solver import TDGLSolver, StepResult
from .dynamics_options import SolverOptions
from ..trackers.base_tracker import SimulationTracker
from ..trackers.conservation_tracker import ConservationTracker
from ..trackers.physical_tracker import PhysicalTracker

logger = logging.getLogger(__name__)


# ============================================================================
# DATA HANDLER — HDF5 FILE MANAGEMENT
# ============================================================================

class DataHandler:
    """
    Context manager for reading/writing HDF5 files.

    Creates two files:
        - output.h5     — main data (snapshots every save_every steps)
        - output.h5.tmp — temporary file for live monitoring (last step only)
    """

    def __init__(
        self,
        output_file: Union[str, None],
        logger: Optional[logging.Logger] = None,
    ):
        self.tempdir = None
        self.mesh_group = None
        self.time_step_group = None
        self.time_series_group = None
        self.save_number = 0
        self.logger = logger if logger is not None else logging.getLogger()
        self._base_output_file = output_file

        self.output_file: Optional[h5py.File] = None
        self.output_path: Optional[str] = None
        self.tmp_file: Optional[h5py.File] = None
        self.tmp_path: Optional[str] = None

    def _create_output_file(self, output: str) -> Tuple[h5py.File, str, h5py.File, str]:
        """Create the output file and the temporary monitoring file."""
        if output is None:
            self.tempdir = tempfile.TemporaryDirectory()
            directory = self.tempdir.name
            name = "output"
            suffix = "h5"
        else:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            name_parts = output.split(".")
            name = ".".join(name_parts[:-1])
            suffix = name_parts[-1]
            directory = os.getcwd()

        serial_number = None
        while True:
            name_suffix = f"-{serial_number}" if serial_number is not None else ""
            file_name = f"{name}{name_suffix}.{suffix}"
            file_path = os.path.join(directory, file_name)
            tmp_file_name = f"{file_name}.tmp"
            tmp_file_path = os.path.join(directory, tmp_file_name)

            try:
                file = h5py.File(file_path, "x")
                tmp_file = h5py.File(tmp_file_path, "x", libver="latest")
            except (OSError, FileExistsError):
                serial_number = 1 if serial_number is None else serial_number + 1
                continue
            else:
                if serial_number is not None:
                    self.logger.warning(
                        f"Output file already exists. Renaming to {file_name}."
                    )
                return file, file_path, tmp_file, tmp_file_path

    def __enter__(self) -> "DataHandler":
        (
            self.output_file,
            self.output_path,
            self.tmp_file,
            self.tmp_path,
        ) = self._create_output_file(self._base_output_file)

        self.time_step_group = self.output_file.create_group("data", track_order=True)
        self.time_series_group = self.output_file.create_group("time_series")

        # Initialize the temporary file for live monitoring
        grp = self.tmp_file.create_group("data/-1")
        grp["step"] = np.array([0])
        grp["time"] = np.array([0.0])
        grp["dt"] = np.array([0.0])

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_value is not None:
            self.logger.warning("Ignoring the following exception in DataHandler.__exit__():")
            self.logger.warning(" ".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        self.close()

    def close(self) -> None:
        """Close files and clean up temporary data."""
        self.output_file.close()
        if self.tmp_file is not None:
            self.tmp_file.flush()
            self.tmp_file.close()
            os.remove(self.tmp_path)
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def save_mesh(self, mesh) -> None:
        """Save the mesh geometry once."""
        self.mesh_group = self.output_file.create_group("mesh")
        mesh.to_hdf5(self.mesh_group)

    def save_fixed_values(self, fixed_data: Dict[str, np.ndarray]) -> None:
        """Save fixed values (do not change over time)."""
        for key, value in fixed_data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.output_file[key] = value
            self.tmp_file[key] = value

    def save_time_step(
        self,
        state: Dict[str, Any],
        data: Dict[str, np.ndarray],
    ) -> None:
        """Save a spatial snapshot (every save_every steps)."""
        group = self.time_step_group.create_group(f"{self.save_number}")
        group.attrs["timestamp"] = datetime.now().isoformat()
        self.save_number += 1

        tmp_grp = self.tmp_file["data/-1"]

        # State attributes
        for key, value in state.items():
            group.attrs[key] = value

        # Field data
        for key, value in data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            group[key] = value

            # Update the temporary monitoring file
            if key in tmp_grp:
                tmp_grp[key][:] = value
            else:
                tmp_grp[key] = value
            tmp_grp[key].flush()

        # Update base attributes in the temporary file
        for key in ("step", "time", "dt"):
            tmp_grp[key][:] = np.array([state[key]])
            tmp_grp[key].flush()


# ============================================================================
# TDGL RUNNER — MAIN CLASS
# ============================================================================

class TDGLRunner:
    """
    Runs a TDGL simulation with thermalization, field saving, and adaptive stepping.

    Trackers are created automatically based on SolverOptions flags:
        - track_conservation=True  → ConservationTracker
        - track_physical=True      → PhysicalTracker (requires device.probe_points)
        - track_vortices=True      → VortexTracker (future)

    Args:
        solver: TDGLSolver instance.
        options: SolverOptions with simulation parameters.
    """

    def __init__(
        self,
        solver: TDGLSolver,
        options: SolverOptions,
    ):
        self.solver = solver
        self.options = options

        # Simulation state
        self.t = 0.0
        self.dt = options.dt_init
        self.step = 0

        # === CREATE TRACKERS based on options flags ===
        self.trackers: List[SimulationTracker] = self._create_trackers()

        # === RUNNER'S OWN BUFFER for time/step/dt/poisson_tolerance ===
        # These are shared across all trackers, so Runner stores them once.
        buffer_size = options.save_every
        self._time_buf = np.zeros(buffer_size, dtype=np.float64)
        self._step_buf = np.zeros(buffer_size, dtype=np.int64)
        self._dt_buf = np.zeros(buffer_size, dtype=np.float64)
        self._poisson_tol_buf = np.zeros(buffer_size, dtype=np.float64)
        self._buf_idx = 0

        # HDF5 datasets created flag (for Runner's own time series)
        self._runner_datasets_created = False

    def _create_trackers(self) -> List[SimulationTracker]:
        """
        Create trackers based on SolverOptions flags.

        Returns:
            List of SimulationTracker instances.

        Raises:
            ValueError: If track_physical=True but device.probe_points is not set.
        """
        trackers: List[SimulationTracker] = []
        buffer_size = self.options.save_every

        # 1. ConservationTracker
        if self.options.track_conservation:
            trackers.append(ConservationTracker(
                mesh=self.solver.device.mesh,
                buffer_size=buffer_size,
                log_every=self.options.tracker_log_every,
            ))

        # 2. PhysicalTracker — requires probe_points
        if self.options.track_physical:
            # PhysicalTracker.__init__ will raise ValueError if probe_points is None
            trackers.append(PhysicalTracker(
                device=self.solver.device,
                buffer_size=buffer_size,
            ))

        # 3. VortexTracker (future)
        # if self.options.track_vortices:
        #     from ..trackers.vortex_tracker import VortexTracker
        #     trackers.append(VortexTracker(
        #         mesh=self.solver.device.mesh,
        #         max_vortices=self.options.max_vortices,
        #         buffer_size=buffer_size,
        #     ))

        logger.info(
            f"Created {len(trackers)} tracker(s): "
            f"{[type(t).__name__ for t in trackers]}"
        )

        return trackers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        psi_init: Optional[np.ndarray] = None,
        mu_init: Optional[np.ndarray] = None,
        seed_solution: Optional[str] = None,
    ) -> Dict[str, Union[str, int, float, bool]]:
        """Run the full simulation (thermalization + main stage)."""
        start_time = time.perf_counter()

        # Load initial conditions
        if seed_solution is not None:
            psi, mu = self._load_seed_solution(seed_solution)
        else:
            psi = psi_init if psi_init is not None else np.ones(
                self.solver.n_sites, dtype=np.complex128
            )
            mu = mu_init if mu_init is not None else np.zeros(
                self.solver.n_sites, dtype=np.float64
            )

        with DataHandler(output_file=self.options.output_file, logger=logger) as data_handler:
            # Save mesh and fixed values once
            data_handler.save_mesh(self.solver.device.mesh)
            data_handler.save_fixed_values({
                "A_for_constant_Bz": self.solver.A_for_constant_Bz,
            })

            # === 1. THERMALIZATION ===
            if self.options.skip_time > 0:
                logger.info(f"Thermalization: t in [0, {self.options.skip_time}]")
                success = self._run_stage(
                    psi=psi, mu=mu,
                    end_time=self.options.skip_time,
                    save=False,
                    desc="Thermalization",
                    data_handler=data_handler,
                )
                if not success:
                    logger.warning("Thermalization cancelled by user")
                    return {"output_path": data_handler.output_path, "cancelled": True}

                # Reset state after thermalization
                self._reset_buffers()
                self.t = 0.0
                self.step = 0

            # === 2. MAIN SIMULATION ===
            logger.info(f"Simulation: t in [0, {self.options.solve_time}]")
            success = self._run_stage(
                psi=psi, mu=mu,
                end_time=self.options.solve_time,
                save=True,
                desc="Simulation",
                data_handler=data_handler,
            )

            # Final flush of any remaining buffered time-series data
            self._flush_time_series(data_handler.time_series_group)

            elapsed = time.perf_counter() - start_time
            self._log_final_statistics(elapsed, success)

            return {
                "output_path": data_handler.output_path,
                "final_step": self.step,
                "final_time": self.t,
                "cancelled": not success,
            }

    # ------------------------------------------------------------------
    # Internal: simulation loop
    # ------------------------------------------------------------------

    def _run_stage(
            self,
            psi: np.ndarray,
            mu: np.ndarray,
            end_time: float,
            save: bool,
            desc: str,
            data_handler: DataHandler,
    ) -> bool:
        """Run one simulation stage (thermalization or main)."""
        psi_abs_sq = np.abs(psi) ** 2

        # Progress bar setup
        bar_format = "{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
        cancelled = False
        save_counter = 0
        last_result: Optional[StepResult] = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)
            with tqdm(
                    initial=self.t,
                    total=end_time,
                    desc=desc,
                    disable=not self.options.use_tqdm,
                    unit="τ₀",
                    bar_format=bar_format,
                    dynamic_ncols=True,
            ) as pbar:

                for _ in itertools.count():
                    try:
                        iter_start = time.perf_counter()

                        # === ONE SOLVER STEP ===
                        result: StepResult = self.solver.solve_for_one_step(
                            psi=psi,
                            psi_abs_sq=psi_abs_sq,
                            mu=mu,
                            t=self.t,
                            dt=self.dt,
                        )

                        # Unpack result
                        psi = result.psi
                        psi_abs_sq = result.psi_abs_sq
                        mu = result.mu
                        self.dt = result.dt  # adaptive step

                        iter_time = time.perf_counter() - iter_start

                        # Update time and step counter
                        self.t += self.dt
                        self.step += 1
                        save_counter += 1
                        last_result = result

                        # Update progress bar
                        pbar.update(self.dt)
                        pbar.set_postfix({
                            'dt': f'{self.dt:.2e}',
                            'tol': f'{result.poisson_tolerance:.2e}',
                            'iter': f'{iter_time * 1000:.1f}ms',
                        })

                        # === UPDATE TRACKERS ===
                        for tracker in self.trackers:
                            tracker.on_step(
                                step=self.step, t=self.t, dt=self.dt,
                                result=result, solver=self.solver,
                            )

                        # === UPDATE RUNNER'S BUFFER ===
                        self._time_buf[self._buf_idx] = self.t
                        self._step_buf[self._buf_idx] = self.step
                        self._dt_buf[self._buf_idx] = self.dt
                        self._poisson_tol_buf[self._buf_idx] = result.poisson_tolerance
                        self._buf_idx += 1

                        # === SAVE EVERY save_every STEPS ===
                        if save and (save_counter % self.options.save_every == 0):
                            self._save_snapshot(data_handler, psi, mu, result)
                            self._flush_time_series(data_handler.time_series_group)

                        # Check end of stage
                        if self.t >= end_time:
                            break

                    except KeyboardInterrupt:
                        if self.options.use_pause:
                            response = input(
                                f"\nSimulation paused at stage {desc!r} (step {self.step}). "
                                "Continue? [yN] "
                            )
                            if response.lower().startswith('y'):
                                logger.info("Resuming simulation")
                                continue
                            else:
                                logger.warning("Cancelling simulation")
                                cancelled = True
                                break
                        else:
                            logger.warning("Cancelling simulation")
                            cancelled = True
                            break

        # Save the last step if it wasn't saved in the loop
        if save and last_result is not None and (save_counter % self.options.save_every != 0):
            self._save_snapshot(data_handler, psi, mu, last_result)

        return not cancelled

    # ------------------------------------------------------------------
    # Internal: HDF5 saving and flushing
    # ------------------------------------------------------------------

    def _save_snapshot(
        self,
        data_handler: DataHandler,
        psi: np.ndarray,
        mu: np.ndarray,
        result: StepResult,
    ) -> None:
        """Save a spatial snapshot (fields) to HDF5."""
        state = {"step": self.step, "time": self.t, "dt": self.dt}
        data = {
            "psi": psi,
            "mu": mu,
            "supercurrent_x": result.supercurrent_x,
            "supercurrent_y": result.supercurrent_y,
            "div_Js": result.div_Js,
            "normal_current": result.normal_current,
        }
        data_handler.save_time_step(state, data)

    def _flush_time_series(self, time_series_group) -> None:
        """Flush Runner's buffer + all trackers' buffers to HDF5.

        This is called every save_every steps and at the end of simulation
        to save buffered time-series data.
        """
        if self._buf_idx == 0 and all(t._idx == 0 for t in self.trackers):
            return  # Nothing to flush

        n = self._buf_idx

        # 1. Flush Runner's own time series
        if n > 0:
            self._ensure_runner_datasets(time_series_group)
            current_size = time_series_group["time"].shape[0]
            new_size = current_size + n

            time_series_group["time"].resize(new_size, axis=0)
            time_series_group["time"][current_size:new_size] = self._time_buf[:n]

            time_series_group["step"].resize(new_size, axis=0)
            time_series_group["step"][current_size:new_size] = self._step_buf[:n]

            time_series_group["dt"].resize(new_size, axis=0)
            time_series_group["dt"][current_size:new_size] = self._dt_buf[:n]

            time_series_group["poisson_tolerance"].resize(new_size, axis=0)
            time_series_group["poisson_tolerance"][current_size:new_size] = self._poisson_tol_buf[:n]

            self._buf_idx = 0

        # 2. Flush all trackers
        for tracker in self.trackers:
            tracker.flush_to_hdf5(time_series_group)

    def _ensure_runner_datasets(self, time_series_group) -> None:
        """Create Runner's HDF5 datasets on first flush."""
        if self._runner_datasets_created:
            return

        time_series_group.create_dataset("time", (0,), maxshape=(None,), dtype='f8')
        time_series_group.create_dataset("step", (0,), maxshape=(None,), dtype='i8')
        time_series_group.create_dataset("dt", (0,), maxshape=(None,), dtype='f8')
        time_series_group.create_dataset("poisson_tolerance", (0,), maxshape=(None,), dtype='f8')

        self._runner_datasets_created = True

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        """Reset Runner's buffer (e.g., after thermalization)."""
        self._buf_idx = 0
        # Note: tracker buffers are not reset here — they keep their data.
        # If you want to discard thermalization data, create trackers fresh
        # or call tracker._idx = 0 manually.

    def _load_seed_solution(self, seed_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load initial conditions from an HDF5 file."""
        logger.info(f"Loading initial conditions from {seed_path}")
        with h5py.File(seed_path, "r") as f:
            data_group = f["data"]
            steps = [int(key) for key in data_group.keys()]
            last_step = str(max(steps))
            psi = np.array(data_group[last_step]["psi"])
            mu = np.array(data_group[last_step]["mu"])
        logger.info(f"Loaded: psi shape={psi.shape}, mu shape={mu.shape}")
        return psi, mu

    def _log_final_statistics(self, elapsed: float, success: bool) -> None:
        """Log final simulation statistics."""
        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.2f} s")
        logger.info(f"Total steps: {self.step}")
        if self.step > 0:
            logger.info(f"Avg time per step: {elapsed / self.step * 1000:.2f} ms")
            logger.info(f"Avg speed: {self.step / elapsed:.2f} steps/s")
        logger.info(f"Status: {'SUCCESS' if success else 'CANCELLED'}")
        logger.info("=" * 60)