"""
runner.py — Simulation loop, HDF5 persistence, and monitoring.
NEW ARCHITECTURE:
- Solver owns the global clock (solver.t, solver.dt).
- Solver returns (StepResult, StepFields); runner chains them step-to-step.
- Thermalization: no trackers, no time series; only the LAST step is saved
  (as the seed snapshot for the main stage).
- Main stage: saves everything with the TRUE solver time; progress bar shows
  t - skip_time (relative), subtraction for plots is done later in plot_solution.
- External fields (A_applied, J_boundary + scalars) are saved per snapshot.
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
from ..external_fields.external_fields import StepFields
from ..trackers.base_tracker import SimulationTracker
from ..trackers.conservation_tracker import ConservationTracker
from ..trackers.physical_tracker import PhysicalTracker

logger = logging.getLogger(__name__)


# ============================================================================
# DATA HANDLER — HDF5 FILE MANAGEMENT (без изменений)
# ============================================================================
class DataHandler:
    """
    Context manager for reading/writing HDF5 files.
    """
    def __init__(self, output_file: Union[str, None], logger: Optional[logging.Logger] = None):
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

    def _create_output_file(self, output: str):
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
                    self.logger.warning(f"Output file already exists. Renaming to {file_name}.")
                return file, file_path, tmp_file, tmp_file_path

    def __enter__(self) -> "DataHandler":
        (self.output_file, self.output_path, self.tmp_file, self.tmp_path) = \
            self._create_output_file(self._base_output_file)
        self.time_step_group = self.output_file.create_group("data", track_order=True)
        self.time_series_group = self.output_file.create_group("time_series")
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
        self.output_file.close()
        if self.tmp_file is not None:
            self.tmp_file.flush()
            self.tmp_file.close()
            os.remove(self.tmp_path)
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def save_mesh(self, mesh) -> None:
        self.mesh_group = self.output_file.create_group("mesh")
        mesh.to_hdf5(self.mesh_group)

    def save_fixed_values(self, fixed_data: Dict[str, np.ndarray]) -> None:
        for key, value in fixed_data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.output_file[key] = value
            self.tmp_file[key] = value

    def save_time_step(self, state: Dict[str, Any], data: Dict[str, np.ndarray]) -> None:
        group = self.time_step_group.create_group(f"{self.save_number}")
        group.attrs["timestamp"] = datetime.now().isoformat()
        self.save_number += 1
        tmp_grp = self.tmp_file["data/-1"]
        for key, value in state.items():
            group.attrs[key] = value
        for key, value in data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            group[key] = value
            if key in tmp_grp:
                tmp_grp[key][:] = value
            else:
                tmp_grp[key] = value
            tmp_grp[key].flush()
        for key in ("step", "time", "dt"):
            tmp_grp[key][:] = np.array([state[key]])
            tmp_grp[key].flush()





# ============================================================================
# TDGL RUNNER — MAIN CLASS
# ============================================================================
class TDGLRunner:
    def __init__(self, solver: TDGLSolver, options: SolverOptions):
        self.solver = solver
        self.options = options
        self.step = 0                      # ← только счётчик; часы — в солвере
        self.trackers = self._create_trackers()
        buffer_size = options.save_every
        self._time_buf = np.zeros(buffer_size, dtype=np.float64)
        self._step_buf = np.zeros(buffer_size, dtype=np.int64)
        self._dt_buf = np.zeros(buffer_size, dtype=np.float64)
        self._poisson_tol_buf = np.zeros(buffer_size, dtype=np.float64)
        self._buf_idx = 0
        self._runner_datasets_created = False

    def _create_trackers(self) -> List[SimulationTracker]:
        trackers: List[SimulationTracker] = []
        buffer_size = self.options.save_every
        if self.options.track_conservation:
            trackers.append(ConservationTracker(
                mesh=self.solver.device.mesh,
                buffer_size=buffer_size,
                log_every=self.options.tracker_log_every,
            ))
        if self.options.track_physical:
            trackers.append(PhysicalTracker(
                device=self.solver.device,
                buffer_size=buffer_size,
            ))
        logger.info(f"Created {len(trackers)} tracker(s): {[type(t).__name__ for t in trackers]}")
        return trackers

    def run(self, psi_init=None, mu_init=None, seed_solution=None,
            reset_clock: bool = False):
        start_time = time.perf_counter()
        psi_derivatives, fields = None, None
        if seed_solution is not None:
            psi, mu, psi_derivatives, seed_time = self._load_seed_solution(seed_solution)
            if reset_clock:
                # ψ,μ из seed как НАЧАЛЬНОЕ условие для новых полей/операторов
                self.solver.t = 0.0
                self.solver.dt = self.options.dt_init
                self.solver.d_psi_sq_history.clear()
                self.solver.poisson_iterations_history.clear()
                self.solver.poisson_tolerance = self.options.poisson_tolerance_init
                psi_derivatives, fields = None, None
                logger.info("Seed as initial condition: clock reset to t=0.")
            else:
                self.solver.t = seed_time
                logger.info(f"Seed: continuing from t={seed_time:.3f}.")
        else:
            psi = psi_init if psi_init is not None else np.ones(self.solver.n_sites, dtype=np.complex128)
            mu = mu_init if mu_init is not None else np.zeros(self.solver.n_sites, dtype=np.float64)

        with DataHandler(output_file=self.options.output_file, logger=logger) as dh:
            dh.save_mesh(self.solver.device.mesh)
            dh.save_fixed_values({"A_for_constant_Bz": self.solver.A_for_constant_Bz})
            # параметры в attrs файла — Solution/plot_solution вычтут skip_time сами
            dh.output_file.attrs["skip_time"] = self.options.skip_time
            dh.output_file.attrs["solve_time"] = self.options.solve_time
            dh.output_file.attrs["dt_init"] = self.options.dt_init

            do_therm = self.options.skip_time > 0
            if do_therm and seed_solution is not None and not reset_clock:
                logger.info("Continuing from seed time: thermalization skipped.")
                do_therm = False

            # === 1. THERMALIZATION: без трекеров, только последний шаг как seed ===
            if do_therm:
                therm_offset = self.solver.t
                therm_end = self.solver.t + self.options.skip_time
                (success, psi, mu, psi_derivatives, fields, last_res) = self._run_stage(
                    psi=psi, mu=mu, psi_derivatives=psi_derivatives, fields=fields,
                    end_time=therm_end, save=False, track=False,
                    desc="Thermalization", data_handler=dh, t_offset=therm_offset,
                )
                if not success:
                    return {"output_path": dh.output_path, "cancelled": True}
                self._save_snapshot(dh, psi, mu, last_res, fields)   # последний шаг

            # === 2. MAIN: истинное время в файле, бар показывает t - main_offset ===
            main_offset = self.solver.t            # = skip_time при старте с нуля
            main_end = self.solver.t + self.options.solve_time
            (success, psi, mu, psi_derivatives, fields, last_res) = self._run_stage(
                psi=psi, mu=mu, psi_derivatives=psi_derivatives, fields=fields,
                end_time=main_end, save=True, track=True,
                desc="Simulation", data_handler=dh, t_offset=main_offset,
            )
            self._flush_time_series(dh.time_series_group)
            elapsed = time.perf_counter() - start_time
            self._log_final_statistics(elapsed, success)
            return {"output_path": dh.output_path, "final_step": self.step,
                    "final_time": self.solver.t, "cancelled": not success}

    def _run_stage(self, psi, mu, psi_derivatives, fields, end_time, save, track,
                   desc, data_handler, t_offset=0.0):
        psi_abs_sq = np.abs(psi) ** 2
        bar_format = "{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"
        cancelled = False
        save_counter = 0
        last_result = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)
            with tqdm(initial=0.0, total=end_time - t_offset, desc=desc,
                      disable=not self.options.use_tqdm, unit="τ₀",
                      bar_format=bar_format, dynamic_ncols=True) as pbar:
                for _ in itertools.count():
                    try:
                        iter_start = time.perf_counter()
                        t_before = self.solver.t
                        # === ОДИН ШАГ: солвер сам двигает часы ===
                        result, fields = self.solver.solve_for_one_step(
                            psi=psi, psi_abs_sq=psi_abs_sq, mu=mu,
                            fields=fields, psi_derivatives=psi_derivatives,
                        )
                        dt_used = self.solver.t - t_before
                        psi = result.psi
                        psi_abs_sq = result.psi_abs_sq
                        mu = result.mu
                        psi_derivatives = result.psi_derivatives
                        self.step += 1
                        save_counter += 1
                        last_result = result
                        iter_time = time.perf_counter() - iter_start

                        pbar.update(dt_used)
                        pbar.set_postfix({
                            't': f'{self.solver.t - t_offset:.2f}',
                            'dt': f'{dt_used:.2e}',
                            'tol': f'{result.poisson_tolerance:.2e}',
                            'iter': f'{iter_time * 1000:.1f}ms',
                        })

                        # трекеры + буфер — только в main-стадии
                        if track:
                            for tracker in self.trackers:
                                tracker.on_step(step=self.step, t=self.solver.t,
                                                dt=dt_used, result=result, solver=self.solver)
                            self._time_buf[self._buf_idx] = self.solver.t  # истинное время
                            self._step_buf[self._buf_idx] = self.step
                            self._dt_buf[self._buf_idx] = dt_used
                            self._poisson_tol_buf[self._buf_idx] = result.poisson_tolerance
                            self._buf_idx += 1

                        if save and (save_counter % self.options.save_every == 0):
                            self._save_snapshot(data_handler, psi, mu, result, fields)
                            self._flush_time_series(data_handler.time_series_group)

                        if self.solver.t >= end_time:
                            break
                    except KeyboardInterrupt:
                        if self.options.use_pause:
                            response = input(f"\nPaused at {desc!r} (step {self.step}). Continue? [yN] ")
                            if response.lower().startswith('y'):
                                continue
                        cancelled = True
                        break
        if save and last_result is not None and (save_counter % self.options.save_every != 0):
            self._save_snapshot(data_handler, psi, mu, last_result, fields)
        return not cancelled, psi, mu, psi_derivatives, fields, last_result

    def _save_snapshot(self, data_handler, psi, mu, result, fields=None):
        state = {"step": self.step, "time": self.solver.t, "dt": self.solver.dt}
        if fields is not None:
            state["Bz"] = fields.Bz
            state["eta"] = fields.eta
            state["gamma"] = fields.gamma
        data = {
            "psi": psi, "mu": mu,
            "psi_derivatives": result.psi_derivatives,  # ← для seed
            "supercurrent_x": result.supercurrent_x,
            "supercurrent_y": result.supercurrent_y,
            "div_Js": result.div_Js,
            "normal_current": result.normal_current,
        }
        if fields is not None:
            data["A_applied"] = fields.A_applied
            data["J_boundary"] = fields.J_boundary
            data["s_applied"] = fields.s_applied
        data_handler.save_time_step(state, data)
        
    def _load_seed_solution(self, seed_path: str):
        logger.info(f"Loading initial conditions from {seed_path}")
        with h5py.File(seed_path, "r") as f:
            grp = f["data"][str(max(int(k) for k in f["data"].keys()))]
            psi = np.array(grp["psi"])
            mu = np.array(grp["mu"])
            psi_derivatives = np.array(grp["psi_derivatives"]) if "psi_derivatives" in grp else None
            seed_time = float(grp.attrs.get("time", 0.0))
        logger.info(f"Loaded: psi {psi.shape}, mu {mu.shape}, "
                    f"derivs={'yes' if psi_derivatives is not None else 'no'}, t={seed_time:.3f}")
        return psi, mu, psi_derivatives, seed_time

    # _flush_time_series / _ensure_runner_datasets / _log_final_statistics — без изменений

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