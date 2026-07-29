"""
runner.py — Цикл симуляции, сохранение в HDF5 и мониторинг.
"""
import itertools
import logging
import numbers
import os
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import TqdmWarning, tqdm

from .solver import TDGLSolver, StepResult
from .dynamics_options import SolverOptions, TimeScheme
from ..operators.fvm_integrator import FVMIntegrator

logger = logging.getLogger(__name__)


# ============================================================================
# DATA HANDLER — УПРАВЛЕНИЕ HDF5 ФАЙЛАМИ
# ============================================================================

class DataHandler:
    """
    Контекстный менеджер для чтения/записи в HDF5.

    Создаёт два файла:
    - output.h5 — основные данные (сохраняются каждые save_every шагов)
    - output.h5.tmp — временный файл для мониторинга (последний шаг)
    """

    def __init__(
        self,
        output_file: Union[str, None],
        logger: Optional[logging.Logger] = None,
    ):
        self.tempdir = None
        self.mesh_group = None
        self.time_step_group = None
        self.save_number = 0
        self.logger = logger if logger is not None else logging.getLogger()
        self._base_output_file = output_file

        self.output_file: Optional[h5py.File] = None
        self.output_path: Optional[str] = None
        self.tmp_file: Optional[h5py.File] = None
        self.tmp_path: Optional[str] = None

    def _create_output_file(self, output: str) -> Tuple[h5py.File, str, h5py.File, str]:
        """Создаёт выходной файл и временный файл для мониторинга."""
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
                if serial_number is None:
                    serial_number = 1
                else:
                    serial_number += 1
                continue
            else:
                if serial_number is not None:
                    self.logger.warning(f"Output file already exists. Renaming to {file_name}.")
                return file, file_path, tmp_file, tmp_file_path

    def __enter__(self) -> "DataHandler":
        (
            self.output_file,
            self.output_path,
            self.tmp_file,
            self.tmp_path,
        ) = self._create_output_file(self._base_output_file)

        self.time_step_group = self.output_file.create_group("data", track_order=True)

        # Инициализация временного файла для мониторинга
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

    def close(self):
        """Закрывает файлы и очищает временные данные."""
        self.output_file.close()
        if self.tmp_file is not None:
            self.tmp_file.flush()
            self.tmp_file.close()
            os.remove(self.tmp_path)
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def save_mesh(self, mesh) -> None:
        """Сохраняет сетку в output_file['mesh']."""
        self.mesh_group = self.output_file.create_group("mesh")
        mesh.to_hdf5(self.mesh_group)

    def save_fixed_values(self, fixed_data: Dict[str, np.ndarray]) -> None:
        """Сохраняет фиксированные значения (не меняются со временем)."""
        for key, value in fixed_data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            self.output_file[key] = value
            self.tmp_file[key] = value

    def save_time_step(
        self,
        state: Dict[str, Any],
        data: Dict[str, np.ndarray],
        running_state: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Сохраняет состояние и данные на каждом шаге."""
        group = self.time_step_group.create_group(f"{self.save_number}")
        group.attrs["timestamp"] = datetime.now().isoformat()
        self.save_number += 1

        tmp_grp = self.tmp_file["data/-1"]

        # Сохранение атрибутов состояния
        for key, value in state.items():
            group.attrs[key] = value

        # Сохранение полей данных
        for key, value in data.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            group[key] = value

            # Обновление временного файла для мониторинга
            if key in tmp_grp:
                tmp_grp[key][:] = value
            else:
                tmp_grp[key] = value
            tmp_grp[key].flush()

        # Обновление базовых атрибутов
        for key in ("step", "time", "dt"):
            tmp_grp[key][:] = np.array([state[key]])
            tmp_grp[key].flush()

        # Сохранение running_state (скаляры на каждом шаге)
        if running_state is not None:
            running_grp = group.create_group("running_state")
            for key, value in running_state.items():
                if not isinstance(value, np.ndarray):
                    value = np.asarray(value)
                running_grp[key] = np.squeeze(value)


# ============================================================================
# RUNNING STATE — БУФЕРИЗАЦИЯ СКАЛЯРОВ
# ============================================================================

class RunningState:
    """
    Хранилище для скалярных данных, сохраняемых на каждом шаге.

    Args:
        names_and_sizes: Dict {name: size} — имена и размеры параметров
        buffer_size: Размер буфера перед записью на диск
    """

    def __init__(
        self,
        names_and_sizes: Dict[str, int],
        buffer_size: int,
        array_module=np,
    ):
        self.step = 0
        self.array_module = array_module
        self.buffer_size = buffer_size
        self.names_and_sizes = names_and_sizes
        self.values = {
            name: array_module.zeros((size, buffer_size))
            for name, size in self.names_and_sizes.items()
        }

    def clear(self) -> None:
        """Очищает буфер."""
        self.step = 0
        for name, size in self.names_and_sizes.items():
            self.values[name] = self.array_module.zeros((size, self.buffer_size))

    def append(self, name: str, value: Sequence[float]) -> None:
        """Добавляет данные в буфер."""
        self.values[name][:, self.step] = value


# ============================================================================
# TDGL RUNNER — ОСНОВНОЙ КЛАСС
# ============================================================================

class TDGLRunner:
    """
    Запускает TDGL симуляцию с учётом термализации, сохранения и адаптивного шага.

    Args:
        solver: Экземпляр TDGLSolver
        options: Параметры симуляции
    """

    def __init__(self, solver: TDGLSolver, options: SolverOptions):
        self.solver = solver
        self.options = options

        # Состояние симуляции
        self.t = 0.0
        self.dt = options.dt_init
        self.step = 0

        # FVM integrator для проверок
        if options.check_conservation or options.check_energy:
            self.fvm = FVMIntegrator(mesh=solver.device.mesh)
        else:
            self.fvm = None

        # === RUNNING STATE для скаляров ===
        running_names_and_sizes = {"dt": 1}

        if solver.device.probe_points is not None:
            running_names_and_sizes["mu_probe"] = len(solver.device.probe_points)
            running_names_and_sizes["theta_probe"] = len(solver.device.probe_points)

        if options.check_energy:
            running_names_and_sizes["energy_voronoi"] = 1
            running_names_and_sizes["energy_triangles"] = 1

        if options.check_conservation:
            running_names_and_sizes["global_surface_flux_edges"] = 1
            running_names_and_sizes["global_surface_flux_sites"] = 1
            running_names_and_sizes["global_div_voronoi"] = 1
            running_names_and_sizes["global_div_triangles"] = 1

        if options.check_residual:
            running_names_and_sizes["poisson_residual_max"] = 1
            running_names_and_sizes["poisson_residual_mean"] = 1

        self.running_state = RunningState(
            running_names_and_sizes,
            buffer_size=options.save_every,
        )

    def run(
            self,
            psi_init: Optional[np.ndarray] = None,
            mu_init: Optional[np.ndarray] = None,
            seed_solution: Optional[str] = None,
    ) -> Dict[str, Union[np.ndarray, List]]:
        """Запустить полную симуляцию."""
        start_time = time.perf_counter()

        # Загрузка начальных условий
        if seed_solution is not None:
            psi, mu = self._load_seed_solution(seed_solution)
        else:
            psi = psi_init if psi_init is not None else np.ones(self.solver.n_sites, dtype=np.complex128)
            mu = mu_init if mu_init is not None else np.zeros(self.solver.n_sites, dtype=np.float64)

        # === ЗАПУСК С DATAHANDLER ===
        with DataHandler(output_file=self.options.output_file, logger=logger) as data_handler:
            # Сохранение сетки
            data_handler.save_mesh(self.solver.device.mesh)

            # Сохранение фиксированных значений
            fixed_data = {
                "A_for_constant_Bz": self.solver.A_for_constant_Bz,
            }
            data_handler.save_fixed_values(fixed_data)

            # === 1. ТЕРМАЛИЗАЦИЯ ===
            if self.options.skip_time > 0:
                logger.info(f"Термализация: t ∈ [0, {self.options.skip_time}]")
                success = self._run_stage(
                    psi=psi,
                    mu=mu,
                    end_time=self.options.skip_time,
                    save=False,
                    desc="Термализация",
                    data_handler=data_handler,
                )

                if not success:
                    logger.warning("Термализация отменена пользователем")
                    return {"output_path": data_handler.output_path, "cancelled": True}

                # Сброс истории после термализации
                self.running_state.clear()
                self.t = 0.0
                self.step = 0

            # === 2. ОСНОВНАЯ СИМУЛЯЦИЯ ===
            logger.info(f"Симуляция: t ∈ [0, {self.options.solve_time}]")
            success = self._run_stage(
                psi=psi,
                mu=mu,
                end_time=self.options.solve_time,
                save=True,
                desc="Симуляция",
                data_handler=data_handler,
            )

            elapsed = time.perf_counter() - start_time

            # === ФИНАЛЬНАЯ СТАТИСТИКА ===
            print("\n" + "=" * 60)
            print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
            print("=" * 60)
            print(f"Общее время: {elapsed:.2f} с")
            print(f"Всего шагов: {self.step}")
            print(f"Среднее время на шаг: {elapsed / self.step * 1000:.2f} мс")
            print(f"Средняя скорость: {self.step / elapsed:.2f} шагов/с")
            print(f"Статус: {'✅ Успешно' if success else '❌ Отменено'}")
            print("=" * 60)

            return {
                "output_path": data_handler.output_path,
                "final_step": self.step,
                "final_time": self.t,
                "cancelled": not success,
            }

    def _load_seed_solution(self, seed_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загружает начальные условия из HDF5 файла."""
        logger.info(f"Загрузка начальных условий из {seed_path}")
        with h5py.File(seed_path, "r") as f:
            # Находим последний шаг
            data_group = f["data"]
            steps = [int(key) for key in data_group.keys()]
            last_step = str(max(steps))

            psi = np.array(data_group[last_step]["psi"])
            mu = np.array(data_group[last_step]["mu"])

        logger.info(f"Загружено: psi shape={psi.shape}, mu shape={mu.shape}")
        return psi, mu

    # runner.py
    def _run_stage(
            self,
            psi: np.ndarray,
            mu: np.ndarray,
            end_time: float,
            save: bool,
            desc: str,
            data_handler: DataHandler,
    ) -> bool:
        """Запустить один этап симуляции."""
        psi_abs_sq = np.abs(psi) ** 2

        # === НАСТРОЙКА ПРОГРЕСС-БАРА ===
        prog_disabled = (
                self.options.progress_interval is not None
                and self.options.progress_interval > 0
        )
        bar_format = "{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]"

        cancelled = False
        save_counter = 0
        now = None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=TqdmWarning)
            with tqdm(
                    initial=self.t,
                    total=end_time,
                    desc=desc,
                    disable=prog_disabled,
                    unit="τ₀",
                    bar_format=bar_format,
                    dynamic_ncols=True,
            ) as pbar:

                for i in itertools.count():
                    try:
                        iter_start = time.perf_counter()

                        # === ОДИН ШАГ СИМУЛЯЦИИ ===
                        result: StepResult = self.solver.solve_for_one_step(
                            psi=psi,
                            psi_abs_sq=psi_abs_sq,
                            mu=mu,
                            t=self.t,
                            dt=self.dt,  # ← Передаём текущий dt
                        )

                        # Распаковка результата
                        psi = result.psi
                        psi_abs_sq = result.psi_abs_sq
                        supercurrent_x = result.supercurrent_x
                        supercurrent_y = result.supercurrent_y
                        div_Js = result.div_Js
                        mu = result.mu
                        normal_current = result.normal_current

                        # === ВАЖНО: Обновляем dt из результата! ===
                        self.dt = result.dt

                        iter_end = time.perf_counter()
                        iter_time = iter_end - iter_start

                        # Обновление времени и шага
                        self.t += self.dt
                        self.step += 1
                        save_counter += 1

                        # Обновление прогресс-бара
                        if self.t < end_time:
                            pbar.update(self.dt)
                        else:
                            pbar.update(end_time - (self.t - self.dt))

                        pbar.set_postfix({
                            'dt': f'{self.dt:.2e}',
                            'iter': f'{iter_time * 1000:.1f}ms',
                        })

                        # === СОХРАНЕНИЕ СКАЛЯРОВ ===
                        self._update_running_state(
                            psi=psi, mu=mu,
                            supercurrent_x=supercurrent_x,
                            supercurrent_y=supercurrent_y,
                            div_Js=div_Js,
                            result=result,
                        )

                        # === СОХРАНЕНИЕ ПОЛЕЙ ===
                        if save and (save_counter % self.options.save_every == 0):
                            state = {
                                "step": self.step,
                                "time": self.t,
                                "dt": self.dt,
                            }
                            data = {
                                "psi": psi,
                                "mu": mu,
                                "supercurrent_x": supercurrent_x,
                                "supercurrent_y": supercurrent_y,
                                "div_Js": div_Js,
                                "normal_current": normal_current,
                            }
                            running_state = self.running_state.values if self.running_state.step > 0 else None
                            data_handler.save_time_step(state, data, running_state)
                            self.running_state.clear()

                        # Ручной прогресс
                        if prog_disabled and (i % self.options.progress_interval == 0):
                            then, now = now, time.perf_counter()
                            speed = self.options.progress_interval / (now - then) if then else 0
                            logger.info(
                                f"{desc}: Time {self.t:.2f}/{end_time:.2f}, "
                                f"dt={self.dt:.2e}, {speed:.2f} it/s"
                            )

                        # Проверка окончания
                        if self.t >= end_time:
                            break

                    except KeyboardInterrupt:
                        msg = f"{{}} simulation at step {self.step} of stage {desc!r}."
                        if self.options.pause_on_interrupt:
                            response = input(
                                f"\nSimulation paused at stage {desc!r} (step {self.step}). "
                                "Continue simulation? [yN] "
                            )
                            resume = response.lower().startswith('y')
                            if resume:
                                logger.info(msg.format("Resuming"))
                            else:
                                logger.warning(msg.format("Cancelling"))
                                cancelled = True
                                break
                        else:
                            logger.warning(msg.format("Cancelling"))
                            cancelled = True
                            break

        # Сохранить последний шаг
        if save and (save_counter % self.options.save_every):
            state = {"step": self.step, "time": self.t, "dt": self.dt}
            data = {
                "psi": psi, "mu": mu,
                "supercurrent_x": supercurrent_x,
                "supercurrent_y": supercurrent_y,
                "div_Js": div_Js,
                "normal_current": normal_current,
            }
            running_state = self.running_state.values if self.running_state.step > 0 else None
            data_handler.save_time_step(state, data, running_state)
            self.running_state.clear()

        return not cancelled

    def _update_running_state(
        self,
        psi: np.ndarray,
        mu: np.ndarray,
        supercurrent_x: np.ndarray,
        supercurrent_y: np.ndarray,
        div_Js: np.ndarray,
        result: StepResult,
    ) -> None:
        """Обновляет running_state скалярами."""
        # Базовые скаляры
        self.running_state.append("dt", [self.dt])

        # Probe points
        if self.solver.device.probe_points is not None:
            probe_indices = [
                self.solver.device.closest_site(xy)
                for xy in self.solver.device.probe_points
            ]
            self.running_state.append("mu_probe", mu[probe_indices])
            self.running_state.append("theta_probe", np.angle(psi[probe_indices]))

        # Энергия
        if self.options.check_energy and result.energy_voronoi is not None:
            self.running_state.append("energy_voronoi", [result.energy_voronoi])
            self.running_state.append("energy_triangles", [result.energy_triangles])

        # Глобальная проверка сохранения
        if self.options.check_conservation and result.conservation_global is not None:
            (surface_flux_edges, surface_flux_sites, div_voronoi, div_triangles) = result.conservation_global

            self.running_state.append("global_surface_flux_edges", [surface_flux_edges])
            self.running_state.append("global_surface_flux_sites", [surface_flux_sites])
            self.running_state.append("global_div_voronoi", [div_voronoi])
            self.running_state.append("global_div_triangles", [div_triangles])

        # Невязка Пуассона
        if self.options.check_residual and result.poisson_residual is not None:
            self.running_state.append("poisson_residual_max", [np.max(np.abs(result.poisson_residual))])
            self.running_state.append("poisson_residual_mean", [np.mean(np.abs(result.poisson_residual))])

        self.running_state.step += 1