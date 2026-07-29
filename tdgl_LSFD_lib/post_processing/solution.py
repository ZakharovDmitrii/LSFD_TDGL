"""
solution.py — Контейнер для результатов TDGL симуляции.
Загружает данные из HDF5 и предоставляет удобный доступ.
"""
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class Solution:
    """
    Контейнер для результатов TDGL симуляции.

    Args:
        path: Путь к HDF5 файлу с результатами
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.path}")

        self._load_metadata()
        self._load_mesh()

        # Текущий шаг (по умолчанию — последний)
        self._current_step = -1
        self._current_data = None

    def _load_metadata(self):
        """Загружает метаданные симуляции."""
        with h5py.File(self.path, 'r') as f:
            self.solve_time = f.attrs.get('solve_time', None)
            self.skip_time = f.attrs.get('skip_time', None)
            self.dt_init = f.attrs.get('dt_init', None)
            self.save_every = f.attrs.get('save_every', 100)

            # Диапазон шагов
            if 'data' in f:
                steps = sorted([int(k) for k in f['data'].keys()])
                self.step_min = min(steps)
                self.step_max = max(steps)
                self.n_saved_steps = len(steps)
            else:
                self.step_min = self.step_max = 0
                self.n_saved_steps = 0

    def _load_mesh(self):
        """Загружает сетку."""
        with h5py.File(self.path, 'r') as f:
            if 'mesh' in f:
                mesh_grp = f['mesh']
                self.sites = np.array(mesh_grp['sites'])
                self.triangles = np.array(mesh_grp['elements'])
                self.voronoi_areas = np.array(mesh_grp['dual_mesh/dual_areas'])
            else:
                self.sites = None
                self.triangles = None

    @property
    def current_step(self) -> int:
        return self._current_step

    @current_step.setter
    def current_step(self, step: int):
        """Переключает текущий шаг и загружает данные."""
        if step < 0:
            step = self.step_max + 1 + step
        if step < self.step_min or step > self.step_max:
            raise ValueError(f"Шаг {step} вне диапазона [{self.step_min}, {self.step_max}]")
        self._current_step = step
        self._load_step_data(step)

    def _load_step_data(self, step: int):
        """Загружает данные для конкретного шага."""
        with h5py.File(self.path, 'r') as f:
            grp = f['data'][str(step)]
            self._current_data = {
                'time': grp.attrs.get('time', 0.0),
                'dt': grp.attrs.get('dt', 0.0),
                'psi': np.array(grp['psi']) if 'psi' in grp else None,
                'mu': np.array(grp['mu']) if 'mu' in grp else None,
                'supercurrent_x': np.array(grp['supercurrent_x']) if 'supercurrent_x' in grp else None,
                'supercurrent_y': np.array(grp['supercurrent_y']) if 'supercurrent_y' in grp else None,
                'div_Js': np.array(grp['div_Js']) if 'div_Js' in grp else None,
                'normal_current': np.array(grp['normal_current']) if 'normal_current' in grp else None,
            }
            # Running state (скаляры)
            if 'running_state' in grp:
                rs = grp['running_state']
                for key in rs.keys():
                    self._current_data[key] = np.array(rs[key])

    def get(self, key: str, step: Optional[int] = None) -> np.ndarray:
        """Получить данные для заданного шага."""
        if step is not None and step != self._current_step:
            self.current_step = step
        if self._current_data is None:
            self.current_step = -1
        return self._current_data.get(key)

    def get_time_series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить временной ряд скалярной величины.

        Returns:
            (times, values) — два массива
        """
        times = []
        values = []
        with h5py.File(self.path, 'r') as f:
            for step in sorted(f['data'].keys(), key=int):
                grp = f['data'][step]
                if 'running_state' in grp and key in grp['running_state']:
                    times.append(grp.attrs['time'])
                    values.append(np.array(grp['running_state'][key]))
        return np.array(times), np.array(values)

    def get_energy_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Получить временной ряд энергии (оба метода)."""
        times, e_vor = self.get_time_series('energy_voronoi')
        _, e_tri = self.get_time_series('energy_triangles')
        return times, e_vor, e_tri

    def get_conservation_series(self):
        """Получить временные ряды проверок сохранения."""
        result = {}
        keys = [
            'global_surface_flux_edges', 'global_surface_flux_sites',
            'global_div_voronoi', 'global_div_triangles',
            'poisson_residual_max', 'poisson_residual_mean',
        ]
        for key in keys:
            times, values = self.get_time_series(key)
            if len(times) > 0:
                result[key] = (times, values)
        return result

    def __repr__(self):
        return (f"Solution(path={self.path.name}, "
                f"steps={self.n_saved_steps}, "
                f"sites={len(self.sites) if self.sites is not None else 0})")