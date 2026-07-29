"""
dynamics_options.py — Параметры TDGL симуляции для LSFD-версии.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union
from pathlib import Path


class TimeScheme(Enum):
    """Схема интегрирования по времени."""
    EULER: str = "euler"  # Явный Эйлер с фиксированным шагом
    ADAPTIVE_EULER: str = "adaptive_euler"  # Адаптивный Эйлер
    CRANK_NICOLSON: str = "crank_nicolson"  # Крэнк-Николсон (пока не реализован)


@dataclass
class SolverOptions:
    """Параметры TDGL солвера для LSFD-версии."""

    # === Время ===
    solve_time: float
    skip_time: float = 0.0

    # === Шаг по времени ===
    dt_init: float = 1e-6  # Начальный шаг
    dt_max: float = 1e-2  # Максимальный шаг (для адаптивной схемы)
    dt_min: float = 1e-8  # Минимальный шаг (защита от нуля)
    adaptive_window: int = 10  # Окно усреднения для адаптивного шага

    # === Временная схема ===
    time_scheme: Union[TimeScheme, str] = TimeScheme.EULER

    # === Numba ===
    use_numba: bool = False
    num_threads: int = 4

    # === Сохранение ===
    output_file: Union[str, Path, None] = None
    save_every: int = 100
    progress_interval: int = 0  # 0 = использовать tqdm
    pause_on_interrupt: bool = True

    # === LSFD специфичные ===
    update_G_angle_threshold: float = 0.05  # рад (~2.9°)

    # === Диагностика ===
    check_residual: bool = False
    check_energy: bool = False
    check_conservation: bool = False

    def validate(self) -> None:
        """Проверить корректность параметров."""
        if self.solve_time <= 0:
            raise ValueError(f"solve_time должен быть > 0 (got {self.solve_time})")
        if self.skip_time < 0:
            raise ValueError(f"skip_time должен быть >= 0 (got {self.skip_time})")
        if self.dt_init <= 0:
            raise ValueError(f"dt_init должен быть > 0 (got {self.dt_init})")
        if self.dt_max <= 0:
            raise ValueError(f"dt_max должен быть > 0 (got {self.dt_max})")
        if self.dt_min <= 0:
            raise ValueError(f"dt_min должен быть > 0 (got {self.dt_min})")
        if self.dt_init > self.dt_max:
            raise ValueError(f"dt_init ({self.dt_init}) должен быть <= dt_max ({self.dt_max})")
        if self.dt_min > self.dt_init:
            raise ValueError(f"dt_min ({self.dt_min}) должен быть <= dt_init ({self.dt_init})")
        if self.adaptive_window < 1:
            raise ValueError(f"adaptive_window должен быть >= 1 (got {self.adaptive_window})")

        # Временная схема
        if isinstance(self.time_scheme, str):
            try:
                self.time_scheme = TimeScheme[self.time_scheme.upper()]
            except KeyError:
                valid = list(TimeScheme.__members__.keys())
                raise ValueError(f"time_scheme должен быть одним из {valid!r}, got {self.time_scheme}")

        # Проверка: для EULER dt_init должен равняться dt_max
        if self.time_scheme == TimeScheme.EULER and self.dt_init != self.dt_max:
            import warnings
            warnings.warn(
                f"time_scheme=EULER, но dt_init ({self.dt_init}) != dt_max ({self.dt_max}). "
                f"Для фиксированного шага используйте dt_init = dt_max.",
                UserWarning
            )

    def __post_init__(self):
        self.validate()
