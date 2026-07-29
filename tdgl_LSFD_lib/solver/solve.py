"""
solve.py — Фасад для запуска TDGL симуляции.
"""
from typing import Optional
import numpy as np

from ..device.device import Device
from ..external_fields.external_fields import ExternalFields
from ..operators.operators import LSFD_operators
from ..solver.solver import TDGLSolver
from ..solver.runner import TDGLRunner
from ..solver.dynamics_options import SolverOptions
from ..post_processing.solution import Solution


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
    Запустить TDGL симуляцию.

    Args:
        device: Устройство
        operators: LSFD операторы
        external_fields: Внешние поля
        options: Параметры симуляции
        psi_init: Начальное psi (по умолчанию = 1)
        mu_init: Начальное mu (по умолчанию = 0)
        seed_solution: Путь к HDF5 файлу для продолжения

    Returns:
        Solution — объект с результатами
    """
    # Создаём солвер
    solver = TDGLSolver(
        device=device,
        operators=operators,
        external_fields=external_fields,
        options=options,
    )

    # Начальные условия
    if psi_init is None:
        psi_init = np.ones(device.n_sites, dtype=np.complex128)
    if mu_init is None:
        mu_init = np.zeros(device.n_sites, dtype=np.float64)

    # Запускаем Runner
    runner = TDGLRunner(solver=solver, options=options)
    result_info = runner.run(
        psi_init=psi_init,
        mu_init=mu_init,
        seed_solution=seed_solution,
    )

    # Возвращаем Solution
    return Solution(path=result_info['output_path'])