"""
solve.py — Фасад для запуска TDGL симуляции.
"""
from typing import Optional
import numpy as np
import logging

from ..device.device import Device
from ..external_fields.external_fields import ExternalFields
from ..operators.operators import LSFD_operators
from .solver import TDGLSolver
from .runner import TDGLRunner
from .dynamics_options import SolverOptions, RunMode  # ← Добавлен импорт RunMode
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
    """
    # 1. Настраиваем логирование ПЕРЕД созданием любых объектов
    _setup_logging(options)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting TDGL simulation: solve_time={options.solve_time}, run_mode={options.run_mode.value}")

    # 2. Создаём солвер
    solver = TDGLSolver(
        device=device,
        operators=operators,
        external_fields=external_fields,
        options=options,
    )

    # 3. Начальные условия
    if psi_init is None:
        psi_init = np.ones(device.n_sites, dtype=np.complex128)
    if mu_init is None:
        mu_init = np.zeros(device.n_sites, dtype=np.float64)

    # 4. Запускаем Runner
    runner = TDGLRunner(solver=solver, options=options)
    result_info = runner.run(
        psi_init=psi_init,
        mu_init=mu_init,
        seed_solution=seed_solution,
    )

    # 5. Возвращаем Solution
    return Solution(path=result_info['output_path'])


def _setup_logging(options: SolverOptions) -> None:
    """Настраивает logging в зависимости от опций."""
    # Формат сообщений
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # Маппинг строковых уровней в константы logging
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Определяем уровень: используем явный log_level из options
    level = level_map.get(options.log_level.upper(), logging.INFO)

    # Создаём handler'ы
    handlers = [logging.StreamHandler()]  # Всегда пишем в консоль

    # Если указан файл — добавляем файловый handler
    if options.log_file is not None:
        handlers.append(logging.FileHandler(str(options.log_file), mode='w'))

    # Настраиваем корневой логгер библиотеки (не root, чтобы не ломать другие библиотеки)
    root_logger = logging.getLogger('tdgl_LSFD_lib')
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Очищаем старые handler'ы на случай повторных запусков

    formatter = logging.Formatter(fmt, datefmt=datefmt)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Запрещаем всплытие к корневому логгеру Python, чтобы сообщения не дублировались
    root_logger.propagate = False