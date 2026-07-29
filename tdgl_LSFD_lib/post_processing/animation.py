"""
animation.py — Создание анимаций из результатов TDGL симуляции.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import numpy as np


def make_video_from_solution(
    solution,
    fig_name: str,
    file_dir: str,
    quantities: Union[str, List[str]] = "order_parameter",
    fps: int = 10,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> str:
    """
    Создаёт GIF-анимацию изменения параметра порядка во времени.

    Args:
        solution: Объект Solution с результатами симуляции
        fig_name: Имя файла (без расширения)
        file_dir: Директория для сохранения
        quantities: Что анимировать:
            - "order_parameter" или "psi" — модуль |ψ|
            - "phase" — фаза arg(ψ)
            - "scalar_potential" — μ
        fps: Кадров в секунду
        figsize: Размер фигуры (ширина, высота)
        dpi: Разрешение
        cmap: Colormap
        vmin, vmax: Диапазон цветовой шкалы

    Returns:
        Путь к созданному GIF файлу
    """
    if isinstance(quantities, str):
        quantities = [quantities]

    # Создаём директорию если нужно
    Path(file_dir).mkdir(parents=True, exist_ok=True)

    # Путь к выходному файлу
    output_path = os.path.join(file_dir, f"{fig_name}.gif")

    # Загружаем данные напрямую из HDF5 файла
    with h5py.File(solution.path, "r") as f:
        # Получаем список всех шагов
        data_group = f["data"]
        steps = sorted([int(k) for k in data_group.keys()])
        n_frames = len(steps)

        if n_frames < 2:
            print(f"⚠️ Недостаточно кадров для анимации (n_frames={n_frames})")
            return None

        # Загружаем сетку
        sites = np.array(f["mesh"]["sites"])
        triangles = np.array(f["mesh"]["elements"])

        x = sites[:, 0]
        y = sites[:, 1]

        # Создаём триангуляцию
        triangulation = mtri.Triangulation(x, y, triangles)

    # Создаём фигуру и оси
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal')
    ax.set_xlabel('x [ξ]')
    ax.set_ylabel('y [ξ]')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    # Заголовок с временем
    title = ax.set_title('')

    # Цветовая шкала
    cbar_label = {
        'order_parameter': r'$|\psi|$',
        'psi': r'$|\psi|$',
        'phase': r'$\theta/\pi$',
        'scalar_potential': r'$\mu$',
    }

    # Инициализация (пустой plot)
    im = ax.tripcolor(triangulation, np.zeros(len(x)),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label.get(quantities[0], quantities[0]))

    def update(frame_idx):
        """Обновление кадра."""
        step = steps[frame_idx]

        with h5py.File(solution.path, "r") as f:
            data_grp = f["data"][str(step)]

            # Загружаем данные
            if quantities[0] in ['order_parameter', 'psi']:
                psi = np.array(data_grp["psi"])
                data = np.abs(psi)
            elif quantities[0] == 'phase':
                psi = np.array(data_grp["psi"])
                data = np.angle(psi) / np.pi
            elif quantities[0] == 'scalar_potential':
                data = np.array(data_grp["mu"])
            else:
                raise ValueError(f"Неизвестная величина: {quantities[0]}")

            # Получаем время
            time = float(data_grp.attrs.get('time', frame_idx))

        # Обновляем данные
        im.set_array(data)

        # Обновляем заголовок
        title.set_text(f'{cbar_label.get(quantities[0], quantities[0])} at t = {time:.3f} τ₀')

        return [im]

    # Создаём анимацию
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000//fps, blit=True
    )

    # Сохраняем как GIF
    print(f"🎬 Создание анимации ({n_frames} кадров, {fps} fps)...")
    anim.save(output_path, writer='pillow', fps=fps)

    plt.close(fig)

    print(f"✅ Анимация сохранена: {output_path}")
    return output_path


def create_animation(
    solution,
    output_path: str,
    quantities: Union[str, List[str]] = "order_parameter",
    fps: int = 10,
    figsize: Tuple[float, float] = (10, 8),
) -> str:
    """
    Универсальная функция для создания анимации.

    Args:
        solution: Объект Solution
        output_path: Путь к выходному файлу (с расширением .gif)
        quantities: Что анимировать
        fps: Кадров в секунду
        figsize: Размер фигуры

    Returns:
        Путь к созданному файлу
    """
    file_dir = os.path.dirname(output_path) or "."
    fig_name = Path(output_path).stem

    return make_video_from_solution(
        solution, fig_name, file_dir,
        quantities=quantities,
        fps=fps, figsize=figsize
    )