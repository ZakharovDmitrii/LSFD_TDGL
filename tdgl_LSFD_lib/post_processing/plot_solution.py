"""
plot_solution.py — Функции для визуализации результатов TDGL симуляции.
"""
from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

from .solution import Solution


# ============================================================================
# УТИЛИТЫ
# ============================================================================

def _setup_axes(ax=None, figsize=(6, 5), **kwargs):
    """Создаёт или возвращает существующие оси."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')
    return fig, ax



# ============================================================================
# 1. МОДУЛЬ И ФАЗА PSI
# ============================================================================

def plot_order_parameter(
        solution: Solution,
        step: int = -1,
        squared: bool = False,
        mag_cmap: str = 'viridis',
        phase_cmap: str = 'twilight_shifted',
        ax=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Строит модуль |ψ| (или |ψ|²) и фазу arg(ψ).

    Args:
        solution: Объект Solution
        step: Шаг для отображения (-1 = последний)
        squared: Если True, строить |ψ|²
    """
    psi = solution.get('psi', step=step)
    if psi is None:
        raise ValueError(f"Нет данных psi для шага {step}")

    mag = np.abs(psi)
    if squared:
        mag = mag ** 2
        label = r'$|\psi|^2$'
    else:
        label = r'$|\psi|$'
    phase = np.angle(psi) / np.pi  # в единицах π

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tri = Triangulation(solution.sites[:, 0], solution.sites[:, 1], solution.triangles)

    # Модуль
    im0 = axes[0].tripcolor(tri, mag, cmap=mag_cmap, vmin=0, vmax=1, shading='gouraud')
    axes[0].set_title(f'{label} (step {solution.current_step})')
    axes[0].set_xlabel('x [ξ]')
    axes[0].set_ylabel('y [ξ]')
    fig.colorbar(im0, ax=axes[0], label=label)

    # Фаза
    im1 = axes[1].tripcolor(tri, phase, cmap=phase_cmap, vmin=-1, vmax=1, shading='gouraud')
    axes[1].set_title(r'Phase $\theta/\pi$')
    axes[1].set_xlabel('x [ξ]')
    axes[1].set_ylabel('y [ξ]')
    fig.colorbar(im1, ax=axes[1], label=r'$\theta/\pi$')

    for ax in axes:
        ax.set_aspect('equal')

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 2. MU И DIV(Js)
# ============================================================================

def plot_scalar_potential_and_divergence(
        solution: Solution,
        step: int = -1,
        ax=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Строит скалярный потенциал μ и дивергенцию сверхтока."""
    mu = solution.get('mu', step=step)
    div_J = solution.get('div_Js', step=step)

    if mu is None or div_J is None:
        raise ValueError(f"Нет данных mu/div_Js для шага {step}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tri = Triangulation(solution.sites[:, 0], solution.sites[:, 1], solution.triangles)

    # μ
    im0 = axes[0].tripcolor(tri, mu, cmap='magma', shading='gouraud')
    axes[0].set_title(r'Scalar potential $\mu$')
    axes[0].set_xlabel('x [ξ]')
    axes[0].set_ylabel('y [ξ]')
    fig.colorbar(im0, ax=axes[0], label=r'$\mu$')

    # div(Js)
    im1 = axes[1].tripcolor(tri, div_J, cmap='RdBu_r', shading='gouraud',
                            vmin=-np.max(np.abs(div_J)), vmax=np.max(np.abs(div_J)))
    axes[1].set_title(r'$\nabla \cdot J_s$')
    axes[1].set_xlabel('x [ξ]')
    axes[1].set_ylabel('y [ξ]')
    fig.colorbar(im1, ax=axes[1], label=r'$\nabla \cdot J_s$')

    for ax in axes:
        ax.set_aspect('equal')

    fig.tight_layout()
    return fig, axes


# ============================================================================
# 3-4. ТОКИ С ЛИНИЯМИ ТОКА
# ============================================================================
def _interpolate_to_grid(x, y, values, grid_shape=200, method='linear'):
    """Интерполирует данные на регулярную сетку для streamplot."""
    xi = np.linspace(x.min(), x.max(), grid_shape)
    yi = np.linspace(y.min(), y.max(), grid_shape)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: fill_value=np.nan
    values_grid = griddata(
        np.column_stack([x, y]), values,
        (xi_grid, yi_grid), method=method, fill_value=np.nan
    )
    return xi_grid, yi_grid, values_grid


def plot_currents(
        solution: Solution,
        dataset: str = 'supercurrent',
        step: int = -1,
        grid_shape: int = 200,
        streamplot: bool = True,
        cmap: str = 'inferno',
        min_stream_amp: float = 0.025,  # ← Добавлено как в pyTDGL
        ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Строит модуль тока с линиями тока.
    """
    if dataset == 'supercurrent':
        Jx = solution.get('supercurrent_x', step=step)
        Jy = solution.get('supercurrent_y', step=step)
        label = r'$|J_s|$'
    elif dataset == 'normal_current':
        nc = solution.get('normal_current', step=step)
        if nc is None:
            raise ValueError("Нет данных normal_current")
        Jx = nc[:, 0]
        Jy = nc[:, 1]
        label = r'$|J_n|$'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if Jx is None or Jy is None:
        raise ValueError(f"Нет данных тока для шага {step}")

    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    fig, ax = _setup_axes(ax, figsize=(7, 6))
    x, y = solution.sites[:, 0], solution.sites[:, 1]
    tri = Triangulation(x, y, solution.triangles)

    # Основной plot
    im = ax.tripcolor(tri, J_mag, cmap=cmap, shading='gouraud')
    fig.colorbar(im, ax=ax, label=label)

    # Линии тока
    if streamplot:
        # Используем 'linear' для стабильности
        xgrid, ygrid, Jx_grid = _interpolate_to_grid(x, y, Jx, grid_shape, method='linear')
        _, _, Jy_grid = _interpolate_to_grid(x, y, Jy, grid_shape, method='linear')
        _, _, J_mag_grid = _interpolate_to_grid(x, y, J_mag, grid_shape, method='linear')

        # Маскируем слабые токи (как в pyTDGL)
        if min_stream_amp is not None:
            cutoff = np.nanmax(J_mag_grid) * min_stream_amp
            mask = J_mag_grid < cutoff
            Jx_grid[mask] = np.nan
            Jy_grid[mask] = np.nan

        # Отладочный вывод
        print(f"Streamplot debug:")
        print(f"  xgrid shape: {xgrid.shape}")
        print(f"  Jx_grid NaN count: {np.sum(np.isnan(Jx_grid))} / {Jx_grid.size}")
        print(f"  Jx_grid range: [{np.nanmin(Jx_grid):.3e}, {np.nanmax(Jx_grid):.3e}]")

        ax.streamplot(xgrid, ygrid, Jx_grid, Jy_grid,
                      color='w', density=1.0, linewidth=0.75)

    ax.set_title(f'{label} (step {solution.current_step})')
    ax.set_xlabel('x [ξ]')
    ax.set_ylabel('y [ξ]')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 5. ЭНЕРГИЯ ОТ ВРЕМЕНИ
# ============================================================================

def plot_energy_vs_time(
        solution: Solution,
        ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Строит энергию GL от времени (оба метода на одном графике)."""
    times, e_vor, e_tri = solution.get_energy_series()

    if len(times) == 0:
        raise ValueError("Нет данных энергии. Проверьте check_energy=True в опциях.")

    fig, ax = _setup_axes(ax, figsize=(8, 4))

    ax.plot(times, e_vor, 'b-', label='Voronoi', linewidth=1.5)
    ax.plot(times, e_tri, 'r--', label='Triangles', linewidth=1.5)

    ax.set_xlabel('Time [τ₀]')
    ax.set_ylabel('GL Energy')
    ax.set_title('Ginzburg-Landau Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Показываем разницу между методами
    ax2 = ax.twinx()
    diff = np.abs(e_vor - e_tri)
    ax2.plot(times, diff, 'g:', alpha=0.6, label='|E_vor - E_tri|')
    ax2.set_ylabel('|ΔE|', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 6. ПРОВЕРКИ СОХРАНЕНИЯ ОТ ВРЕМЕНИ
# ============================================================================

def plot_conservation_checks(
        solution: Solution,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Строит проверки сохранения от времени.

    Два графика:
    - Интегральные: surface flux vs div integral
    - Поточные: residual Пуассона
    """
    cons = solution.get_conservation_series()

    if not cons:
        raise ValueError("Нет данных conservation checks. Проверьте check_conservation=True.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # === Левый: интегральные ===
    ax = axes[0]
    if 'global_surface_flux_edges' in cons:
        t, v = cons['global_surface_flux_edges']
        ax.plot(t, v, label='Surface flux (edges)', linewidth=1.5)
    if 'global_surface_flux_sites' in cons:
        t, v = cons['global_surface_flux_sites']
        ax.plot(t, v, label='Surface flux (sites)', linewidth=1.5)
    if 'global_div_voronoi' in cons:
        t, v = cons['global_div_voronoi']
        ax.plot(t, v, label='∫div(J) Voronoi', linewidth=1.5, linestyle='--')
    if 'global_div_triangles' in cons:
        t, v = cons['global_div_triangles']
        ax.plot(t, v, label='∫div(J) Triangles', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Time [τ₀]')
    ax.set_ylabel('Integral value')
    ax.set_title('Global Conservation: ∮J·dl vs ∫div(J)dS')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Правый: невязка Пуассона ===
    ax = axes[1]
    if 'poisson_residual_max' in cons:
        t, v = cons['poisson_residual_max']
        ax.semilogy(t, v, label='Max |residual|', color='r', linewidth=1.5)
    if 'poisson_residual_mean' in cons:
        t, v = cons['poisson_residual_mean']
        ax.semilogy(t, v, label='Mean |residual|', color='b', linewidth=1.5)

    ax.set_xlabel('Time [τ₀]')
    ax.set_ylabel('Residual')
    ax.set_title('Poisson Equation Residual')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    return fig, axes


# ============================================================================
# УДОБНАЯ ФУНКЦИЯ ДЛЯ БЫСТРОГО ОБЗОРА
# ============================================================================
def plot_summary(
        solution: Solution,
        step: int = -1,
        save_path: Optional[str] = None,
):
    """
    Строит все основные графики сразу.

    Args:
        solution: Объект Solution
        step: Шаг для отображения
        save_path: Если задан, сохраняет в файл
    """
    fig = plt.figure(figsize=(18, 14))

    # 1. Order parameter
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    psi = solution.get('psi', step=step)
    tri = Triangulation(solution.sites[:, 0], solution.sites[:, 1], solution.triangles)

    im1 = ax1.tripcolor(tri, np.abs(psi), cmap='viridis', vmin=0, vmax=1, shading='gouraud')
    ax1.set_title(r'$|\psi|$')
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.tripcolor(tri, np.angle(psi) / np.pi, cmap='twilight_shifted',
                        vmin=-1, vmax=1, shading='gouraud')
    ax2.set_title(r'Phase $\theta/\pi$')
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2)

    # 2. mu
    ax3 = fig.add_subplot(3, 3, 3)
    mu = solution.get('mu', step=step)
    mu_plot = mu - np.min(mu)
    im3 = ax3.tripcolor(tri, mu_plot, cmap='magma', shading='gouraud')
    ax3.set_title(r'$\mu$')
    ax3.set_aspect('equal')
    fig.colorbar(im3, ax=ax3)

    # 3. Supercurrent С ЛИНИЯМИ ТОКА (используем plot_currents)
    ax4 = fig.add_subplot(3, 3, 4)
    from .plot_solution import plot_currents
    plot_currents(
        solution=solution,
        ax=ax4,
        dataset='supercurrent',
        step=step,
        streamplot=True,
        min_stream_amp=0.025,
        cmap='inferno',
    )

    # 4. Normal current С ЛИНИЯМИ ТОКА
    ax5 = fig.add_subplot(3, 3, 5)
    plot_currents(
        solution=solution,
        ax=ax5,
        dataset='normal_current',
        step=step,
        streamplot=True,
        min_stream_amp=0.025,
        cmap='plasma',
    )

    # 5. div(Js)
    ax6 = fig.add_subplot(3, 3, 6)
    div_J = solution.get('div_Js', step=step)
    div_J = np.real(div_J)  # Гарантируем вещественный тип
    #vmax = np.max(np.abs(div_J))
    #vmin = np.min(np.abs(div_J))
    div_J_plot = div_J - np.min(div_J)
    im6 = ax6.tripcolor(tri, div_J_plot, cmap='RdBu_r', shading='gouraud')
    ax6.set_title(r'$\nabla \cdot J_s$')
    ax6.set_aspect('equal')
    fig.colorbar(im6, ax=ax6)

    # 6. Energy vs time
    ax7 = fig.add_subplot(3, 3, 7)
    try:
        times, e_vor, e_tri = solution.get_energy_series()
        if len(times) > 0:
            ax7.plot(times, e_vor, 'b-', label='Voronoi', linewidth=1.5)
            ax7.plot(times, e_tri, 'r--', label='Triangles', linewidth=1.5)
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Energy')
            ax7.set_title('GL Energy')
            #ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)
    except Exception as e:
        ax7.text(0.5, 0.5, f'No energy data\n{e}', ha='center', va='center',
                 transform=ax7.transAxes)

    # 7. Conservation checks
    ax8 = fig.add_subplot(3, 3, 8)
    try:
        cons = solution.get_conservation_series()
        if cons:
            if 'global_surface_flux_edges' in cons:
                t, v = cons['global_surface_flux_edges']
                ax8.plot(t, v, label='J·dl (edges)', linewidth=1.5)
            if 'global_div_voronoi' in cons:
                t, v = cons['global_div_voronoi']
                ax8.plot(t, v, label='∫div(J) (Voronoi)', linewidth=1.5, linestyle='--')
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Integral')
            ax8.set_title('Conservation Check')
            #ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
    except Exception as e:
        ax8.text(0.5, 0.5, f'No conservation data\n{e}', ha='center', va='center',
                 transform=ax8.transAxes)

    # 8. Poisson residual
    ax9 = fig.add_subplot(3, 3, 9)
    try:
        cons = solution.get_conservation_series()
        if cons and 'poisson_residual_max' in cons:
            t, v = cons['poisson_residual_max']
            ax9.semilogy(t, v, 'r-', label='Max residual', linewidth=1.5)
            if 'poisson_residual_mean' in cons:
                t, v = cons['poisson_residual_mean']
                ax9.semilogy(t, v, 'b-', label='Mean residual', linewidth=1.5)
            ax9.set_xlabel('Time')
            ax9.set_ylabel('Residual')
            ax9.set_title('Poisson Solver Convergence')
            #ax9.legend(fontsize=8)
            ax9.grid(True, alpha=0.3, which='both')
    except Exception as e:
        ax9.text(0.5, 0.5, f'No residual data\n{e}', ha='center', va='center',
                 transform=ax9.transAxes)

    fig.suptitle(f'TDGL Solution — Step {solution.current_step}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Сохранено в {save_path}")

    return fig