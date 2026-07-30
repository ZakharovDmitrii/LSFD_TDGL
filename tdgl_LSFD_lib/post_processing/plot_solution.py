"""
plot_solution.py — Visualization functions for TDGL simulation results.

Provides functions for plotting:
    - Spatial fields (psi, mu, currents) at specific time steps
    - Time series from trackers (energy, conservation, probe points)
    - Summary plots for quick overview
"""
from typing import Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

from .solution import Solution


# ============================================================================
# UTILITIES
# ============================================================================

def _setup_axes(ax=None, figsize=(6, 5), **kwargs):
    """Create or return existing axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
    else:
        fig = ax.get_figure()
    ax.set_aspect('equal')
    return fig, ax


# ============================================================================
# 1. ORDER PARAMETER (MODULUS AND PHASE)
# ============================================================================

def plot_order_parameter(
        solution: Solution,
        step: int = -1,
        squared: bool = False,
        mag_cmap: str = 'viridis',
        phase_cmap: str = 'twilight_shifted',
        figsize: Tuple[float, float] = (12, 5),
        ax=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot modulus |ψ| (or |ψ|²) and phase arg(ψ).

    Args:
        solution: Solution object
        step: Step to display (-1 = last)
        squared: If True, plot |ψ|²
        mag_cmap: Colormap for modulus
        phase_cmap: Colormap for phase
        figsize: Figure size (width, height) in inches
    """
    data = solution.get_spatial_data(step=step)
    psi = data.get('psi')
    if psi is None:
        raise ValueError(f"No psi data for step {step}")

    mag = np.abs(psi)
    if squared:
        mag = mag ** 2
        label = r'$|\psi|^2$'
    else:
        label = r'$|\psi|$'
    phase = np.angle(psi) / np.pi  # in units of π

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    tri = Triangulation(solution.sites[:, 0], solution.sites[:, 1], solution.triangles)

    # Modulus
    im0 = axes[0].tripcolor(tri, mag, cmap=mag_cmap, vmin=0, vmax=1, shading='gouraud')
    axes[0].set_title(f'{label} (step {step})')
    axes[0].set_xlabel('x [ξ]')
    axes[0].set_ylabel('y [ξ]')
    fig.colorbar(im0, ax=axes[0], label=label)

    # Phase
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
# 2. MU AND DIV(Js)
# ============================================================================

def plot_scalar_potential_and_divergence(
        solution: Solution,
        step: int = -1,
        figsize: Tuple[float, float] = (12, 5),
        ax=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot scalar potential μ and supercurrent divergence."""
    data = solution.get_spatial_data(step=step)
    mu = data.get('mu')
    div_J = data.get('div_Js')

    if mu is None or div_J is None:
        raise ValueError(f"No mu/div_Js data for step {step}")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
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
# 3-4. CURRENTS WITH STREAMLINES
# ============================================================================

def _interpolate_to_grid(x, y, values, grid_shape=200, method='linear'):
    """Interpolate data to regular grid for streamplot."""
    xi = np.linspace(x.min(), x.max(), grid_shape)
    yi = np.linspace(y.min(), y.max(), grid_shape)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

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
        min_stream_amp: float = 0.025,
        figsize: Tuple[float, float] = (7, 6),
        ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot current modulus with streamlines.

    Args:
        solution: Solution object
        dataset: 'supercurrent' or 'normal_current'
        step: Step to display (-1 = last)
        grid_shape: Grid resolution for streamplot interpolation
        streamplot: Whether to draw streamlines
        cmap: Colormap
        min_stream_amp: Minimum amplitude for streamlines (relative to max)
        figsize: Figure size (width, height) in inches
    """
    data = solution.get_spatial_data(step=step)

    if dataset == 'supercurrent':
        Jx = data.get('supercurrent_x')
        Jy = data.get('supercurrent_y')
        label = r'$|J_s|$'
    elif dataset == 'normal_current':
        nc = data.get('normal_current')
        if nc is None:
            raise ValueError("No normal_current data")
        Jx = nc[:, 0]
        Jy = nc[:, 1]
        label = r'$|J_n|$'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if Jx is None or Jy is None:
        raise ValueError(f"No current data for step {step}")

    J_mag = np.sqrt(Jx ** 2 + Jy ** 2)

    fig, ax = _setup_axes(ax, figsize=figsize)
    x, y = solution.sites[:, 0], solution.sites[:, 1]
    tri = Triangulation(x, y, solution.triangles)

    # Main plot
    im = ax.tripcolor(tri, J_mag, cmap=cmap, shading='gouraud')
    fig.colorbar(im, ax=ax, label=label)

    # Streamlines
    if streamplot:
        xgrid, ygrid, Jx_grid = _interpolate_to_grid(x, y, Jx, grid_shape, method='linear')
        _, _, Jy_grid = _interpolate_to_grid(x, y, Jy, grid_shape, method='linear')
        _, _, J_mag_grid = _interpolate_to_grid(x, y, J_mag, grid_shape, method='linear')

        # Mask weak currents
        if min_stream_amp is not None:
            cutoff = np.nanmax(J_mag_grid) * min_stream_amp
            mask = J_mag_grid < cutoff
            Jx_grid[mask] = np.nan
            Jy_grid[mask] = np.nan

        ax.streamplot(xgrid, ygrid, Jx_grid, Jy_grid,
                      color='w', density=1.0, linewidth=0.75)

    ax.set_title(f'{label} (step {step})')
    ax.set_xlabel('x [ξ]')
    ax.set_ylabel('y [ξ]')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 5. ENERGY VS TIME
# ============================================================================

def plot_energy_vs_time(
        solution: Solution,
        figsize: Tuple[float, float] = (8, 4),
        ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot GL energy vs time (both methods on one plot)."""
    time, e_vor, e_tri = solution.get_energy_series()

    if len(time) == 0:
        raise ValueError("No energy data. Check track_conservation=True in options.")

    fig, ax = _setup_axes(ax, figsize=figsize)

    ax.plot(time, e_vor, 'b-', label='Voronoi', linewidth=1.5)
    ax.plot(time, e_tri, 'r--', label='Triangles', linewidth=1.5)

    ax.set_xlabel('Time [τ₀]')
    ax.set_ylabel('GL Energy')
    ax.set_title('Ginzburg-Landau Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show difference between methods
    ax2 = ax.twinx()
    diff = np.abs(e_vor - e_tri)
    ax2.plot(time, diff, 'g:', alpha=0.6, label='|E_vor - E_tri|')
    ax2.set_ylabel('|ΔE|', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    return fig, ax


# ============================================================================
# 6. CONSERVATION CHECKS VS TIME
# ============================================================================

def plot_conservation_checks(
        solution: Solution,
        figsize: Tuple[float, float] = (14, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot conservation checks vs time.

    Two plots:
    - Left: Integral checks (flux vs div)
    - Right: Poisson residual
    """
    cons = solution.get_conservation_series()

    if not cons:
        raise ValueError("No conservation data. Check track_conservation=True.")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # === Left: integral checks ===
    ax = axes[0]
    if 'flux_edges' in cons:
        t, v = cons['flux_edges']
        ax.plot(t, v, label='Flux (edges)', linewidth=1.5)
    if 'flux_sites' in cons:
        t, v = cons['flux_sites']
        ax.plot(t, v, label='Flux (sites)', linewidth=1.5)
    if 'div_voronoi' in cons:
        t, v = cons['div_voronoi']
        ax.plot(t, v, label='∫div(J) Voronoi', linewidth=1.5, linestyle='--')
    if 'div_triangles' in cons:
        t, v = cons['div_triangles']
        ax.plot(t, v, label='∫div(J) Triangles', linewidth=1.5, linestyle='--')

    ax.set_xlabel('Time [τ₀]')
    ax.set_ylabel('Integral value')
    ax.set_title('Global Conservation: ∮J·dl vs ∫div(J)dS')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Right: Poisson residual ===
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
# 7. SUMMARY PLOT (SPATIAL FIELDS)
# ============================================================================

def plot_summary(
        solution: Solution,
        step: int = -1,
        figsize: Tuple[float, float] = (18, 8),
        save_path: Optional[str] = None,
):
    """
    Plot all spatial fields in a 2×3 grid.

    Top row: |ψ|, phase(ψ), μ
    Bottom row: |Js|, |Jn|, div(Js)

    Args:
        solution: Solution object
        step: Step to display (-1 = last)
        figsize: Figure size (width, height) in inches
        save_path: If provided, save figure to file
    """
    fig = plt.figure(figsize=figsize)
    data = solution.get_spatial_data(step=step)

    psi = data.get('psi')
    mu = data.get('mu')
    div_J = data.get('div_Js')

    if psi is None or mu is None or div_J is None:
        raise ValueError(f"Incomplete data for step {step}")

    tri = Triangulation(solution.sites[:, 0], solution.sites[:, 1], solution.triangles)

    # === TOP ROW ===

    # 1. |ψ|
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.tripcolor(tri, np.abs(psi), cmap='viridis', vmin=0, vmax=1, shading='gouraud')
    ax1.set_title(r'$|\psi|$')
    ax1.set_aspect('equal')
    fig.colorbar(im1, ax=ax1)

    # 2. phase(ψ)
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.tripcolor(tri, np.angle(psi) / np.pi, cmap='twilight_shifted',
                        vmin=-1, vmax=1, shading='gouraud')
    ax2.set_title(r'Phase $\theta/\pi$')
    ax2.set_aspect('equal')
    fig.colorbar(im2, ax=ax2)

    # 3. μ
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.tripcolor(tri, mu, cmap='magma', shading='gouraud')
    ax3.set_title(r'$\mu$')
    ax3.set_aspect('equal')
    fig.colorbar(im3, ax=ax3)

    # === BOTTOM ROW ===

    # 4. |Js| with streamlines
    ax4 = fig.add_subplot(2, 3, 4)
    plot_currents(
        solution=solution,
        ax=ax4,
        dataset='supercurrent',
        step=step,
        streamplot=True,
        min_stream_amp=0.025,
        cmap='inferno',
    )

    # 5. |Jn| with streamlines
    ax5 = fig.add_subplot(2, 3, 5)
    plot_currents(
        solution=solution,
        ax=ax5,
        dataset='normal_current',
        step=step,
        streamplot=True,
        min_stream_amp=0.025,
        cmap='plasma',
    )

    # 6. div(Js)
    ax6 = fig.add_subplot(2, 3, 6)
    div_J_real = np.real(div_J)
    im6 = ax6.tripcolor(tri, div_J_real, cmap='RdBu_r', shading='gouraud',
                        vmin=-np.max(np.abs(div_J_real)), vmax=np.max(np.abs(div_J_real)))
    ax6.set_title(r'$\nabla \cdot J_s$')
    ax6.set_aspect('equal')
    fig.colorbar(im6, ax=ax6)

    fig.suptitle(f'TDGL Solution — Step {step}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig


# ============================================================================
# 8. CONSERVATION TRACKER SUMMARY
# ============================================================================

def plot_conservation_summary(
        solution: Solution,
        figsize: Tuple[float, float] = (18, 12),
        save_path: Optional[str] = None,
):
    """
    Plot ConservationTracker data in a 3×3 grid.

    Top row:
    - Energy (Voronoi + Triangles)
    - Flux (sites + edges)
    - div(J) (Voronoi + Triangles)

    Bottom row:
    - Poisson residual (max + mean)
    - min|Jn| vs max|Js|
    - mean|Jn| vs max|Jn|

    Args:
        solution: Solution object
        figsize: Figure size (width, height) in inches
        save_path: If provided, save figure to file

    Raises:
        ValueError: If ConservationTracker data is not available
    """
    # Check if conservation data exists
    ts_keys = solution.list_time_series()
    required_keys = ['energy_voronoi', 'flux_edges', 'poisson_residual_max']
    if not all(k in ts_keys for k in required_keys):
        raise ValueError(
            "ConservationTracker data not found. "
            "Check that track_conservation=True in SolverOptions."
        )

    fig = plt.figure(figsize=figsize)

    # === TOP ROW ===

    # 1. Energy
    ax1 = fig.add_subplot(2, 3, 1)
    time, e_vor, e_tri = solution.get_energy_series()
    # Filter out zeros (if any)
    mask = (e_vor != 0) & (e_tri != 0)
    ax1.plot(time[mask], e_vor[mask], 'b-', label='Voronoi', linewidth=1.5)
    ax1.plot(time[mask], e_tri[mask], 'r--', label='Triangles', linewidth=1.5)
    ax1.set_xlabel('Time [τ₀]')
    ax1.set_ylabel('GL Energy')
    ax1.set_title('Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Flux
    ax2 = fig.add_subplot(2, 3, 2)
    cons = solution.get_conservation_series()
    if 'flux_sites' in cons:
        t, v = cons['flux_sites']
        mask = v != 0
        ax2.plot(t[mask], v[mask], label='Flux (sites)', linewidth=1.5)
    if 'flux_edges' in cons:
        t, v = cons['flux_edges']
        mask = v != 0
        ax2.plot(t[mask], v[mask], label='Flux (edges)', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('Time [τ₀]')
    ax2.set_ylabel('Flux')
    ax2.set_title('Boundary Flux')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. div(J)
    ax3 = fig.add_subplot(2, 3, 3)
    if 'div_voronoi' in cons:
        t, v = cons['div_voronoi']
        mask = v != 0
        ax3.plot(t[mask], v[mask], label='div Voronoi', linewidth=1.5)
    if 'div_triangles' in cons:
        t, v = cons['div_triangles']
        mask = v != 0
        ax3.plot(t[mask], v[mask], label='div Triangles', linewidth=1.5, linestyle='--')
    ax3.set_xlabel('Time [τ₀]')
    ax3.set_ylabel('∫div(J)dS')
    ax3.set_title('Divergence Integral')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === BOTTOM ROW ===

    # 4. Poisson residual
    ax4 = fig.add_subplot(2, 3, 4)
    time, res_max, res_mean = solution.get_poisson_residual_series()
    mask = (res_max != 0) & (res_mean != 0)
    ax4.semilogy(time[mask], res_max[mask], 'r-', label='Max', linewidth=1.5)
    ax4.semilogy(time[mask], res_mean[mask], 'b-', label='Mean', linewidth=1.5)
    ax4.set_xlabel('Time [τ₀]')
    ax4.set_ylabel('Residual')
    ax4.set_title('Poisson Residual')
    ax4.legend()
    ax4.grid(True, alpha=0.3, which='both')

    # 5. min|Jn| vs max|Js|
    ax5 = fig.add_subplot(2, 3, 5)
    bc = solution.get_currents_series()
    if 'min_supercurrent_mag' in bc:
        t, v = bc['min_supercurrent_mag']
        mask = v != 0
        ax5.plot(t[mask], v[mask], label='min|Js|', linewidth=1.5)
    if 'max_normal_current' in bc:
        t, v = bc['max_normal_current']
        mask = v != 0
        ax5.plot(t[mask], v[mask], label='max|Jn|', linewidth=1.5, linestyle='--')
    ax5.set_xlabel('Time [τ₀]')
    ax5.set_ylabel('Current')
    ax5.set_title('Equilibrium Check')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. mean|Jn| vs max|Jn|
    ax6 = fig.add_subplot(2, 3, 6)
    if 'mean_Jn_boundary' in bc:
        t, v = bc['mean_Jn_boundary']
        mask = v != 0
        ax6.plot(t[mask], v[mask], label='mean|Jn|', linewidth=1.5)
    if 'max_Jn_boundary' in bc:
        t, v = bc['max_Jn_boundary']
        mask = v != 0
        ax6.plot(t[mask], v[mask], label='max|Jn|', linewidth=1.5, linestyle='--')
    ax6.set_xlabel('Time [τ₀]')
    ax6.set_ylabel('Jn at boundary')
    ax6.set_title('Boundary Normal Current')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    fig.suptitle('ConservationTracker Summary', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig


# ============================================================================
# 9. PHYSICAL TRACKER SUMMARY (PROBE POINTS)
# ============================================================================

def plot_probe_summary(
        solution: Solution,
        groups: Optional[List[List[int]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
):
    """
    Plot PhysicalTracker data (mu and phase at probe points).

    User can group probe points to plot them together.

    Args:
        solution: Solution object
        groups: List of probe point index groups.
                Example: [[0, 1], [2, 3]] → 2×2 grid (mu and phase for each group)
                Example: [[0, 1, 2, 3]] → 2×1 grid (mu and phase for all points)
                If None, all points on one plot (2×1 grid)
        figsize: Figure size (width, height) in inches. If None, auto-calculated.
        save_path: If provided, save figure to file

    Raises:
        ValueError: If PhysicalTracker data is not available

    Example:
        >>> # Plot probes 0,1 together and probes 2,3 together
        >>> plot_probe_summary(solution, groups=[[0, 1], [2, 3]])

        >>> # Plot all probes on one plot
        >>> plot_probe_summary(solution, groups=[[0, 1, 2, 3]])
    """
    # Check if probe data exists
    ts_keys = solution.list_time_series()
    if 'mu_probe' not in ts_keys:
        raise ValueError(
            "PhysicalTracker data not found. "
            "Check that device.probe_points is set and track_physical=True."
        )

    probe_data = solution.get_probe_data()
    time = probe_data['time']
    mu_probe = probe_data['mu_probe']  # shape: (n_times, n_probes)
    phase_probe = probe_data['phase_probe']  # shape: (n_times, n_probes)
    coords = probe_data.get('coords')  # shape: (n_probes, 2)

    n_probes = mu_probe.shape[1]

    # Default grouping: all probes in one group
    if groups is None:
        groups = [list(range(n_probes))]

    # Validate groups
    for group in groups:
        for idx in group:
            if idx < 0 or idx >= n_probes:
                raise ValueError(f"Probe index {idx} out of range [0, {n_probes-1}]")

    n_groups = len(groups)

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (10, 4 * n_groups)

    fig, axes = plt.subplots(n_groups, 2, figsize=figsize)

    # Handle case when n_groups = 1
    if n_groups == 1:
        axes = axes.reshape(1, 2)

    for i, group in enumerate(groups):
        ax_mu = axes[i, 0]
        ax_phase = axes[i, 1]

        # Plot mu for each probe in group
        for probe_idx in group:
            label = f'Probe {probe_idx}'
            if coords is not None:
                label += f' ({coords[probe_idx, 0]:.2f}, {coords[probe_idx, 1]:.2f})'
            ax_mu.plot(time, mu_probe[:, probe_idx], linewidth=1.5, label=label)

        ax_mu.set_xlabel('Time [τ₀]')
        ax_mu.set_ylabel('μ')
        ax_mu.set_title(f'μ at probes {group}')
        ax_mu.legend(fontsize=8)
        ax_mu.grid(True, alpha=0.3)

        # Plot phase for each probe in group
        for probe_idx in group:
            label = f'Probe {probe_idx}'
            if coords is not None:
                label += f' ({coords[probe_idx, 0]:.2f}, {coords[probe_idx, 1]:.2f})'
            ax_phase.plot(time, phase_probe[:, probe_idx], linewidth=1.5, label=label)

        ax_phase.set_xlabel('Time [τ₀]')
        ax_phase.set_ylabel('Phase [π]')
        ax_phase.set_title(f'Phase at probes {group}')
        ax_phase.legend(fontsize=8)
        ax_phase.grid(True, alpha=0.3)

    fig.suptitle('PhysicalTracker Summary', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig