"""
animation.py — Create animations from TDGL simulation results.

Supports animation of spatial fields over time:
    - order_parameter: |ψ| (modulus of order parameter)
    - phase: arg(ψ)/π (phase of order parameter)
    - mu: scalar potential
    - div_Js: divergence of supercurrent
    - supercurrent: |Js| (modulus of supercurrent)
    - normal_current: |Jn| (modulus of normal current)

Layouts:
    - 1 quantity: single panel
    - 2-3 quantities: vertical stack (N × 1)
    - 6 quantities: 2 × 3 grid (same layout as plot_summary)
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import numpy as np

from .solution import Solution


# ============================================================================
# QUANTITY CONFIGURATION
# ============================================================================

QUANTITY_CONFIG: Dict[str, Dict] = {
    'order_parameter': {
        'title': r'$|\psi|$',
        'label': r'$|\psi|$',
        'cmap': 'viridis',
        'vmin': 0.0, 'vmax': 1.0,
    },
    'psi': {
        'title': r'$|\psi|$',
        'label': r'$|\psi|$',
        'cmap': 'viridis',
        'vmin': 0.0, 'vmax': 1.0,
    },
    'phase': {
        'title': r'Phase $\theta/\pi$',
        'label': r'$\theta/\pi$',
        'cmap': 'twilight_shifted',
        'vmin': -1.0, 'vmax': 1.0,
    },
    'mu': {
        'title': r'$\mu$',
        'label': r'$\mu$',
        'cmap': 'magma',
        'vmin': None, 'vmax': None,
    },
    'div_Js': {
        'title': r'$\nabla \cdot J_s$',
        'label': r'$\nabla \cdot J_s$',
        'cmap': 'RdBu_r',
        'vmin': None, 'vmax': None,
    },
    'supercurrent': {
        'title': r'$|J_s|$',
        'label': r'$|J_s|$',
        'cmap': 'inferno',
        'vmin': 0.0, 'vmax': None,
    },
    'normal_current': {
        'title': r'$|J_n|$',
        'label': r'$|J_n|$',
        'cmap': 'plasma',
        'vmin': 0.0, 'vmax': None,
    },
}


def _extract_quantity(step_data: Dict[str, np.ndarray], quantity: str) -> np.ndarray:
    """Extract the spatial field for a given quantity from step data."""
    if quantity in ['order_parameter', 'psi']:
        return np.abs(step_data['psi'])
    elif quantity == 'phase':
        return np.angle(step_data['psi']) / np.pi
    elif quantity == 'mu':
        return step_data['mu']
    elif quantity == 'div_Js':
        return np.real(step_data['div_Js'])
    elif quantity == 'supercurrent':
        Jx, Jy = step_data['supercurrent_x'], step_data['supercurrent_y']
        return np.sqrt(Jx ** 2 + Jy ** 2)
    elif quantity == 'normal_current':
        nc = step_data['normal_current']
        return np.sqrt(nc[:, 0] ** 2 + nc[:, 1] ** 2)
    else:
        raise ValueError(f"Unknown quantity: {quantity!r}. Available: {list(QUANTITY_CONFIG.keys())}")


def _compute_auto_range(solution: Solution, quantity: str, steps: List[int]) -> Tuple[float, float]:
    """Compute automatic vmin/vmax by scanning all frames."""
    config = QUANTITY_CONFIG[quantity]
    if config['vmin'] is not None and config['vmax'] is not None:
        return config['vmin'], config['vmax']

    global_min, global_max = np.inf, -np.inf
    for step in steps:
        data = solution.get_spatial_data(step=step)
        values = _extract_quantity(data, quantity)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        global_min = min(global_min, float(np.min(values)))
        global_max = max(global_max, float(np.max(values)))

    vmin = config['vmin'] if config['vmin'] is not None else global_min
    vmax = config['vmax'] if config['vmax'] is not None else global_max

    if quantity == 'div_Js':
        abs_max = max(abs(global_min), abs(global_max))
        vmin, vmax = -abs_max, abs_max

    if config['vmin'] is None and config['vmax'] is None:
        span = vmax - vmin
        if span > 0:
            vmin -= 0.05 * span
            vmax += 0.05 * span

    return vmin, vmax


def _hide_inner_axes(axes, nrows: int, ncols: int) -> None:
    """
    Hide axis labels on inner panels for cleaner multi-panel figures.

    Rules:
        - Y-axis labels only on the leftmost column (j == 0)
        - X-axis labels only on the bottom row (i == nrows - 1)
    """
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j] if nrows > 1 or ncols > 1 else axes
            if j != 0:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
            if i != nrows - 1:
                ax.set_xlabel('')
                ax.tick_params(labelbottom=False)


# ============================================================================
# MAIN ANIMATION FUNCTION
# ============================================================================

def make_video_from_solution(
    solution: Solution,
    fig_name: str,
    file_dir: str,
    quantities: Union[str, List[str]] = "order_parameter",
    fps: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 100,
    vmin: Optional[Union[float, List[float]]] = None,
    vmax: Optional[Union[float, List[float]]] = None,
    steps: Optional[List[int]] = None,
) -> str:
    """
    Create a GIF animation of spatial fields over time.

    Layouts:
        - 1 quantity: single panel
        - 2-3 quantities: vertical stack (N × 1)
        - 6 quantities: 2 × 3 grid (same as plot_summary)
            Row 0: |ψ|, phase(ψ), μ
            Row 1: div(Js), |Js|, |Jn|

    Args:
        solution: Solution object with simulation results.
        fig_name: Output file name (without extension).
        file_dir: Directory to save the animation.
        quantities: Quantity name or list of names.
        fps: Frames per second.
        figsize: Figure size (width, height). If None, auto-calculated.
        dpi: Resolution.
        vmin, vmax: Color scale bounds (None = auto, float = fixed).
        steps: List of step indices to animate. If None, uses all saved steps.

    Returns:
        Path to the created GIF file, or None if animation failed.
    """
    if isinstance(quantities, str):
        quantities = [quantities]

    n_quantities = len(quantities)
    for q in quantities:
        if q not in QUANTITY_CONFIG:
            raise ValueError(f"Unknown quantity: {q!r}. Available: {list(QUANTITY_CONFIG.keys())}")

    Path(file_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(file_dir, f"{fig_name}.gif")

    if steps is None:
        steps = solution._saved_steps

    n_frames = len(steps)
    if n_frames < 2:
        print(f"⚠️ Not enough frames for animation (n_frames={n_frames})")
        return None

    # Determine grid layout
    if n_quantities == 1:
        nrows, ncols = 1, 1
    elif n_quantities <= 3:
        nrows, ncols = n_quantities, 1
    elif n_quantities == 6:
        nrows, ncols = 2, 3
    else:
        raise ValueError(f"Unsupported number of quantities: {n_quantities}. Use 1, 2, 3, or 6.")

    # Auto-calculate figsize if not provided
    if figsize is None:
        if n_quantities == 1:
            figsize = (8, 6)
        elif n_quantities <= 3:
            figsize = (8, 4 * n_quantities)
        else:  # 6 quantities
            figsize = (14, 10)

    # Compute vmin/vmax for each quantity
    vmin_list, vmax_list = [], []
    for i, q in enumerate(quantities):
        user_vmin = vmin if isinstance(vmin, (int, float)) or vmin is None else (
            vmin[i] if isinstance(vmin, list) and i < len(vmin) else None
        )
        user_vmax = vmax if isinstance(vmax, (int, float)) or vmax is None else (
            vmax[i] if isinstance(vmax, list) and i < len(vmax) else None
        )

        if user_vmin is not None and user_vmax is not None:
            vmin_list.append(user_vmin)
            vmax_list.append(user_vmax)
        else:
            auto_vmin, auto_vmax = _compute_auto_range(solution, q, steps)
            vmin_list.append(user_vmin if user_vmin is not None else auto_vmin)
            vmax_list.append(user_vmax if user_vmax is not None else auto_vmax)

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    # Setup triangulation
    x = solution.sites[:, 0]
    y = solution.sites[:, 1]
    triangulation = mtri.Triangulation(x, y, solution.triangles)

    # Initialize plots
    ims = []
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx < n_quantities:
                ax = axes[i, j]
                config = QUANTITY_CONFIG[quantities[idx]]
                ax.set_aspect('equal')
                ax.set_xlabel('x [ξ]')
                ax.set_ylabel('y [ξ]')
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())
                ax.set_title(config['title'])

                im = ax.tripcolor(
                    triangulation, np.zeros(len(x)),
                    cmap=config['cmap'],
                    vmin=vmin_list[idx], vmax=vmax_list[idx],
                    shading='gouraud',
                )
                ims.append(im)
                fig.colorbar(im, ax=ax, label=config['label'])

    # Hide inner axes
    _hide_inner_axes(axes, nrows, ncols)

    # Shared title with time info
    title = fig.suptitle('')

    def update(frame_idx):
        """Update all panels for one frame."""
        step = steps[frame_idx]
        data = solution.get_spatial_data(step=step)
        time_val = data.get('time', frame_idx)

        for idx, im in enumerate(ims):
            values = _extract_quantity(data, quantities[idx])
            im.set_array(values)

        title.set_text(f't = {time_val:.3f} τ₀  (step {step})')
        return ims

    # Create animation
    print(f"🎬 Creating animation ({n_frames} frames, {fps} fps, {n_quantities} panel(s))...")
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=True
    )

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"✅ Animation saved: {output_path}")
    return output_path


def create_animation(
    solution: Solution,
    output_path: str,
    quantities: Union[str, List[str]] = "order_parameter",
    fps: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
) -> str:
    """
    Convenience wrapper for make_video_from_solution.

    Args:
        solution: Solution object.
        output_path: Output file path (with .gif extension).
        quantities: Quantity name or list of names.
        fps: Frames per second.
        figsize: Figure size.

    Returns:
        Path to the created file.
    """
    file_dir = os.path.dirname(output_path) or "."
    fig_name = Path(output_path).stem
    return make_video_from_solution(
        solution, fig_name, file_dir,
        quantities=quantities,
        fps=fps, figsize=figsize,
    )