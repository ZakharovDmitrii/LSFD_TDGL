"""
animation.py — Create animations from TDGL simulation results.

Supports animation of spatial fields over time:
    - order_parameter: |ψ| (modulus of order parameter)
    - phase: arg(ψ)/π (phase of order parameter)
    - mu: scalar potential
    - div_Js: divergence of supercurrent
    - supercurrent: |Js| (modulus of supercurrent)
    - normal_current: |Jn| (modulus of normal current)

Can create single-panel or multi-panel animations.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import numpy as np

from .solution import Solution


# ============================================================================
# QUANTITY CONFIGURATION
# ============================================================================

# Mapping from quantity name to (title, colorbar label, cmap, vmin, vmax)
QUANTITY_CONFIG: Dict[str, Dict] = {
    'order_parameter': {
        'title': r'$|\psi|$',
        'label': r'$|\psi|$',
        'cmap': 'viridis',
        'vmin': 0.0,
        'vmax': 1.0,
    },
    'psi': {
        'title': r'$|\psi|$',
        'label': r'$|\psi|$',
        'cmap': 'viridis',
        'vmin': 0.0,
        'vmax': 1.0,
    },
    'phase': {
        'title': r'Phase $\theta/\pi$',
        'label': r'$\theta/\pi$',
        'cmap': 'twilight_shifted',
        'vmin': -1.0,
        'vmax': 1.0,
    },
    'mu': {
        'title': r'Scalar potential $\mu$',
        'label': r'$\mu$',
        'cmap': 'magma',
        'vmin': None,  # auto
        'vmax': None,  # auto
    },
    'div_Js': {
        'title': r'$\nabla \cdot J_s$',
        'label': r'$\nabla \cdot J_s$',
        'cmap': 'RdBu_r',
        'vmin': None,  # auto (symmetric)
        'vmax': None,  # auto (symmetric)
    },
    'supercurrent': {
        'title': r'$|J_s|$',
        'label': r'$|J_s|$',
        'cmap': 'inferno',
        'vmin': 0.0,
        'vmax': None,  # auto
    },
    'normal_current': {
        'title': r'$|J_n|$',
        'label': r'$|J_n|$',
        'cmap': 'plasma',
        'vmin': 0.0,
        'vmax': None,  # auto
    },
}


def _extract_quantity(step_data: Dict[str, np.ndarray], quantity: str) -> np.ndarray:
    """
    Extract the spatial field for a given quantity from step data.

    Args:
        step_data: Dictionary with keys like 'psi', 'mu', 'supercurrent_x', etc.
        quantity: Name of the quantity to extract.

    Returns:
        1D numpy array of values at each mesh site.

    Raises:
        ValueError: If quantity is unknown or data is missing.
    """
    if quantity in ['order_parameter', 'psi']:
        if 'psi' not in step_data:
            raise ValueError("Missing 'psi' in step data")
        return np.abs(step_data['psi'])

    elif quantity == 'phase':
        if 'psi' not in step_data:
            raise ValueError("Missing 'psi' in step data")
        return np.angle(step_data['psi']) / np.pi

    elif quantity == 'mu':
        if 'mu' not in step_data:
            raise ValueError("Missing 'mu' in step data")
        return step_data['mu']

    elif quantity == 'div_Js':
        if 'div_Js' not in step_data:
            raise ValueError("Missing 'div_Js' in step data")
        return np.real(step_data['div_Js'])

    elif quantity == 'supercurrent':
        if 'supercurrent_x' not in step_data or 'supercurrent_y' not in step_data:
            raise ValueError("Missing supercurrent components in step data")
        Jx = step_data['supercurrent_x']
        Jy = step_data['supercurrent_y']
        return np.sqrt(Jx ** 2 + Jy ** 2)

    elif quantity == 'normal_current':
        if 'normal_current' not in step_data:
            raise ValueError("Missing 'normal_current' in step data")
        nc = step_data['normal_current']
        return np.sqrt(nc[:, 0] ** 2 + nc[:, 1] ** 2)

    else:
        raise ValueError(
            f"Unknown quantity: {quantity!r}. "
            f"Available: {list(QUANTITY_CONFIG.keys())}"
        )


def _compute_auto_range(
    solution: Solution,
    quantity: str,
    steps: List[int],
) -> Tuple[float, float]:
    """
    Compute automatic vmin/vmax for a quantity by scanning all frames.

    For symmetric quantities (div_Js), uses symmetric range [-max, max].
    For positive quantities (|ψ|, |Js|, |Jn|), uses [0, max].
    For signed quantities (mu), uses [min, max].
    """
    config = QUANTITY_CONFIG[quantity]

    # If both are fixed, return them
    if config['vmin'] is not None and config['vmax'] is not None:
        return config['vmin'], config['vmax']

    # Scan all frames to find min/max
    global_min = np.inf
    global_max = -np.inf

    for step in steps:
        data = solution.get_spatial_data(step=step)
        values = _extract_quantity(data, quantity)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        global_min = min(global_min, float(np.min(values)))
        global_max = max(global_max, float(np.max(values)))

    # Apply fixed bounds from config
    vmin = config['vmin'] if config['vmin'] is not None else global_min
    vmax = config['vmax'] if config['vmax'] is not None else global_max

    # For symmetric quantities, make range symmetric
    if quantity == 'div_Js':
        abs_max = max(abs(global_min), abs(global_max))
        vmin, vmax = -abs_max, abs_max

    # Safety margin (5%)
    if config['vmin'] is None and config['vmax'] is None:
        span = vmax - vmin
        if span > 0:
            vmin -= 0.05 * span
            vmax += 0.05 * span

    return vmin, vmax


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

    Supports single-panel or multi-panel animations. For multi-panel,
    quantities are arranged in a grid (1 column, N rows).

    Args:
        solution: Solution object with simulation results.
        fig_name: Output file name (without extension).
        file_dir: Directory to save the animation.
        quantities: Quantity name or list of names.
            Available: 'order_parameter', 'phase', 'mu', 'div_Js',
            'supercurrent', 'normal_current'.
        fps: Frames per second.
        figsize: Figure size (width, height). If None, auto-calculated.
        dpi: Resolution.
        vmin: Color scale minimum. Can be:
            - None: use default from QUANTITY_CONFIG
            - float: fixed value (for single quantity)
            - list of floats: one per quantity (for multi-panel)
            - 'auto': scan all frames to find min
        vmax: Color scale maximum (same options as vmin).
        steps: List of step indices to animate. If None, uses all saved steps.

    Returns:
        Path to the created GIF file, or None if animation failed.

    Examples:
        >>> # Single quantity
        >>> make_video_from_solution(sol, "energy", "./", quantities="order_parameter")

        >>> # Multiple quantities in one video
        >>> make_video_from_solution(sol, "overview", "./",
        ...     quantities=["order_parameter", "phase", "mu"])

        >>> # Custom range
        >>> make_video_from_solution(sol, "mu", "./", quantities="mu",
        ...     vmin=-1.0, vmax=1.0)
    """
    # Normalize quantities to list
    if isinstance(quantities, str):
        quantities = [quantities]

    n_quantities = len(quantities)

    # Validate quantities
    for q in quantities:
        if q not in QUANTITY_CONFIG:
            raise ValueError(
                f"Unknown quantity: {q!r}. "
                f"Available: {list(QUANTITY_CONFIG.keys())}"
            )

    # Create output directory
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(file_dir, f"{fig_name}.gif")

    # Get step list
    if steps is None:
        steps = solution._saved_steps

    n_frames = len(steps)
    if n_frames < 2:
        print(f"⚠️ Not enough frames for animation (n_frames={n_frames})")
        return None

    # Auto-calculate figsize if not provided
    if figsize is None:
        if n_quantities == 1:
            figsize = (8, 6)
        else:
            figsize = (8, 4 * n_quantities)

    # Compute vmin/vmax for each quantity
    vmin_list = []
    vmax_list = []
    for i, q in enumerate(quantities):
        # Handle user-provided vmin/vmax
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
            # Auto-compute
            auto_vmin, auto_vmax = _compute_auto_range(solution, q, steps)
            vmin_list.append(user_vmin if user_vmin is not None else auto_vmin)
            vmax_list.append(user_vmax if user_vmax is not None else auto_vmax)

    # Create figure and axes
    fig, axes = plt.subplots(n_quantities, 1, figsize=figsize, dpi=dpi)
    if n_quantities == 1:
        axes = [axes]

    # Setup triangulation
    x = solution.sites[:, 0]
    y = solution.sites[:, 1]
    triangulation = mtri.Triangulation(x, y, solution.triangles)

    # Initialize plots
    ims = []
    cbars = []
    for i, (ax, q) in enumerate(zip(axes, quantities)):
        config = QUANTITY_CONFIG[q]
        ax.set_aspect('equal')
        ax.set_xlabel('x [ξ]')
        ax.set_ylabel('y [ξ]')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        # Initial empty plot
        im = ax.tripcolor(
            triangulation, np.zeros(len(x)),
            cmap=config['cmap'],
            vmin=vmin_list[i], vmax=vmax_list[i],
            shading='gouraud',
        )
        ims.append(im)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(config['label'])
        cbars.append(cbar)

    # Title (shared)
    title = fig.suptitle('')

    def update(frame_idx):
        """Update all panels for one frame."""
        step = steps[frame_idx]
        data = solution.get_spatial_data(step=step)
        time = data.get('time', frame_idx)

        for i, (im, q) in enumerate(zip(ims, quantities)):
            values = _extract_quantity(data, q)
            im.set_array(values)

        title.set_text(f't = {time:.3f} τ₀  (step {step})')
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