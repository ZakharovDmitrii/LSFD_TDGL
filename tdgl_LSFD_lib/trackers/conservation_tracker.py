"""
conservation_tracker.py — Tracker for energy, conservation, and boundary currents.

Monitors:
    - Ginzburg-Landau energy (two integration methods: Voronoi and triangles)
    - Current conservation (4 values from FVM check)
    - Poisson residual (max and mean)
    - Boundary current diagnostics

Uses fixed-size numpy buffers in RAM, flushed to HDF5 periodically.
"""
import logging
from typing import Dict, Any, Optional

import numpy as np

from .base_tracker import SimulationTracker
from ..solver.solver import StepResult, TDGLSolver
from ..operators.fvm_integrator import FVMIntegrator
from ..mesh.mesh import Mesh

logger = logging.getLogger(__name__)


class ConservationTracker(SimulationTracker):
    """
    Tracks energy conservation, boundary currents, and Poisson residual.

    Args:
        mesh: Mesh object for FVM integration.
        buffer_size: Size of memory buffer (typically = save_every).
        log_every: Log diagnostics every N steps (None to disable).
    """

    def __init__(
        self,
        mesh: Mesh,
        buffer_size: int = 100,
        log_every: Optional[int] = 1000,
    ):
        self.mesh = mesh
        self.buffer_size = buffer_size
        self.log_every = log_every

        # FVM integrator for energy and conservation
        self.fvm = FVMIntegrator(mesh=mesh)

        # === BUFFERS (fixed-size numpy arrays) ===
        # Note: time, step, dt are stored by Runner, not here

        # Energy
        self._energy_voronoi_buf = np.zeros(buffer_size, dtype=np.float64)
        self._energy_triangles_buf = np.zeros(buffer_size, dtype=np.float64)

        # Conservation (4 values)
        self._flux_edges_buf = np.zeros(buffer_size, dtype=np.float64)
        self._flux_sites_buf = np.zeros(buffer_size, dtype=np.float64)
        self._div_voronoi_buf = np.zeros(buffer_size, dtype=np.float64)
        self._div_triangles_buf = np.zeros(buffer_size, dtype=np.float64)

        # Poisson residual
        self._poisson_residual_max_buf = np.zeros(buffer_size, dtype=np.float64)
        self._poisson_residual_mean_buf = np.zeros(buffer_size, dtype=np.float64)

        # Boundary currents
        self._max_normal_current_buf = np.zeros(buffer_size, dtype=np.float64)
        self._min_supercurrent_mag_buf = np.zeros(buffer_size, dtype=np.float64)
        self._mean_Jn_boundary_buf = np.zeros(buffer_size, dtype=np.float64)
        self._max_Jn_boundary_buf = np.zeros(buffer_size, dtype=np.float64)

        # Current buffer index
        self._idx = 0

        # HDF5 datasets created flag
        self._datasets_created = False

    def on_step(
        self,
        step: int,
        t: float,
        dt: float,
        result: StepResult,
        solver: TDGLSolver,
    ) -> Dict[str, Any]:
        """Compute scalars from StepResult and store in buffer."""

        idx = self._idx

        # Energy
        energy_v, energy_t = self._compute_energy(result)
        self._energy_voronoi_buf[idx] = energy_v
        self._energy_triangles_buf[idx] = energy_t

        # Conservation
        flux_e, flux_s, div_v, div_t, mean_Jn, max_Jn = self._check_conservation(result)
        self._flux_edges_buf[idx] = flux_e
        self._flux_sites_buf[idx] = flux_s
        self._div_voronoi_buf[idx] = div_v
        self._div_triangles_buf[idx] = div_t
        self._mean_Jn_boundary_buf[idx] = mean_Jn
        self._max_Jn_boundary_buf[idx] = max_Jn

        # Poisson residual
        res_max = float(np.max(np.abs(result.poisson_residual)))
        res_mean = float(np.mean(np.abs(result.poisson_residual)))
        self._poisson_residual_max_buf[idx] = res_max
        self._poisson_residual_mean_buf[idx] = res_mean

        # Max normal current
        max_normal = float(np.max(np.abs(result.normal_current)))
        # Min supercurrent magnitude
        sc_mag = np.sqrt(result.supercurrent_x ** 2 + result.supercurrent_y ** 2)
        min_super = float(np.min(sc_mag))

        self._max_normal_current_buf[idx] = max_normal
        self._min_supercurrent_mag_buf[idx] = min_super

        self._idx += 1

        # Log diagnostics
        if self.log_every and step % self.log_every == 0:
            self._log_diagnostics(step, t, energy_v, flux_s, res_max, max_Jn, max_normal, min_super)

        return {
            "energy_voronoi": energy_v,
            "energy_triangles": energy_t,
            "flux_edges": flux_e,
            "flux_sites": flux_s,
            "div_voronoi": div_v,
            "div_triangles": div_t,
            "poisson_residual_max": res_max,
            "poisson_residual_mean": res_mean,
            "max_normal_current": max_normal,
            "min_supercurrent_mag": min_super,
            "mean_Jn_boundary": mean_Jn,
            "max_Jn_boundary": max_Jn,
        }

    def flush_to_hdf5(self, hdf5_group) -> None:
        """Flush filled buffer to HDF5 file."""
        if self._idx == 0:
            return

        n = self._idx

        # Create datasets on first flush (resizable)
        if not self._datasets_created:
            # Note: time, step, dt, poisson_tolerance are created by Runner
            hdf5_group.create_dataset("energy_voronoi", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("energy_triangles", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("flux_edges", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("flux_sites", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("div_voronoi", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("div_triangles", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("poisson_residual_max", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("poisson_residual_mean", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("max_normal_current", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("min_supercurrent_mag", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("mean_Jn_boundary", (0,), maxshape=(None,), dtype='f8')
            hdf5_group.create_dataset("max_Jn_boundary", (0,), maxshape=(None,), dtype='f8')
            self._datasets_created = True

        # Resize and append
        current_size = hdf5_group["energy_voronoi"].shape[0]
        new_size = current_size + n

        hdf5_group["energy_voronoi"].resize(new_size, axis=0)
        hdf5_group["energy_voronoi"][current_size:new_size] = self._energy_voronoi_buf[:n]

        hdf5_group["energy_triangles"].resize(new_size, axis=0)
        hdf5_group["energy_triangles"][current_size:new_size] = self._energy_triangles_buf[:n]

        hdf5_group["flux_edges"].resize(new_size, axis=0)
        hdf5_group["flux_edges"][current_size:new_size] = self._flux_edges_buf[:n]

        hdf5_group["flux_sites"].resize(new_size, axis=0)
        hdf5_group["flux_sites"][current_size:new_size] = self._flux_sites_buf[:n]

        hdf5_group["div_voronoi"].resize(new_size, axis=0)
        hdf5_group["div_voronoi"][current_size:new_size] = self._div_voronoi_buf[:n]

        hdf5_group["div_triangles"].resize(new_size, axis=0)
        hdf5_group["div_triangles"][current_size:new_size] = self._div_triangles_buf[:n]

        hdf5_group["poisson_residual_max"].resize(new_size, axis=0)
        hdf5_group["poisson_residual_max"][current_size:new_size] = self._poisson_residual_max_buf[:n]

        hdf5_group["poisson_residual_mean"].resize(new_size, axis=0)
        hdf5_group["poisson_residual_mean"][current_size:new_size] = self._poisson_residual_mean_buf[:n]

        hdf5_group["max_normal_current"].resize(new_size, axis=0)
        hdf5_group["max_normal_current"][current_size:new_size] = self._max_normal_current_buf[:n]

        hdf5_group["min_supercurrent_mag"].resize(new_size, axis=0)
        hdf5_group["min_supercurrent_mag"][current_size:new_size] = self._min_supercurrent_mag_buf[:n]

        hdf5_group["mean_Jn_boundary"].resize(new_size, axis=0)
        hdf5_group["mean_Jn_boundary"][current_size:new_size] = self._mean_Jn_boundary_buf[:n]

        hdf5_group["max_Jn_boundary"].resize(new_size, axis=0)
        hdf5_group["max_Jn_boundary"][current_size:new_size] = self._max_Jn_boundary_buf[:n]

        # Reset buffer index
        self._idx = 0

        logger.debug(f"ConservationTracker flushed {n} steps to HDF5")

    # =========================================================================
    # Internal computation methods
    # =========================================================================

    def _compute_energy(self, result: StepResult) -> tuple[float, float]:
        """Compute GL energy using two integration methods."""
        s_x, s_y = result.s_applied[0], result.s_applied[1]
        Dx, Dy = result.psi_derivatives[:, 0], result.psi_derivatives[:, 1]
        psi = result.psi
        psi_conj = psi.conjugate()

        sq_psi = np.real(psi * psi_conj)
        sq_Dx = np.real(Dx * Dx.conjugate())
        sq_Dy = np.real(Dy * Dy.conjugate())

        s_grad_psi = psi_conj * (s_x * Dx + s_y * Dy)

        F_density = (
            -sq_psi + 0.5 * sq_psi ** 2 + (sq_Dx + sq_Dy)
            + 2 * 0.5 * np.real(s_grad_psi) #  result.eta
            + result.Bz ** 2
        )

        F_voronoi = self.fvm.compute_divergence_integral(F_density, method='voronoi')
        F_triangles = self.fvm.compute_divergence_integral(F_density, method='triangles')

        return float(F_voronoi), float(F_triangles)

    def _check_conservation(self, result: StepResult) -> tuple[float, float, float, float, float, float]:
        """Check global current conservation via FVM."""
        return self.fvm.global_conservation_check(
            J_x=result.supercurrent_x.real,
            J_y=result.supercurrent_y.real,
            div_J=result.div_Js.real,
        )

    def _log_diagnostics(
        self,
        step: int,
        t: float,
        energy_v: float,
        flux_s: float,
        res_max: float,
        max_Jn: float,
        max_normal: float,
        min_super: float,
    ) -> None:
        """Log conservation diagnostics."""
        equilibrium = "PASS" if max_normal <= min_super else "FAIL"
        logger.debug(
            f"\n[Step {step:5d}, t={t:.3f}] Conservation:\n"
            f"  Energy (Voronoi) = {energy_v:+.6e}\n"
            f"  ∮J·dl(sites)    = {flux_s:+.3e}\n"
            f"  Poisson residual = {res_max:.3e}\n"
            f"  max|J_s·n|       = {max_Jn:.3e}\n"
            f"  max|J_n|         = {max_normal:.3e}\n"
            f"  min|J_s|         = {min_super:.3e}\n"
            f"  Equilibrium:     {equilibrium}"
        )