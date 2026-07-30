import numpy as np
from typing import NamedTuple, Optional, Tuple
from collections import deque

from ..device.device import Device
from ..external_fields.external_fields import ExternalFields
from ..operators.operators import LSFD_operators
from .dynamics_options import SolverOptions, TimeScheme
import logging
logger = logging.getLogger(__name__)

# ============================================================================
# КОНТЕЙНЕР ДЛЯ РЕЗУЛЬТАТОВ ОДНОГО ШАГА
# ============================================================================
class StepResult(NamedTuple):
    """
    Container for one step simulation result.

    Main fields (always returned):
        psi: Wave function (N,) complex128
        psi_abs_sq: |ψ|² (N,) float64
        mu: Scalar potential (N,) float64
        supercurrent_x: Supercurrent along x (N,) float64
        supercurrent_y: Supercurrent along y (N,) float64
        div_Js: Divergence of supercurrent (N,) float64
        normal_current: Normal current -∇μ (N, 2) float64
        dt: Current time step
        poisson_tolerance: Tolerance for poisson solver, may be adaptive

    Optional fields (only if corresponding flags are enabled):
        psi_derivatives: Derivatives of psi (N, 14) complex128
        poisson_residual: Poisson equation residual (N,) float64
        poisson_iterations: Number of Poisson solver iterations
        energy_voronoi: GL energy (Voronoi method) float
        energy_triangles: GL energy (triangle method) float
        conservation_global: Global conservation check tuple
        conservation_local: Local conservation check tuple
    """
    psi: np.ndarray  # (N,) complex128
    psi_abs_sq: np.ndarray  # (N,) float64
    psi_derivatives: np.ndarray  # (N, 14) complex128
    mu: np.ndarray  # (N,) float64
    supercurrent_x: np.ndarray  # (N,) float64
    supercurrent_y: np.ndarray  # (N,) float64
    div_Js: np.ndarray  # (N,) float64
    normal_current: np.ndarray  # (N, 2) float64
    dt: float  # scalar
    s_applied: np.ndarray  # (2,) float64
    Bz: float  # scalar
    eta: float  # scalar
    gamma: float  # scalar
    poisson_tolerance: float # scalar
    poisson_residual: np.ndarray  # (N,) float64
    poisson_iterations: int  # scalar


# ============================================================================
# TDGL SOLVER
# ============================================================================

class TDGLSolver:

    def __init__(
            self,
            device: Device,
            operators: LSFD_operators,
            external_fields: ExternalFields,
            options: SolverOptions | None = None,
    ):
        # === Raise error if mesh doesn't exist===
        if device.mesh is None:
            raise ValueError(
                "device.mesh is None. Call device.make_mesh() before creating TDGLSolver."
            )

        # === VALIDATE OPTIONS ===
        if options is None:
            options = SolverOptions(solve_time=1.0)
        self.options = options

        self.device = device
        self.external_fields = external_fields
        self.operators = operators

        self.mesh = self.device.mesh
        mesh = self.mesh
        self.n_sites = len(mesh.sites)
        self.lsfd_neighbors_amount = mesh.n_lsfd_neighbors

        # Вычисление фиксированного векторного потенциала
        if external_fields.Bz_time_dependent:
            self.operators._use_sparse_delta = False
        else:
            self.operators._use_sparse_delta = True

        self.A_for_constant_Bz = self.external_fields.calculate_fixed_applied_vector_potential(
            x=mesh.sites[:, 0], y=mesh.sites[:, 1]
        )

        # === TERMINALS ===
        if device.terminals:
            self.total_source_lenght = self.device.get_terminal_total_length('source')
            self.total_drain_lenght = self.device.get_terminal_total_length('drain')
            self.boundary_source_indices = self.device.source_site_indices
            self.boundary_drain_indices = self.device.drain_site_indices
            self.boundary_sites = device.mesh.sites[mesh.tri_mesh.boundary_site_indices]

            self.direct_current_amplitude = self.external_fields.calculate_fixed_mu_neuman_boundary_values(
                boundary_sites=self.boundary_sites,
                boundary_ind_source=self.boundary_source_indices,
                boundary_ind_drain=self.boundary_drain_indices,
                total_source_lenght=self.total_source_lenght,
                total_drain_lenght=self.total_drain_lenght,
            )
        else:
            self.total_source_lenght = 0.0
            self.total_drain_lenght = 0.0
            self.boundary_source_indices = np.array([], dtype=np.int64)
            self.boundary_drain_indices = np.array([], dtype=np.int64)
            self.boundary_sites = mesh.sites[mesh.boundary_indices]
            self.direct_current_amplitude = self.external_fields.calculate_fixed_mu_neuman_boundary_values(
                boundary_sites=self.boundary_sites,
                boundary_ind_source=self.boundary_source_indices,
                boundary_ind_drain=self.boundary_drain_indices,
                total_source_lenght=self.total_source_lenght,
                total_drain_lenght=self.total_drain_lenght,
            )
            #self.direct_current_amplitude = np.zeros(len(self.boundary_sites), dtype=np.int32)

        # Вычисление фиксированного направления s
        self.s_applied = self.external_fields.s_constant
        self.s_previous = self.external_fields.s_constant.copy()

        # === ИСТОРИЯ ДЛЯ АДАПТИВНОГО ШАГА ===
        self.d_psi_sq_history = deque(maxlen=options.adaptive_window)

        self.poisson_tolerance = options.poisson_tolerance_init
        self.poisson_iterations_history = deque(maxlen=options.adaptive_window)

    def compute_psi_derivatives(self, psi: np.ndarray, A_applied: np.ndarray,
                                s_applied: np.ndarray, eta: float, gamma: float, Bz: float):
        """Вычисляет все производные psi через LSFD."""
        delta_psi = self.operators.compute_delta_psi(
            psi, A_applied, s_applied=s_applied, eta=eta, gamma=gamma, Bz=Bz
        )
        psi_derivatives = self.operators._batched_dot(
            self.operators.G_matrix_psi_gamma, delta_psi
        )
        return psi_derivatives

    def solve_mu(self, div_J: np.ndarray, J_boundary: np.ndarray,
                 mu_guess: np.ndarray = None,
                 tolerance: float = 1e-5, max_iterations: int = 1000):
        """Решает уравнение Пуассона для μ."""

        # mu = np.zeros_like(div_J)
        # gradients = np.zeros((len(div_J), 2))
        # actual_iters = 0
        # residual = np.zeros_like(div_J)
        #
        mu, gradients, laplacian, achieved_iter_error, actual_iters = self.operators.solve_poisson(
                div_J=div_J,
                I_boundary=J_boundary,
                mu_guess=mu_guess,
                tolerance=tolerance,
                max_iterations=max_iterations
            )

        residual = laplacian - div_J

        return mu, -gradients, residual, actual_iters

    def get_supercurrent(self, psi: np.ndarray, psi_derivatives: np.ndarray,
                         s_applied: np.ndarray, eta: float, gamma: float):
        """Вычисляет сверхток."""
        s_x, s_y = s_applied[0], s_applied[1]

        Dpsi_x = psi_derivatives[:, 0]
        Dpsi_y = psi_derivatives[:, 1]
        laplasian_psi = psi_derivatives[:, 2] + psi_derivatives[:, 3]

        psi_conj = psi.conjugate()
        abs_psi_sq = np.abs(psi)**2

        J0_x = (psi_conj * Dpsi_x).imag + eta * s_x * abs_psi_sq
        J0_y = (psi_conj * Dpsi_y).imag + eta * s_y * abs_psi_sq

        s_grad_psi = s_x * Dpsi_x + s_y * Dpsi_y

        if gamma == 0:
            return J0_x, J0_y, s_grad_psi

        Dpsi_xx = psi_derivatives[:, 2]
        Dpsi_yy = psi_derivatives[:, 3]
        Dpsi_xy = psi_derivatives[:, 4]

        J1_x = gamma * Dpsi_x.conjugate() * s_grad_psi
        J1_y = gamma * Dpsi_y.conjugate() * s_grad_psi

        J2_x = -gamma * psi_conj * (s_x * Dpsi_xx + s_y * Dpsi_xy)
        J2_y = -gamma * psi_conj * (s_x * Dpsi_xy + s_y * Dpsi_yy)

        J3_x = -gamma * s_x * (psi_conj * laplasian_psi)
        J3_y = -gamma * s_y * (psi_conj * laplasian_psi)

        J_x = J0_x + J1_x.real + J2_x.real + J3_x.real
        J_y = J0_y + J1_y.real + J2_y.real + J3_y.real

        return J_x, J_y, s_grad_psi

    def compute_divergence_J(self, psi: np.ndarray, psi_derivatives: np.ndarray,
                             s_grad_psi: np.ndarray, s_applied: np.ndarray,
                             eta: float, gamma: float):
        """Вычисляет дивергенцию сверхтока."""
        psi_conj = np.conj(psi)
        s_x, s_y = s_applied[0], s_applied[1]

        Dpsi_x = psi_derivatives[:, 0]
        Dpsi_y = psi_derivatives[:, 1]
        Dpsi_xx = psi_derivatives[:, 2]
        Dpsi_yy = psi_derivatives[:, 3]
        Dpsi_xxx = psi_derivatives[:, 5]
        Dpsi_yyy = psi_derivatives[:, 6]
        Dpsi_xxy = psi_derivatives[:, 7]
        Dpsi_yyx = psi_derivatives[:, 8]

        div_J0 = (psi_conj * (Dpsi_xx + Dpsi_yy)).imag
        divJ_eta = eta * (psi_conj * s_grad_psi + psi * s_grad_psi.conjugate())

        if gamma == 0:
            return div_J0 + divJ_eta

        divJ_gamma = -2 * gamma * (
                psi_conj * s_x * (Dpsi_xxx + Dpsi_yyx) +
                psi_conj * s_y * (Dpsi_yyy + Dpsi_xxy)
        ).real

        return div_J0 + divJ_eta + divJ_gamma

    def solve_for_psi_squared(self, psi: np.ndarray, psi_derivatives: np.ndarray,
                              abs_sq_psi: np.ndarray, mu: np.ndarray,
                              s_applied: np.ndarray, Bz: float,
                              eta: float, gamma: float, dt: float, u=5.79):
        """Решает TDGL уравнение для psi."""
        s_x, s_y = s_applied[0], s_applied[1]
        U = np.cos(mu * dt) - 1j * np.sin(mu * dt)

        Dpsi_x = psi_derivatives[:, 0]
        Dpsi_y = psi_derivatives[:, 1]
        Dpsi_xx = psi_derivatives[:, 2]
        Dpsi_yy = psi_derivatives[:, 3]
        Dpsi_xxx = psi_derivatives[:, 5]
        Dpsi_yyy = psi_derivatives[:, 6]
        Dpsi_xxy = psi_derivatives[:, 7]
        Dpsi_yyx = psi_derivatives[:, 8]

        if gamma == 0:
            psi = U * (psi + (dt / u) * (
                    psi * (1 - abs_sq_psi) +
                    Dpsi_xx + Dpsi_yy +
                    2 * eta * 1j * (s_x * Dpsi_x + s_y * Dpsi_y)
            ))
        else:
            psi = U * (psi + (dt / u) * (
                    psi * (1 - abs_sq_psi) +
                    Dpsi_xx + Dpsi_yy +
                    2 * eta * 1j * (s_x * Dpsi_x + s_y * Dpsi_y) +
                    2 * gamma * 1j * s_x * (Dpsi_xxx + Dpsi_yyx + 1j * Bz * Dpsi_y) +
                    2 * gamma * 1j * s_y * (Dpsi_yyy + Dpsi_xxy - 1j * Bz * Dpsi_x)
            ))

        new_sq_psi = np.absolute(psi) ** 2
        return psi, new_sq_psi


    def solve_for_one_step(self, psi: np.ndarray, psi_abs_sq: np.ndarray,
                           mu: np.ndarray, t: float, dt: float) -> StepResult:
        """
        Один шаг TDGL динамики с адаптивным шагом по времени.
        """
        options = self.options

        # 1) Get external fields
        A_applied, Bz = self.external_fields.update_vector_potential(t)
        J_boundary = self.external_fields.update_mu_boundary(t)
        eta, gamma = self.external_fields.get_ferromagnetic(t)
        s_applied = self.external_fields.update_s_direction(t)

        # Обновление G при вращении s (если нужно)
        if gamma != 0:
            angle_change = np.arccos(np.clip(np.dot(s_applied, self.s_previous), -1, 1))
            if angle_change >= options.update_G_angle_threshold:
                self.operators.update_G_matrix_psi_gamma(s_direction=s_applied)
                self.s_previous = s_applied.copy()

        # 2) Compute psi_derivatives
        psi_derivatives = self.compute_psi_derivatives(
            psi, A_applied, s_applied=s_applied, eta=eta, gamma=gamma, Bz=Bz
        )

        # 3) Solve TDGL equation for psi
        psi, new_psi_abs_sq = self.solve_for_psi_squared(
            psi=psi, psi_derivatives=psi_derivatives,
            abs_sq_psi=psi_abs_sq, mu=mu,
            dt=dt, gamma=gamma, eta=eta, s_applied=s_applied, Bz=Bz
        )

        # 3.5) Пересчитать производные для НОВОГО psi
        psi_derivatives = self.compute_psi_derivatives(
            psi, A_applied, s_applied=s_applied, eta=eta, gamma=gamma, Bz=Bz
        )

        # 4) Get supercurrent
        supercurrent_x, supercurrent_y, s_grad_psi = self.get_supercurrent(
            psi=psi, psi_derivatives=psi_derivatives,
            s_applied=s_applied, eta=eta, gamma=gamma
        )

        # 5) Get divergence J
        div_Js = self.compute_divergence_J(
            psi=psi, psi_derivatives=psi_derivatives,
            s_grad_psi=s_grad_psi, s_applied=s_applied, eta=eta, gamma=gamma
        )

        div_Js = np.real(div_Js)

        # 6) Solve Poisson equation for mu
        mu_new, normal_current, poisson_residual, poisson_iterations = self.solve_mu(
            div_J=div_Js,
            J_boundary=J_boundary,
            mu_guess=mu,
            tolerance=self.poisson_tolerance,
            max_iterations=options.poisson_iterations
        )

        if options.poisson_adaptive:

            self.poisson_iterations_history.append(poisson_iterations)
            # Адаптация tolerance (каждые adaptive_window шагов)
            if len(self.poisson_iterations_history) == options.adaptive_window:
                avg_iterations = np.mean(self.poisson_iterations_history)

                # Логика:
                # - Если avg_iterations < 10: система стабильна → ужесточить tolerance (к min)
                # - Если avg_iterations > 100: система меняется быстро → ослабить tolerance (к max)

                if avg_iterations < 10:
                    new_tolerance = max(self.poisson_tolerance / 2, options.poisson_tolerance_min)
                elif avg_iterations > 100:
                    new_tolerance = min(self.poisson_tolerance * 2, options.poisson_tolerance_max)
                else:
                    new_tolerance = self.poisson_tolerance

                # Плавное изменение
                self.poisson_tolerance = 0.5 * (self.poisson_tolerance + new_tolerance)

                # Сбрасываем историю
                self.poisson_iterations_history.clear()

        # 7) АДАПТИВНЫЙ ШАГ
        new_dt = dt  # По умолчанию оставляем текущий

        if options.time_scheme == TimeScheme.ADAPTIVE_EULER:
            # Вычисляем изменение |ψ|²
            delta = float(np.max(np.abs(new_psi_abs_sq - psi_abs_sq)))
            self.d_psi_sq_history.append(delta)

            # Если окно заполнено — вычисляем новый dt
            if len(self.d_psi_sq_history) == options.adaptive_window:
                delta_n = np.mean(self.d_psi_sq_history)

                # Формула из pyTDGL: dt_new = dt_init / delta_n
                # Защита от деления на ноль
                delta_n = max(delta_n, 1e-15)
                dt_candidate = options.dt_init / delta_n

                # Сглаживание: среднее между текущим и кандидатом
                dt_smooth = 0.5 * (dt_candidate + dt)

                # Ограничение диапазона
                new_dt = max(options.dt_min, min(dt_smooth, options.dt_max))

        return StepResult(
            psi=psi,
            psi_abs_sq=new_psi_abs_sq,
            psi_derivatives=psi_derivatives,
            mu=mu_new,
            supercurrent_x=supercurrent_x,
            supercurrent_y=supercurrent_y,
            div_Js=div_Js,
            normal_current=normal_current,
            dt=new_dt,
            poisson_tolerance=self.poisson_tolerance,
            poisson_residual=poisson_residual,
            poisson_iterations=poisson_iterations,
            s_applied=s_applied,
            Bz=Bz,
            eta=eta,
            gamma=gamma,
        )
