import numpy as np
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import time
import scipy.sparse as sp
from collections import deque

from ..device.device import Device
from ..external_fields.external_fields import ExternalFields
from ..operators.operators import LSFD_operators
from .dynamics_options import SolverOptions, TimeScheme
from ..operators.fvm_integrator import FVMIntegrator


# ============================================================================
# КОНТЕЙНЕР ДЛЯ РЕЗУЛЬТАТОВ ОДНОГО ШАГА
# ============================================================================
class StepResult(NamedTuple):
    """
    Контейнер для результатов одного шага TDGL симуляции.

    Основные поля (всегда возвращаются):
        psi: Волновая функция (N,) complex128
        psi_abs_sq: |ψ|² (N,) float64
        mu: Скалярный потенциал (N,) float64
        supercurrent_x: Сверхток по x (N,) float64
        supercurrent_y: Сверхток по y (N,) float64
        div_Js: Дивергенция сверхтока (N,) float64
        normal_current: Нормальный ток -∇μ (N, 2) float64
        dt: Текущий шаг по времени

    Опциональные поля (только если включены соответствующие флаги):
        psi_derivatives: Производные psi (N, 14) complex128
        poisson_residual: Невязка уравнения Пуассона (N,) float64
        poisson_iterations: Число итераций солвера Пуассона
        energy_voronoi: Энергия ГЛ (метод Вороного) float
        energy_triangles: Энергия ГЛ (метод треугольников) float
        conservation_global: Глобальная проверка сохранения tuple
        conservation_local: Локальная проверка сохранения tuple
    """
    # === Основные поля (всегда) ===
    psi: np.ndarray  # (N,) complex128
    psi_abs_sq: np.ndarray  # (N,) float64
    mu: np.ndarray  # (N,) float64
    supercurrent_x: np.ndarray  # (N,) float64
    supercurrent_y: np.ndarray  # (N,) float64
    div_Js: np.ndarray  # (N,) float64
    normal_current: np.ndarray  # (N, 2) float64
    dt: float  # scalar

    # === Опциональные поля ===
    psi_derivatives: Optional[np.ndarray] = None  # (N, 14) complex128
    poisson_residual: Optional[np.ndarray] = None  # (N,) float64
    poisson_iterations: Optional[int] = None  # scalar
    energy_voronoi: Optional[float] = None  # scalar
    energy_triangles: Optional[float] = None  # scalar
    conservation_global: Optional[Tuple] = None
    conservation_local: Optional[Tuple] = None


# ============================================================================
# TDGL SOLVER
# ============================================================================

class TDGLSolver:

    def __init__(
            self,
            device: Device,
            operators: LSFD_operators,
            external_fields: ExternalFields,
            options: SolverOptions = None,
    ):
        self.current_step = 0  # ← ДОБАВИТЬ ЭТУ СТРОКУ
        # === ВАЛИДАЦИЯ OPTIONS ===
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

        # === ТЕРМИНАЛЫ ===
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

        # === ИСТОРИЯ ДЛЯ АДАПТИВНОГО ШАГА ===
        self.d_psi_sq_history = deque(maxlen=options.adaptive_window)
        self.s_previous = self.external_fields.s_constant.copy()

        # === FVM INTEGRATOR ДЛЯ ПРОВЕРОК ===
        self.fvm = None
        if options.check_conservation or options.check_energy:
            self.fvm = FVMIntegrator(mesh=mesh)

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
                 tolerance: float = 1e-4, max_iterations: int = 1000):
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

    def compute_GL_energy(self, psi: np.ndarray, psi_derivatives: np.ndarray,
                          s_applied: np.ndarray, Bz: float,
                          eta: float, gamma: float) -> Tuple[float, float]:
        """
        Вычисляет полную энергию Гинзбурга-Ландау двумя методами.

        F = ∫ [-|ψ|² + |ψ|⁴/2 + |∇ψ|² + η·s·(2·Re(∇ψ)) + B²] dV

        Args:
            psi: (N,) — волновая функция
            psi_derivatives: (N, 14) — производные psi
            s_applied: (2,) — направление ферромагнетика
            Bz: float — магнитное поле
            eta: float — параметр ферромагнетика
            gamma: float — кубическая поправка

        Returns:
            (F_voronoi, F_triangles) — энергия, вычисленная двумя методами
        """
        s_x, s_y = s_applied[0], s_applied[1]

        # Плотность энергии в каждой вершине
        sq_psi = np.real(psi * psi.conjugate())
        sq_Dx_psi = np.real(psi_derivatives[:, 0] * psi_derivatives[:, 0].conjugate())
        sq_Dy_psi = np.real(psi_derivatives[:, 1] * psi_derivatives[:, 1].conjugate())

        s_grad_psi = (s_x * (psi_derivatives[:, 0] + psi_derivatives[:, 0].conjugate()) +
                      s_y * (psi_derivatives[:, 1] + psi_derivatives[:, 1].conjugate()))

        F_density = (-sq_psi + 0.5 * sq_psi ** 2 +
                     sq_Dx_psi + sq_Dy_psi +
                     eta * np.real(s_grad_psi) +
                     Bz ** 2)

        # Интегрирование двумя методами через FVMIntegrator
        F_voronoi = self.fvm.compute_divergence_integral(F_density, method='voronoi')
        F_triangles = self.fvm.compute_divergence_integral(F_density, method='triangles')

        return float(F_voronoi), float(F_triangles)

    # solver.py
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
            div_J=div_Js, J_boundary=J_boundary, mu_guess=mu
        )

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

                # # Логирование изменения dt (каждые 100 шагов)
                # if self.current_step % 100 == 0:
                #     print(f"  [Адаптивный шаг] dt: {dt:.3e} → {new_dt:.3e}, "
                #           f"delta_n: {delta_n:.3e}")

        # 8) Опциональные вычисления
        energy_voronoi = None
        energy_triangles = None
        conservation_global = None

        if options.check_energy and self.fvm is not None:
            energy_voronoi, energy_triangles = self.compute_GL_energy(
                psi=psi, psi_derivatives=psi_derivatives,
                s_applied=s_applied, Bz=Bz, eta=eta, gamma=gamma
            )

        if options.check_conservation and self.fvm is not None:
            conservation_global = self.fvm.global_conservation_check(
                J_x=supercurrent_x.real, J_y=supercurrent_y.real, div_J=div_Js.real
            )

        # 9) Собираем результат
        self.current_step += 1

        # В конце solve_for_one_step, перед return:

        # Вывод conservation каждые 1000 шагов
        if self.options.check_conservation and self.current_step % 1000 == 0:
            if conservation_global is not None:
                surface_edges, surface_sites, div_voronoi, div_triangles = conservation_global
                print(f"\n[Шаг {self.current_step:5d}] Conservation check:")
                print(f"  ∮J·dl(edges)  = {surface_edges:+.3e}")
                print(f"  J·dl(sites)  = {surface_sites:+.3e}")
                print(f"  div(J)vor    = {div_voronoi:+.3e}")
                print(f"  ∫div(J)tri    = {div_triangles:+.3e}")
                print(f"  Разница       = {abs(surface_edges - div_voronoi):.3e}")

                # Диагностика
                Jn_super = (supercurrent_x[self.operators.boundary_indices] * self.operators.normal_vecs[:, 0] +
                            supercurrent_y[self.operators.boundary_indices] * self.operators.normal_vecs[:, 1])

                print(f"[Шаг {self.current_step}] J_s·n на границе:")
                print(f"  max|J_s·n| = {np.max(np.abs(Jn_super)):.3e}")
                print(f"  mean|J_s·n| = {np.mean(np.abs(Jn_super)):.3e}")
                print('Max psi: ', np.round(np.max(np.abs(psi)),3))
                print('Min psi: ', np.round(np.min(np.abs(psi)), 3))
                print('Максимальный нормальный ток: ', np.max(np.abs(normal_current)))
                print('Минимальный сверхток: ', np.min( np.sqrt(supercurrent_x**2 + supercurrent_y**2)))
                print('Максимальная разность потенциала: ', np.max(mu - np.min(mu)))
                print('Максимальная разность дивергенции: ', np.max(div_Js - np.min(div_Js)))

        return StepResult(
            psi=psi,
            psi_abs_sq=new_psi_abs_sq,
            mu=mu_new,
            supercurrent_x=supercurrent_x,
            supercurrent_y=supercurrent_y,
            div_Js=div_Js,
            normal_current=normal_current,
            dt=new_dt,  # ← Возвращаем новый dt!
            psi_derivatives=psi_derivatives if options.check_energy else None,
            poisson_residual=poisson_residual,
            poisson_iterations=poisson_iterations,
            energy_voronoi=energy_voronoi,
            energy_triangles=energy_triangles,
            conservation_global=conservation_global,
            conservation_local=None,
        )

    # def solve_for_one_step(self, psi: np.ndarray, psi_abs_sq: np.ndarray,
    #                        mu: np.ndarray, t: float, dt: float) -> StepResult:
    #     """
    #     Один шаг TDGL динамики.
    #
    #     Returns:
    #         StepResult с основными и опциональными полями
    #     """
    #     options = self.options
    #     import time
    #
    #     # Профилирование
    #     timings = {}
    #     t_total_start = time.perf_counter()
    #
    #     # 1) Get external fields
    #     t0 = time.perf_counter()
    #     A_applied, Bz = self.external_fields.update_vector_potential(t)
    #     J_boundary = self.external_fields.update_mu_boundary(t)
    #     eta, gamma = self.external_fields.get_ferromagnetic(t)
    #     s_applied = self.external_fields.update_s_direction(t)
    #     timings['external_fields'] = time.perf_counter() - t0
    #
    #     # Обновление G при вращении s (если нужно)
    #     if gamma != 0:
    #         angle_change = np.arccos(np.clip(np.dot(s_applied, self.s_previous), -1, 1))
    #         if angle_change >= options.update_G_angle_threshold:
    #             self.operators.update_G_matrix_psi_gamma(s_direction=s_applied)
    #             self.s_previous = s_applied.copy()
    #
    #     # 2) Compute psi_derivatives
    #     t0 = time.perf_counter()
    #     psi_derivatives = self.compute_psi_derivatives(
    #         psi, A_applied, s_applied=s_applied, eta=eta, gamma=gamma, Bz=Bz
    #     )
    #     timings['psi_derivatives'] = time.perf_counter() - t0
    #
    #     # 3) Solve TDGL equation for psi
    #     t0 = time.perf_counter()
    #     psi, new_psi_abs_sq = self.solve_for_psi_squared(
    #         psi=psi, psi_derivatives=psi_derivatives,
    #         abs_sq_psi=psi_abs_sq, mu=mu,
    #         dt=dt, gamma=gamma, eta=eta, s_applied=s_applied, Bz=Bz
    #     )
    #     timings['solve_psi'] = time.perf_counter() - t0
    #
    #     # 4) Get supercurrent
    #     t0 = time.perf_counter()
    #     supercurrent_x, supercurrent_y, s_grad_psi = self.get_supercurrent(
    #         psi=psi, psi_derivatives=psi_derivatives,
    #         s_applied=s_applied, eta=eta, gamma=gamma
    #     )
    #     timings['supercurrent'] = time.perf_counter() - t0
    #
    #     # 5) Get divergence J
    #     t0 = time.perf_counter()
    #     div_Js = self.compute_divergence_J(
    #         psi=psi, psi_derivatives=psi_derivatives,
    #         s_grad_psi=s_grad_psi, s_applied=s_applied, eta=eta, gamma=gamma
    #     )
    #     div_Js = np.real(div_Js)
    #     timings['divergence_J'] = time.perf_counter() - t0
    #
    #     # 6) Solve Poisson equation for mu
    #     t0 = time.perf_counter()
    #     mu_new, normal_current, poisson_residual, poisson_iterations = self.solve_mu(
    #         div_J=div_Js, J_boundary=J_boundary, mu_guess=mu
    #     )
    #     timings['solve_poisson'] = time.perf_counter() - t0
    #     timings['poisson_iterations'] = poisson_iterations
    #
    #
    #     # 7) Адаптивный шаг (если включён)
    #     new_dt = dt
    #     if options.time_scheme == TimeScheme.ADAPTIVE_EULER:
    #         delta = float(np.max(np.abs(new_psi_abs_sq - psi_abs_sq)))
    #         self.d_psi_sq_history.append(delta)
    #
    #         if len(self.d_psi_sq_history) == options.adaptive_window:
    #             delta_n = np.mean(self.d_psi_sq_history)
    #             new_dt = options.dt_init / max(1e-10, delta_n)
    #             new_dt = max(0.0, min(0.5 * (new_dt + dt), options.dt_max))
    #
    #     # 8) Опциональные вычисления
    #     t0 = time.perf_counter()
    #     energy_voronoi = None
    #     energy_triangles = None
    #     conservation_global = None
    #     conservation_local = None
    #
    #     # Энергия ГЛ (двумя методами)
    #     if options.check_energy and self.fvm is not None:
    #         energy_voronoi, energy_triangles = self.compute_GL_energy(
    #             psi=psi, psi_derivatives=psi_derivatives,
    #             s_applied=s_applied, Bz=Bz, eta=eta, gamma=gamma
    #         )
    #
    #     timings['energy'] = time.perf_counter() - t0
    #     t0 = time.perf_counter()
    #     # Проверка сохранения (4 значения!)
    #     if options.check_conservation and self.fvm is not None:
    #         conservation_global = self.fvm.global_conservation_check(
    #             J_x=supercurrent_x.real,
    #             J_y=supercurrent_y.real,
    #             div_J=div_Js.real
    #         )
    #     timings['conservation'] = time.perf_counter() - t0
    #     timings['total'] = time.perf_counter() - t_total_start
    #
    #     # Печать каждые 5 шагов
    #     if self.current_step % 1000 == 0:
    #         print(f"\n=== Профилирование (шаг {self.current_step}) ===")
    #         for key, val in timings.items():
    #             if key != 'poisson_iterations':
    #                 print(f"  {key:20s}: {val * 1000:8.2f} мс")
    #         print(f"  {'poisson_iterations':20s}: {timings['poisson_iterations']} итераций")
    #
    #     self.current_step += 1
    #
    #
    #     # 9) Собираем результат
    #     return StepResult(
    #         psi=psi,
    #         psi_abs_sq=new_psi_abs_sq,
    #         mu=mu_new,
    #         supercurrent_x=supercurrent_x,
    #         supercurrent_y=supercurrent_y,
    #         div_Js=div_Js,
    #         normal_current=normal_current,
    #         dt=new_dt,
    #         psi_derivatives=psi_derivatives if options.check_energy else None,
    #         poisson_residual=poisson_residual,
    #         poisson_iterations=poisson_iterations,
    #         energy_voronoi=energy_voronoi,
    #         energy_triangles=energy_triangles,
    #         conservation_global=conservation_global,
    #         conservation_local=conservation_local,
    #     )
