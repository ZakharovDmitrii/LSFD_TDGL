from typing import Optional, Tuple, Dict, Callable, Union, Literal
import numpy as np

from typing import NamedTuple

class StepFields(NamedTuple):
    """Все внешние поля на одном временном слое."""
    A_applied: np.ndarray   # (N, 2)
    Bz: float
    J_boundary: np.ndarray  # (N_b,)
    eta: float
    gamma: float
    s_applied: np.ndarray   # (2,)

class  ExternalFields:
    """
    Внешние источники для TDGL решателя (безразмерные).

    Поддерживает:
    1. Перпендикулярное магнитное поле Bz: Bz = B0 × B_time(w × t)
    2. Ток на терминалах (линейная плотность, распределяется по длине): I = I0 × I_time(w × t)
    3. Параметры ферромагнетика (eta, gamma), h_direction - направление намагниченности,
                                              h_rotation_frequency - частота поворота вектора
    """

    def __init__(
        self,
        # === Магнитное поле: Bz = B0 × B_time(w × t) ===
        Bz_amplitude: float = 0.0,
        Bz_frequency: float = 0.0,
        Bz_time_func: Optional[Callable[[float], float]] = None,
        Bz_start_time: float = 0.0,
        Bz_duration: float = np.inf,

        # === Ток: I = I0 × I_time(w × t) ===
        I_amplitude_source: float = 0.0,
        I_frequency: float = 0.0,
        I_time_func: Optional[Callable[[float], float]] = None,
        I_start_time: float = 0.0,
        I_duration: float = np.inf,

        # === Ферромагнетик ===
        eta: float = 0.0,
        gamma: float = 0.0,
        ferromagnetic_start_time: float = 0.0,
        ferromagnetic_duration: float = np.inf,
        h_direction: Tuple[float, float] = (1.0, 0.0),  # ← Начальное направление
        h_rotation_frequency: float = 0.0,  # w [1/τ₀]
        h_rotation_direction: Literal["cw", "ccw"] = "ccw",
        h_rotation_start_time: float = 0.0,  # ← НОВОЕ: когда начинать поворот
        h_rotation_duration: float = np.inf,  # ← НОВОЕ: длительность поворота (0 = мгновенно)
    ):

        # ----------------------------------------------------------------
        # 1. Магнитное поле
        # ----------------------------------------------------------------
        self.Bz_amplitude = Bz_amplitude
        self.Bz_frequency = Bz_frequency
        self.Bz_time_func = Bz_time_func
        self.Bz_time_dependent = True if Bz_time_func is not None else False

        self.Bz_fixed = Bz_amplitude

        self.Bz_start_time = Bz_start_time
        self.Bz_duration = Bz_duration

        if Bz_duration == np.inf:
            self.Bz_end_time = np.inf
        else:
            self.Bz_end_time = Bz_start_time + Bz_duration

        if not self.Bz_time_dependent:
            self.update_vector_potential = self._update_A_constant
        elif self.Bz_duration != np.inf:
            self.update_vector_potential = self._update_A_ramp
        else:
            self.update_vector_potential = self._update_A

        self.A_fixed: Optional[np.ndarray] = None


        # ----------------------------------------------------------------
        # 2. Ток
        # ----------------------------------------------------------------
        self.I_amplitude = I_amplitude_source
        self.I_frequency = I_frequency
        self.I_time_func = I_time_func
        self.I_time_dependent = True if I_time_func is not None else False

        self.I_start_time = I_start_time
        self.I_duration = I_duration

        if I_duration == np.inf:
            self.I_end_time = np.inf
        else:
            self.I_end_time = I_start_time + I_duration

        if not self.I_time_dependent:
            self.update_mu_boundary = self._update_nabla_mu_boundary_constant
        elif self.I_duration != np.inf:  # ← НОВОЕ УСЛОВИЕ!
            self.update_mu_boundary = self._update_nabla_mu_boundary_ramp
        else:
            self.update_mu_boundary = self._update_nabla_mu_boundary

        self.J_fixed: Optional[np.ndarray] = None

        # ----------------------------------------------------------------
        # 3. Ферромагнетик
        # ----------------------------------------------------------------
        self.eta = eta
        self.gamma = gamma

        self.ferromagnetic_start_time = ferromagnetic_start_time
        self.ferromagnetic_duration = ferromagnetic_duration

        if ferromagnetic_duration == np.inf:
            self.ferromagnetic_end_time = np.inf
        else:
            self.ferromagnetic_end_time = ferromagnetic_start_time + ferromagnetic_duration

        # ← Метод для получения eta(t) и gamma(t)
        if ferromagnetic_duration != np.inf:
            self.get_ferromagnetic = self.get_ferromagnetic_time_dependent
        else:
            self.get_ferromagnetic = self.get_ferromagnetic_constant


        # === НОРМАЛИЗАЦИЯ h_direction (сразу в __init__) ===
        h = np.asarray(h_direction, dtype=float)
        norm = np.linalg.norm(h)

        if norm > 0:
            self.h_direction = h / norm  # ← Нормализуем
        else:
            # Если передан нулевой вектор — ставим по умолчанию (1, 0)
            self.h_direction = np.array([1.0, 0.0])

        # === Параметры вращения ===
        self.h_rotation_start_time = h_rotation_start_time
        self.h_rotation_duration = h_rotation_duration
        self.h_rotation_frequency = h_rotation_frequency
        # Направление: +1 = против часовой (ccw), -1 = по часовой (cw)
        self.h_rotation_direction = 1.0 if h_rotation_direction == "ccw" else -1.0

        self.s_constant = np.array([ - self.h_direction[1], self.h_direction[0]])

        if h_rotation_duration == np.inf:
            self.h_rotation_end_time = np.inf  # ← Явно задаём!
        else:
            self.h_rotation_end_time = h_rotation_start_time + h_rotation_duration

        if h_rotation_frequency == 0:
            self.update_s_direction = self._update_s_constant
        else:
            self.update_s_direction = self._update_s_rotation

    # === Applied vector potential for uniform Bz ===

    def calculate_fixed_applied_vector_potential(self, x: np.ndarray, y:np.ndarray) -> np.ndarray:
        fixed_A_x = - y * self.Bz_amplitude / 2
        fixed_A_y = x * self.Bz_amplitude / 2
        fixed_A = np.column_stack([fixed_A_x, fixed_A_y])
        self.A_fixed = fixed_A
        return fixed_A

    def _update_A_constant(self, t: float):
        """Режим 1: CONSTANT — просто вернуть fixed_A."""

        return self.A_fixed, self.Bz_fixed

    def _update_A(self, t: float):
        """Режим 2: PERIODIC — вычислять на каждом шаге."""
        x = self.Bz_frequency * t
        time_factor = self.Bz_time_func(x)
        return time_factor * self.A_fixed, time_factor * self.Bz_fixed

    def _update_A_ramp(self, t: float):
        """Режим: RAMP с start_time и duration."""

        # До начала
        if t < self.Bz_start_time:
            return self.A_fixed * 0.0, self.Bz_fixed * 0.0
        # После завершения → переключаем на constant
        if t >= self.Bz_end_time:
            self.update_vector_potential = self._update_A_constant
            print(f"[ExternalFields] Bz: t={t:.2f} >= t_end={self.Bz_end_time:.2f}")

            # Возвращаем финальное значение
            x = self.Bz_frequency * self.Bz_duration
            time_factor = self.Bz_time_func(x)
            self.A_fixed = time_factor * self.A_fixed
            self.Bz_fixed = time_factor * self.Bz_fixed
            return self.A_fixed, self.Bz_fixed

        # Во время изменения
        t_rel = t - self.Bz_start_time
        x = self.Bz_frequency * t_rel
        time_factor = self.Bz_time_func(x)
        return time_factor * self.A_fixed, time_factor * self.Bz_fixed

    # === Boundary J for \nabla mu in Poisson equation ===

    def calculate_fixed_mu_neuman_boundary_values(self, boundary_sites: np.ndarray,
                          boundary_ind_source: np.ndarray, boundary_ind_drain: np.ndarray,
                          total_source_lenght: float, total_drain_lenght: float) -> np.ndarray:
        mu_neuman_boundary_values = np.zeros_like(boundary_sites[:,0])
        if total_source_lenght != 0.0:
            mu_neuman_boundary_values[boundary_ind_source] = - self.I_amplitude / total_source_lenght
            mu_neuman_boundary_values[boundary_ind_drain] = self.I_amplitude / total_drain_lenght
        self.J_fixed = mu_neuman_boundary_values
        return mu_neuman_boundary_values

    def _update_nabla_mu_boundary_constant(self, t: float) -> np.ndarray:
        """Режим 1: CONSTANT — просто вернуть fixed_A."""
        return self.J_fixed

    def _update_nabla_mu_boundary(self, t: float) -> np.ndarray:
        """Режим 2: PERIODIC — вычислять на каждом шаге."""
        x = self.I_frequency * t
        time_factor = self.I_time_func(x)
        return time_factor * self.J_fixed

    def _update_nabla_mu_boundary_ramp(self, t: float) -> np.ndarray:
        """Режим 3: RAMP — до t_sat вычислять, после — переключиться на constant."""

        # До начала
        if t < self.I_start_time:
            return self.J_fixed * 0.0
            # После завершения → переключаем на constant
        if t >= self.I_end_time:
            self.update_mu_boundary= self._update_nabla_mu_boundary_constant
            print(f"[ExternalFields] I: t={t:.2f} >= t_end={self.I_end_time:.2f}")
            # Возвращаем финальное значение
            x = self.I_frequency * self.I_duration
            time_factor = self.I_time_func(x)
            self.J_fixed = time_factor * self.J_fixed
            return self.J_fixed

        # Во время изменения
        t_rel = t - self.I_start_time
        x = self.I_frequency * t_rel
        time_factor =self.I_time_func(x)
        return time_factor * self.J_fixed

    # === Selected direction s = [n × h] induced by ferromagnetic substrate===

    def get_ferromagnetic_constant(self, t: float) -> Tuple[float, float]:
        """Ферромагнетик всегда ВКЛ."""
        return self.eta, self.gamma

    def get_ferromagnetic_time_dependent(self, t: float) -> Tuple[float, float]:
        """Ферромагнетик ВКЛ/ВЫКЛ по времени."""
        if t < self.ferromagnetic_start_time:
            return 0.0, 0.0  # ← ВЫКЛ
        elif t >= self.ferromagnetic_end_time:
            return 0.0, 0.0  # ← ВЫКЛ после завершения
        else:
            return self.eta, self.gamma  # ← ВКЛ


    def _update_s_constant(self, t: float) -> np.ndarray:
        return self.s_constant

    def rotate_h_fixed_angle(self,h_rotation_fixed_angle: float,h_rotation_direction: Literal["cw", "ccw"],):

        h_rotation_direction = 1.0 if h_rotation_direction == "ccw" else -1.0
        phi = h_rotation_fixed_angle * h_rotation_direction
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        h_x = cos_phi * self.h_direction[0] - sin_phi * self.h_direction[1]
        h_y = sin_phi * self.h_direction[0] + cos_phi * self.h_direction[1]

        self.h_direction = np.array([h_x, h_y])
        self.s_constant = np.array([-h_y, h_x])

        return self.s_constant

    def _update_s_rotation(self, t: float) -> np.ndarray:

        if t < self.h_rotation_start_time:
            return self.s_constant
        elif t > self.h_rotation_end_time:
            self.update_s_direction = self._update_s_constant
            # ← ВЫЧИСЛЯЕМ финальное состояние и обновляем h_direction!
            phi = self.h_rotation_frequency * (self.h_rotation_end_time - self.h_rotation_start_time) * self.h_rotation_direction
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            h_x = cos_phi * self.h_direction[0] - sin_phi * self.h_direction[1]
            h_y = sin_phi * self.h_direction[0] + cos_phi * self.h_direction[1]

            self.h_direction = np.array([h_x, h_y])  # ← Обновляем!
            self.s_constant = np.array([-h_y, h_x])

            print(f"[ExternalFields] h: rotation complete at t={t:.2f}")
            return self.s_constant
        else:
            phi = self.h_rotation_frequency * (t - self.h_rotation_start_time) * self.h_rotation_direction
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
            h_x = cos_phi * self.h_direction[0] - sin_phi * self.h_direction[1]
            h_y = sin_phi * self.h_direction[0] + cos_phi * self.h_direction[1]
            self.s_constant = np.array([-h_y, h_x])

            return self.s_constant

    def get_fields_at(self, t: float) -> StepFields:
        """Собрать все поля на момент t одним вызовом."""
        A_applied, Bz = self.update_vector_potential(t)
        J_boundary = self.update_mu_boundary(t)
        eta, gamma = self.get_ferromagnetic(t)
        s_applied = self.update_s_direction(t)
        return StepFields(
            A_applied=A_applied, Bz=Bz, J_boundary=J_boundary,
            eta=eta, gamma=gamma, s_applied=s_applied,
        )