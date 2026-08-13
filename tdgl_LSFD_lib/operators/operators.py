import warnings
from typing import Callable, Tuple, Union
import time
import itertools as it
import numpy as np
import scipy.sparse as sp
from scipy.misc import derivative
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import factorized, spsolve, splu
from scipy.sparse import lil_matrix, bmat, csc_matrix, coo_matrix
from collections import defaultdict
from scipy import interpolate
from scipy.spatial import KDTree
from scipy.sparse.linalg import lsqr
from .numba_kernels import get_batched_dot

from tdgl_LSFD_lib.mesh.mesh import Mesh

# ----------------------------------------------------------------
# 1. LSFD_psi_operators
# ----------------------------------------------------------------

class LSFD_operators:
    def __init__(
            self,
            mesh: Mesh,
            s_direction: np.ndarray,
            use_ghost_points: bool = False,
            ghost_version: str = 'sym', # 'asym'
            check_condition_number: bool = False,  # ← НОВЫЙ ПАРАМЕТР
            weight_function: str = 'exp',
            use_numba: bool = False,  # ← НОВОЕ
            num_threads: int = 4,  # ← НОВОЕ (по умолчанию 4)
    ):

        self.mesh = mesh
        self.sites = mesh.sites
        self.voronoi_areas = self.mesh.dual_mesh.dual_areas
        self.n_sites = len(self.sites)
        self.boundary_indices = mesh.boundary_indices
        self.lsfd_neighbors_amount = mesh.n_lsfd_neighbors
        self.normal_vecs = mesh.tri_mesh.boundary_site_normals
        self.normal_vecs_length = 10**(-5) # 10**(-5) for 1r
                                                # use as weight for boundary conditions and Poisson equation,
                                                 # depends on weight_function

        self.use_ghost_points = use_ghost_points
        self.ghost_version = ghost_version

        if self.use_ghost_points == True:
            self.nb_indices = mesh.lsfd_neighbors.indices_with_ghosts  # (N, K) -> индексы в sites_with_ghosts
            self.nb_coords = mesh.lsfd_neighbors.coords_with_ghosts  # (N, K, 2)
            self.nb_dist = mesh.lsfd_neighbors.distances_with_ghosts
            # ← Предвычисленные векторы из mesh
            self.nb_edge_vectors = mesh.lsfd_neighbors.lsfd_edge_vectors_with_ghosts  # (N, K, 2)
            self.nb_edge_vectors_flat = self.nb_edge_vectors.reshape(-1, 2)  # (N*K, 2)

            # === GHOST POINTS DATA ===
            self.ghost_sites = mesh.lsfd_neighbors.ghost_coords
            self.n_ghosts = len(self.ghost_sites)
            self.ghost_dist = mesh.lsfd_neighbors.ghost_dist
            self.own_ghost_col_idx = mesh.lsfd_neighbors.indices_of_own_ghost_point_in_sites_with_ghosts
            self.ghost_coords = mesh.lsfd_neighbors.ghost_coords - self.sites[self.boundary_indices] # (B,2)
        else:
            self.nb_indices = mesh.lsfd_neighbors.indices
            self.nb_coords = mesh.lsfd_neighbors.coords
            self.nb_dist = mesh.lsfd_neighbors.distances
            # ← Предвычисленные векторы из mesh
            self.nb_edge_vectors = mesh.lsfd_neighbors.lsfd_edge_vectors  # (N, K, 2)
            self.nb_edge_vectors_flat = self.nb_edge_vectors.reshape(-1, 2)  # (N*K, 2)


        self.dtype = self.nb_dist.dtype
        self.weight_func_chose = weight_function

        # === 1. PSI (gamma=0) — только условие Неймана на -1 ===
        self.nb_indices_psi, self.nb_coords_psi, self.nb_dist_psi = self.setup_boundary_conditions(
            matrix_type='psi')

        # === 2. PSI_GAMMA (gamma≠0) — условия на -1 и -2 ===
        self.nb_indices_psi_gamma, self.nb_coords_psi_gamma, self.nb_dist_psi_gamma = self.setup_boundary_conditions(
            matrix_type='psi_gamma')

        # === 3. MU (Poisson) — дивергенция на -1 для всех, Нейман на -2 для границ ===
        self.nb_indices_mu, self.nb_coords_mu, self.nb_dist_mu = self.setup_boundary_conditions(
            matrix_type='mu')

        # Scalling matrix
        self.domain_radii =  np.max(self.nb_dist, axis=1)
        self.boundary_domain_inv = 1.0 / self.domain_radii[self.boundary_indices]
        self.H_matrix_psi, self.H_matrix_mu = self.compute_H_matrix()

        # Weight matrix

        self.W_matrix_psi = self.compute_W_matrix(matrix_type='psi')
        self.W_matrix_psi_gamma = self.compute_W_matrix(matrix_type='psi_gamma')
        self.W_matrix_mu = self.compute_W_matrix(matrix_type='mu')

        # S matrix

        self.S_matrix_psi = self.compute_S_matrix(s_direction = s_direction, matrix_type='psi')
        self.S_matrix_psi_gamma = self.compute_S_matrix(s_direction = s_direction, matrix_type='psi_gamma')
        self.S_matrix_mu = self.compute_S_matrix(s_direction = s_direction, matrix_type='mu')

        # Final G matrix

        self.cond_stats = {}
        self.check_condition_number = check_condition_number

        self.G_matrix_psi = self.compute_G_matrix(matrix_type='psi')
        self.G_matrix_psi_gamma = self.compute_G_matrix(matrix_type='psi_gamma')
        self.G_matrix_mu = self.compute_G_matrix(matrix_type='mu')

        # Печать статистики в конце инициализации
        if self.check_condition_number:
            self._print_condition_stats()

        # Psi_rhs

        self.precompute_psi_indices() # psi indices for rhs
        self.weights, self.rows, self.cols = self.precompute_delta_matrix()

        # ← Кэш для U_link
        self._use_sparse_delta = False
        self._U_link_cache = None
        self._A_cache = None
        self.Delta_matrix = None

        # mu rhs

        self.mu_rhs_indices = self.nb_indices_mu  # (N, K)
        self._idx_last = self.lsfd_neighbors_amount - 1  # Уравнение Пуассона (div J)
        self._idx_second_last = self.lsfd_neighbors_amount - 2  # Условие Неймана (I_boundary)

        # Chose einsum form: parallel or not

        self._batched_dot, self._method_name = get_batched_dot(
            use_numba=use_numba,
            num_threads=num_threads
        )

        print(self._method_name, 'Number of threads: ', num_threads)

    def compute_H_matrix(self): # Compute scalling matrix for GL and Poisson equations

        n_sites = len(self.domain_radii)
        h = self.domain_radii  # [N_sites]

        h_inv = 1.0 / h  # [N_sites]
        h_inv2 = h_inv ** 2  # [N_sites]
        h_inv3 = h_inv ** 3  # [N_sites]
        h_inv4 = h_inv ** 4  # [N_sites]

        # Порядок: [Dx, Dy, Dxx, Dyy, Dxy, Dxxx, Dyyy, Dxxy, Dyyx, Dxxxx, Dyyyy, Dxxxy, Dyyyx, Dxxyy]
        H_matrix_psi = np.stack([
            h_inv, h_inv,  # 1-й порядок (2)
            h_inv2, h_inv2, h_inv2,  # 2-й порядок (3)
            h_inv3, h_inv3, h_inv3, h_inv3,  # 3-й порядок (4)
            h_inv4, h_inv4, h_inv4, h_inv4, h_inv4,  # 4-й порядок (5)
        ], axis=1)  # [N_sites, 14]

        # ← Mu: константа + производные (15 элементов)
        H_matrix_mu = np.hstack([
            np.ones((n_sites, 1), dtype=self.dtype),  # Константа (μ)
            H_matrix_psi  # Производные
        ])  # [N_sites, 15]

        return H_matrix_psi, H_matrix_mu

    def setup_boundary_conditions(self, matrix_type: str = 'psi',  # 'psi', 'psi_gamma', или 'mu'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Векторизованная установка граничных условий для LSFD матриц.

        Args:
            matrix_type: Тип матрицы ('psi', 'psi_gamma', 'mu')
        Returns:
            Обновлённые массивы Neighb_sites, Neighb_sites_coords, Neighb_sites_dist
        """

        n_sites = len(self.nb_indices)
        n_neighbors = self.nb_indices.shape[1]

        # Создаём копии чтобы не модифицировать оригинал
        nb_indices = self.nb_indices.copy()
        nb_coords = self.nb_coords.copy()
        nb_dist = self.nb_dist.copy()

        # Индексы для последней и предпоследней колонки
        idx_last = n_neighbors - 1
        idx_second_last = n_neighbors - 2
        boundary_indices = self.boundary_indices

        if matrix_type == 'mu':
            # Для Mu: ВСЕ точки получают уравнение Пуассона
            nb_indices[:, idx_last] = np.arange(n_sites)
            nb_coords[:, idx_last] = np.array([1.0, 1.0])
            nb_dist[:, idx_last] = self.normal_vecs_length

            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                nb_indices[boundary_indices, col_idx] = boundary_indices
                nb_coords[boundary_indices, col_idx] = self.normal_vecs
                nb_dist[boundary_indices, col_idx] = self.normal_vecs_length
            else:
                nb_indices[boundary_indices, idx_second_last] = boundary_indices
                nb_coords[boundary_indices, idx_second_last] = self.normal_vecs
                nb_dist[boundary_indices, idx_second_last] = self.normal_vecs_length

        if matrix_type == 'psi':
            # Psi
            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                nb_indices[boundary_indices, col_idx] = boundary_indices
                nb_coords[boundary_indices, col_idx] = self.normal_vecs
                nb_dist[boundary_indices, col_idx] = self.normal_vecs_length
            else:
                nb_indices[boundary_indices, idx_last] = boundary_indices
                nb_coords[boundary_indices, idx_last] = self.normal_vecs
                nb_dist[boundary_indices, idx_last] = self.normal_vecs_length


        if matrix_type == 'psi_gamma':
            # Psi с gamma: граничные точки получают кубическое условие

            nb_indices[boundary_indices, idx_last] = boundary_indices
            nb_coords[boundary_indices, idx_last] = self.normal_vecs
            nb_dist[boundary_indices, idx_last] = self.normal_vecs_length

            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                nb_indices[boundary_indices, col_idx] = boundary_indices
                nb_coords[boundary_indices, col_idx] = self.normal_vecs
                nb_dist[boundary_indices, col_idx] = self.normal_vecs_length
            else:
                nb_indices[boundary_indices, idx_second_last] = boundary_indices
                nb_coords[boundary_indices, idx_second_last] = self.normal_vecs
                nb_dist[boundary_indices, idx_second_last] = self.normal_vecs_length

        return nb_indices, nb_coords, nb_dist

    def compute_W_matrix(self, matrix_type: str = 'psi',  # 'psi', 'psi_gamma', или 'mu'
    ) -> np.ndarray:

        """Compute weight matrix for GL and Poisson equations.
        Returns W as vector (N, n_neighbors) for memory efficiency.
        Will be used with broadcasting in G_matrix computation.
        """

        if matrix_type == 'psi':
            nb_dist = self.nb_dist_psi
        elif matrix_type == 'psi_gamma':
            nb_dist = self.nb_dist_psi_gamma
        elif matrix_type == 'mu':
            nb_dist = self.nb_dist_mu
        else:
            raise ValueError(f'Wrong matrix_type: {matrix_type}. '
                             f'Choose psi, psi_gamma, or mu')
        d = nb_dist / (self.domain_radii[:, np.newaxis])

        if self.weight_func_chose == 'exp':
            W_matrix = (np.exp(- d ** 2)).astype(self.dtype)
        elif self.weight_func_chose == '1r':
            W_matrix = (1.0 / d).astype(self.dtype)
        elif self.weight_func_chose == '1r2':
            W_matrix = (1.0 / d**2).astype(self.dtype)
        elif self.weight_func_chose == '1':
            W_matrix = (np.ones_like(d)).astype(self.dtype)
        elif self.weight_func_chose == 'poly4':
            W_matrix =  ( np.sqrt(4 / np.pi) * (1 - d**2)**4 ).astype(self.dtype)
        else:
            W_matrix = (np.exp(- d ** 2)).astype(self.dtype)

        #W_matrix = self._apply_boundary_boost(W_matrix, nb_dist, matrix_type)
        return W_matrix


    def compute_S_matrix(self, s_direction: np.ndarray, matrix_type: str = 'psi',  # 'psi', 'psi_gamma', или 'mu'
    ) -> np.ndarray:

        boundary_indices = self.boundary_indices
        s_x, s_y = s_direction[0] * self.boundary_domain_inv, s_direction[1] * self.boundary_domain_inv

        if matrix_type == 'psi':
            nb_coords = self.nb_coords_psi / self.domain_radii[:, np.newaxis, np.newaxis]
        elif matrix_type == 'psi_gamma':
            nb_coords = self.nb_coords_psi_gamma / self.domain_radii[:, np.newaxis, np.newaxis]
        elif matrix_type == 'mu':
            nb_coords = self.nb_coords_mu / self.domain_radii[:, np.newaxis, np.newaxis]
        else:
            raise ValueError(f'Wrong matrix_type: {matrix_type}. '
                             f'Choose psi, psi_gamma, or mu')

        x, y = nb_coords[:,  :, 0], nb_coords[:, :,  1]
        S_base = np.stack([x, y,
                           x**2 / 2, y**2 / 2, x * y,
                           x**3 / 6, y**3 / 6, (x**2) * y / 2, (y**2) * x / 2,
                           x**4 / 24, y**4 / 24, (x**3) * y / 6, (y**3) * x / 6, (x**2) * (y**2) / 4], axis = 2)

        if matrix_type == 'mu':

            n_sites, n_neighbors, n_unknown = np.shape(S_base)
            zero_deriv = np.ones_like(x, dtype=self.dtype)
            S_base = np.concatenate([zero_deriv[:, :, np.newaxis], S_base], axis=2)

            x_poiss = x[:, n_neighbors - 1]
            y_poiss = y[:, n_neighbors - 1]
            S_base[:, n_neighbors - 1] = 0.0
            S_base[:, n_neighbors - 1, 3] = x_poiss ** 2
            S_base[:, n_neighbors - 1, 4] = y_poiss ** 2

            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                S_base[boundary_indices, col_idx] = 0.0  # Обнуляем всё
                S_base[boundary_indices, col_idx, 1] = x[boundary_indices, col_idx]
                S_base[boundary_indices, col_idx, 2] = y[boundary_indices, col_idx]
            else:
                S_base[boundary_indices, n_neighbors - 2] = 0.0  # Обнуляем всё
                S_base[boundary_indices, n_neighbors - 2, 1] = x[boundary_indices, n_neighbors - 2]
                S_base[boundary_indices, n_neighbors - 2, 2] = y[boundary_indices, n_neighbors - 2]

        elif  matrix_type == 'psi':
            n_sites, n_neighbors, n_unknown = np.shape(S_base)

            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                S_base[boundary_indices, col_idx] = 0.0  # Обнуляем всё
                S_base[boundary_indices, col_idx, 0] = x[boundary_indices, col_idx]
                S_base[boundary_indices, col_idx, 1] = y[boundary_indices, col_idx]
            else:
                S_base[boundary_indices, n_neighbors - 1] = 0.0  # Обнуляем всё
                S_base[boundary_indices, n_neighbors - 1, 0] = x[boundary_indices, n_neighbors - 1]
                S_base[boundary_indices, n_neighbors - 1, 1] = y[boundary_indices, n_neighbors - 1]


        elif matrix_type == 'psi_gamma':
            n_sites, n_neighbors, n_unknown = np.shape(S_base)

            S_base[boundary_indices, n_neighbors - 1] = 0.0
            S_base[boundary_indices, n_neighbors - 1, 0] = x[boundary_indices, n_neighbors - 1]
            S_base[boundary_indices, n_neighbors - 1, 1] = y[boundary_indices, n_neighbors - 1]
            self.idx_second_last = n_neighbors - 2

            if self.use_ghost_points and self.ghost_version == 'asym':
                col_idx = self.own_ghost_col_idx
                self.x_bc2 = x[boundary_indices, col_idx].copy()
                self.y_bc2 = y[boundary_indices, col_idx].copy()

                S_base[boundary_indices, col_idx] = 0.0
                S_base[boundary_indices, col_idx, 2] = 2 * self.x_bc2 * s_x + self.y_bc2 * s_y
                S_base[boundary_indices, col_idx, 3] = 2 * self.y_bc2 * s_y + self.x_bc2 * s_x
                S_base[boundary_indices, col_idx, 4] = self.x_bc2 * s_y + self.y_bc2 * s_x

            else:
                self.x_bc2 = x[boundary_indices, n_neighbors - 2].copy()
                self.y_bc2 = y[boundary_indices, n_neighbors - 2].copy()

                S_base[boundary_indices, n_neighbors - 2] = 0.0
                S_base[boundary_indices, n_neighbors - 2, 2] = 2 * self.x_bc2 * s_x + self.y_bc2 * s_y
                S_base[boundary_indices, n_neighbors - 2, 3] = 2 * self.y_bc2 * s_y + self.x_bc2 * s_x
                S_base[boundary_indices, n_neighbors - 2, 4] = self.x_bc2 * s_y + self.y_bc2 * s_x

        return S_base

    def compute_G_matrix(self, matrix_type: str = 'psi')-> np.ndarray:

        if matrix_type == 'psi':
            S = self.S_matrix_psi
            H = self.H_matrix_psi
            W = self.W_matrix_psi

        elif matrix_type == 'psi_gamma':
            S = self.S_matrix_psi_gamma
            H = self.H_matrix_psi
            W = self.W_matrix_psi_gamma

        elif matrix_type == 'mu':
            S = self.S_matrix_mu
            H = self.H_matrix_mu
            W = self.W_matrix_mu

        else:
            raise ValueError(f'Wrong matrix_type: {matrix_type}. '
                             f'Choose psi, psi_gamma, or mu')

        STW = np.einsum('nji,ni->nji', S.transpose(0, 2, 1), W)
        STWS = np.einsum('nji,nik->njk', STW, S)

        if self.check_condition_number:
            # Вычисляем cond для каждой вершины (STWS shape: N x 14x14 или N x 15x15)
            # Используем 2-норму (по умолчанию). Для больших сеток ~2-5 сек.
            conds = np.array([np.linalg.cond(STWS[i]) for i in range(self.n_sites)])
            self.cond_stats[matrix_type] = conds

        STWS_inv = np.linalg.inv(STWS) #np.linalg.inv(STWS)
        HSTWS_inv = np.einsum('nj,nji->nji', H, STWS_inv)
        G = np.einsum('njk,nki->nji', HSTWS_inv, STW)

        return G

    def update_G_matrix_psi_gamma(self, s_direction: np.ndarray)-> np.ndarray:

        # NOT USED WITH GHOST POINTS!!!!!
        # Update S matrix boundary values due to new s_direction
        s_x, s_y = s_direction[0] * self.boundary_domain_inv, s_direction[1] * self.boundary_domain_inv
        coeff_dxx = 2 * self.x_bc2 * s_x + self.y_bc2 * s_y
        coeff_dyy = 2 * self.y_bc2 * s_y + self.x_bc2 * s_x
        coeff_dxy = self.x_bc2 * s_y + self.y_bc2 * s_x

        self.S_matrix_psi_gamma[self.boundary_indices, self.idx_second_last, 2] = coeff_dxx
        self.S_matrix_psi_gamma[self.boundary_indices, self.idx_second_last, 3] = coeff_dyy
        self.S_matrix_psi_gamma[self.boundary_indices, self.idx_second_last, 4] = coeff_dxy

        # Update G matrix boundary values

        S_bc = self.S_matrix_psi_gamma[self.boundary_indices]  # (N_bc, K, 14)
        H_bc = self.H_matrix_psi[self.boundary_indices]  # (N_bc, 14)
        W_bc = self.W_matrix_psi_gamma[self.boundary_indices]  # (N_bc, K)

        STW_bc = np.einsum('nji,ni->nji', S_bc.transpose(0, 2, 1), W_bc)
        STWS_bc = np.einsum('nji,nik->njk', STW_bc, S_bc)
        STWS_inv_bc = np.linalg.inv(STWS_bc)  # (N_bc, 14, 14)
        HSTWS_inv_bc = np.einsum('nj,nji->nji', H_bc, STWS_inv_bc)  # (N_bc, 14, 14)
        G_bc = np.einsum('njk,nki->nji', HSTWS_inv_bc, STW_bc)  # (N_bc, 14, K)

        self.G_matrix_psi_gamma[self.boundary_indices] = G_bc

        return self.G_matrix_psi_gamma


    # Функции для вычисления RHS для psi derivetives

    def precompute_psi_indices(self):
        """
        Предвычисляет индексы i и j для всех рёбер.
        """
        N_sites = len(self.sites)
        K = self.lsfd_neighbors_amount

        # Индексы центральных точек (повторяются K раз)
        i_idx = np.repeat(np.arange(N_sites), K)  # (N*K,)

        # mask = (self.nb_indices_psi == self.nb_indices)
        #
        # print(mask)

        # Индексы соседних точек
        j_idx = self.nb_indices_psi.flatten()  # (N*K,)

        self.psi_i_idx = i_idx  # (N*K,)
        self.psi_j_idx = j_idx  # (N*K,)

    def set_link_variables(self, A_applied: np.ndarray = None):
        """
        Вычисляет U_{ij} = exp(-i * A_i · e_ij).
        Кэширует результат и возвращает из кэша если A не изменился.

        Работает для всех режимов:
        - A = None (B=0)
        - A = constant
        - A = time-dependent
        - A = ramp (меняется → constant)
        """
        # Проверяем изменился ли A
        if self._A_cache is not None:
            # Быстрая проверка: если массивы идентичны → возвращаем кэш
            if np.array_equal(A_applied, self._A_cache):
                return self._U_link_cache

        # Вычисляем новый U_link

        A_i = A_applied[self.psi_i_idx]   # (N*K, 2)
        A_dot_e = np.einsum('ij,ij->i', A_i, self.nb_edge_vectors_flat)  # (N*K,)
        U_link = np.cos(A_dot_e) - 1j * np.sin(A_dot_e) #np.exp(-1j * A_dot_e)  # (N*K,)
        #U_link = U_link.reshape(self.n_sites, self.lsfd_neighbors_amount)  # (N, K)

        #← Обновляем кэш
        self._U_link_cache = U_link
        self._A_cache = A_applied.copy()
        return U_link

    def compute_ghost_values(self, psi: np.ndarray, A_applied: np.ndarray, s_applied: np.ndarray,
                          eta: float, psi_derivatives: np.ndarray = None):

        #print('Ghost_dist', self.ghost_dist)

        n_vec = self.normal_vecs
        nx,ny = n_vec[:, 0], n_vec[:, 1]
        s_x, s_y = s_applied[0], s_applied[1]
        n_dot_s = nx * s_x + ny * s_y
        psi_b = psi[self.boundary_indices]

        if psi_derivatives is None:
            D_xx = np.zeros_like(psi_b)
            D_yy = np.zeros_like(psi_b)
            D_xy = np.zeros_like(psi_b)
        else:
            D_xx = psi_derivatives[:, 2][self.boundary_indices]
            D_yy = psi_derivatives[:, 3][self.boundary_indices]
            D_xy = psi_derivatives[:, 4][self.boundary_indices]

        A_i = A_applied[self.boundary_indices]  # (B, 2)
        A_dot_e = np.einsum('ij,ij->i', A_i, self.ghost_coords)  # (N*K,)
        U_link_ghost = np.cos(A_dot_e) + 1j * np.sin(A_dot_e)  # np.exp(1j * A_dot_e)  # (N*K,)
        psi_ghost = U_link_ghost * (
                    psi_b - 1j * eta * n_dot_s * psi_b * self.ghost_dist
                    #+ 0.5 * D_xx * (nx * self.ghost_dist)**2 + 0.5 * D_yy * (ny * self.ghost_dist)**2
                    #+ D_xy * (nx * self.ghost_dist) * (ny * self.ghost_dist)
        )

        return psi_ghost

    def compute_delta_psi(self, psi: np.ndarray, A_applied: np.ndarray, s_applied: np.ndarray,
                          eta: float, gamma: float, Bz: float, psi_derivatives: np.ndarray = None):
        """
        Вычисляет разности psi с калибровочными фазами.

        delta_psi_{ij} = U_{ij} * psi_j - psi_i

        Args:
            psi: (N,) — волновая функция
            U_link: (N, K) — калибровочные множители (если None, то U=1)

        Returns:
            delta_psi: (N, K) — разности для каждой точки и её соседей
        """
        # Извлекаем psi[i] и psi[j] через предвычисленные индексы

        n_vec = self.normal_vecs
        s_x, s_y = s_applied[0], s_applied[1]
        n_dot_s = n_vec[:, 0] * s_x + n_vec[:, 1] * s_y

        if self.use_ghost_points:
            psi_ghost = self.compute_ghost_values(psi,A_applied, s_applied, eta, psi_derivatives)
            psi_ext = np.concatenate([psi_ghost, psi])

            if self._use_sparse_delta == False:
                psi_i = psi[self.psi_i_idx]  # (N*K,)
                psi_j = psi_ext[self.psi_j_idx] # (N*K,)
                U_link = self.set_link_variables(A_applied)  # U_{ij} * psi_j - psi_i
                delta_psi = U_link * psi_j - psi_i  # (N*K,)
            else:
                Delta_matrix = self.build_delta_matrix(A_applied=A_applied)
                delta_psi = (Delta_matrix @ psi_ext)

        else:

            if self._use_sparse_delta == False:
                psi_i = psi[self.psi_i_idx]  # (N*K,)
                psi_j = psi[self.psi_j_idx] # (N*K,)
                U_link = self.set_link_variables(A_applied)  # U_{ij} * psi_j - psi_i
                delta_psi = U_link * psi_j - psi_i  # (N*K,)
            else:
                Delta_matrix = self.build_delta_matrix(A_applied=A_applied)
                delta_psi = (Delta_matrix @ psi)

         # ← ГРАНИЧНЫЕ УСЛОВИЯ: модифицируем последние 1-2 колонки
        K = self.lsfd_neighbors_amount
        delta_psi = delta_psi.reshape(self.n_sites, K)  # (N, K)
        idx_last = K - 1  # Всегда: уравнение для границы
        idx_second_last = K - 2  # Только если gamma ≠ 0

        if gamma == 0:

            if self.use_ghost_points and self.ghost_version == 'asym':
                delta_psi[self.boundary_indices, self.own_ghost_col_idx] = -1j * eta * n_dot_s * psi[self.boundary_indices]
            else:
                delta_psi[self.boundary_indices, idx_last] = -1j * eta * n_dot_s * psi[self.boundary_indices]

        else:
            n_rot_K = n_vec[:, 0] * s_y - n_vec[:, 1] * s_x
            rhs1 = (eta / gamma) * n_dot_s * psi[self.boundary_indices]
            rhs2 = 1j * (Bz / 2) * n_rot_K * psi[self.boundary_indices]
            delta_psi[self.boundary_indices, idx_last] = np.zeros(len(self.boundary_indices), dtype = self.dtype)
            if self.use_ghost_points and self.ghost_version == 'asym':
                delta_psi[self.boundary_indices, self.own_ghost_col_idx] = rhs1 + rhs2  # Или другое значение
            else:
                delta_psi[self.boundary_indices, idx_second_last] = rhs1 + rhs2  # Или другое значение


        return delta_psi.reshape(self.n_sites, self.lsfd_neighbors_amount)

    def precompute_delta_matrix(self):

        self.edge_indices = np.arange(len(self.psi_j_idx))
        weights = np.ones(len(self.edge_indices))
        rows = np.concatenate([self.edge_indices, self.edge_indices])
        cols = np.concatenate([self.psi_j_idx, self.psi_i_idx])
        return weights, rows, cols

    def build_delta_matrix(self, A_applied) -> sp.csr_array:

        if self.Delta_matrix is not None:
            # Быстрая проверка: если массивы идентичны → возвращаем кэш
            if np.array_equal(A_applied, self._A_cache):
                return self.Delta_matrix

        A_i = A_applied[self.psi_i_idx]  # (N*K, 2)
        A_dot_e = np.einsum('ij,ij->i', A_i, self.nb_edge_vectors_flat)  # (N*K,)
        U_link = np.cos(A_dot_e) - 1j * np.sin(A_dot_e)
        values = np.concatenate([U_link * self.weights, -self.weights])

        if self.use_ghost_points:
            Delta_matrix = sp.csr_array(
                (values, (self.rows, self.cols)), shape=(len(self.edge_indices), len(self.sites) + self.n_ghosts)
            )
        else:
            Delta_matrix = sp.csr_array(
                (values, (self.rows, self.cols)), shape=(len(self.edge_indices), len(self.sites))
            )

        self._A_cache = A_applied
        self.Delta_matrix = Delta_matrix

        return Delta_matrix

    # Функции для вычисления RHS для mu derivetives

    def compute_mu_rhs(self, mu_guess: np.ndarray, div_J: np.ndarray,
                       I_boundary: np.ndarray, derivatives: np.ndarray):
        """
        Вычисляет правую часть для уравнения Пуассона.

        Args:
            mu_guess: (N,) — mu с прошлой итерации
            div_J: (N,) — дивергенция тока (источник)
            I_boundary: (N_boundary,) — ток на границе (условие Неймана)

        Returns:
            rhs: (N, K) — правая часть для каждой точки и соседа
        """
        # Основное заполнение: mu[neighbor_j]

        if self.use_ghost_points:

            mu_xx = derivatives[:, 3][self.boundary_indices]
            mu_yy = derivatives[:, 4][self.boundary_indices]
            mu_xy = derivatives[:, 5][self.boundary_indices]
            x = self.ghost_coords[:, 0]
            y = self.ghost_coords[:, 1]

            mu_ghost = (mu_guess[self.boundary_indices] + I_boundary * self.ghost_dist ) #+
                       # 0.5 * mu_xx * x**2 + 0.5 * mu_yy * y**2 + mu_xy * x*y
            mu_guess = np.concatenate([ mu_ghost, mu_guess])

        rhs = mu_guess[self.mu_rhs_indices]  # (N, K)
        # Последняя колонка: уравнение Пуассона ∇²μ = div J
        rhs[:, self._idx_last] = div_J  # (N,)
        # Предпоследняя колонка: условие Неймана для границ
        if self.use_ghost_points and self.ghost_version == 'asym':
            rhs[self.boundary_indices, self.own_ghost_col_idx] = I_boundary
        else:
            rhs[self.boundary_indices, self._idx_second_last] = I_boundary

        return rhs

    def solve_poisson(self, div_J: np.ndarray, I_boundary: np.ndarray,
                      mu_guess: np.ndarray = None,
                      tolerance: float = 1e-5,
                      max_iterations: int = 5000):

        N_sites = self.n_sites
        areas = self.voronoi_areas

        if mu_guess is None:
            mu_guess = np.zeros(N_sites)

        mu = mu_guess.copy()
        derivatives = np.zeros((N_sites, 15), dtype = self.dtype)

        final_error = 1.0  # Значение по умолчанию, если цикл не выполнится
        actual_iterations = max_iterations
        gradients = None
        laplacian = None

        batched_dot = self._batched_dot
        G_matrix_mu = self.G_matrix_mu

        for iteration in range(max_iterations):
            rhs = self.compute_mu_rhs(mu, div_J, I_boundary, derivatives)
            #derivatives = np.einsum('nji,ni->nj', self.G_matrix_mu, rhs, optimize='optimal')
            derivatives = batched_dot(G_matrix_mu, rhs)

            mu_new = derivatives[:, 0]

            err = np.abs(mu_new - mu)
            norm_err = float(np.sqrt(np.sum(err ** 2 * areas) / np.sum(areas)))
            norm_mu_new = float(np.sqrt(np.sum(mu_new ** 2 * areas) / np.sum(areas)))
            error = norm_err / (norm_mu_new + 1e-15)

            if error < tolerance:
                mu = mu_new
                final_error = error
                actual_iterations = iteration + 1
                gradients = derivatives[:, 1:3]
                laplacian = derivatives[:, 3] + derivatives[:, 4]
                break

            if error > 10:
                mu = mu_new
                final_error = error
                actual_iterations = iteration + 1
                gradients = derivatives[:, 1:3]
                laplacian = derivatives[:, 3] + derivatives[:, 4]
                break

            mu = mu_new
            final_error = error  # Сохраняем ошибку последней итерации, если вышли по лимиту
            actual_iterations = iteration + 1
            gradients = derivatives[:, 1:3]
            laplacian = derivatives[:, 3] + derivatives[:, 4]

        return mu, gradients, laplacian, final_error, actual_iterations

    def _print_condition_stats(self):
        """Выводит Min/Mean/Max числа обусловленности для всех типов матриц."""
        print("\n📊 Condition Number Statistics (STWS matrices):")
        print("="*65)
        print(f"{'Type':<12s} | {'Min':<10s} {'Mean':<10s} {'Max':<10s} {'Status'}")
        print("-"*65)
        for mtype, conds in self.cond_stats.items():
            c_min, c_mean, c_max = np.min(conds), np.mean(conds), np.max(conds)
            # Оценка "здоровья" матрицы
            if c_max < 1e6:
                status = "✅ Excellent"
            elif c_max < 1e10:
                status = "⚠️  Acceptable"
            else:
                status = "❌ Risky (check mesh/neighbors)"
            print(f"{mtype:<12s} | {c_min:<10.2e} {c_mean:<10.2e} {c_max:<10.2e} {status}")
        print("="*65 + "\n")

    def plot_condition_numbers(
            self,
            matrix_type: str = 'psi',
            subtract_min: bool = True,
            log_scale: bool = True,
            highlight_boundary: bool = True,
            ax=None,
            figsize=(10, 8),
            cmap='viridis',
            marker_size: float = 2.0,
            show_colorbar: bool = True,
            show_stats: bool = True,
    ):
        """
        Визуализирует числа обусловленности LSFD-матриц для каждой вершины.

        Args:
            matrix_type: Тип матрицы ('psi', 'psi_gamma', 'mu').
                        Должен быть предварительно вычислен с check_condition_number=True.
            subtract_min: Если True, строит (cond - min(cond)) для лучшего контраста.
            log_scale: Если True (и subtract_min=False), использует лог-шкалу для цветов.
            highlight_boundary: Если True, выделяет граничные точки красной обводкой.
            ax: Matplotlib axes (если None, создаётся новый).
            figsize: Размер фигуры.
            cmap: Цветовая карта.
            marker_size: Размер маркеров для внутренних точек.
            show_colorbar: Показывать ли цветовую шкалу.
            show_stats: Показывать ли текстовую статистику в углу.

        Returns:
            fig, ax
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        if matrix_type not in self.cond_stats:
            raise ValueError(
                f"Condition numbers for '{matrix_type}' not computed.\n"
                f"Available: {list(self.cond_stats.keys())}.\n"
                f"Создайте LSFD_operators с check_condition_number=True."
            )

        conds = self.cond_stats[matrix_type].copy()

        if subtract_min:
            conds_plot = conds - conds.min()
            label = f"cond(SᵀWS) − {conds.min():.2e}"
            use_log = False  # после вычитания min лог-шкала не нужна
        else:
            conds_plot = conds
            label = f"cond(SᵀWS)"
            use_log = log_scale

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Маска границы
        is_boundary = np.zeros(self.n_sites, dtype=bool)
        is_boundary[self.boundary_indices] = True

        # Цветовая норма
        if use_log:
            norm = LogNorm(vmin=conds_plot.min() + 1e-15, vmax=conds_plot.max())
        else:
            norm = Normalize(vmin=conds_plot.min(), vmax=conds_plot.max())

        # Отрисовка
        if highlight_boundary:
            # Сначала внутренние (мелкие маркеры)
            sc_int = ax.scatter(
                self.sites[~is_boundary, 0],
                self.sites[~is_boundary, 1],
                c=conds_plot[~is_boundary],
                s=marker_size,
                cmap=cmap,
                norm=norm,
                alpha=0.8,
            )
            # Потом граничные (крупнее, с красной обводкой)
            sc_bnd = ax.scatter(
                self.sites[is_boundary, 0],
                self.sites[is_boundary, 1],
                c=conds_plot[is_boundary],
                s=marker_size * 4,
                cmap=cmap,
                norm=norm,
                edgecolors='red',
                linewidths=0.6,
                alpha=1.0,
            )
            sc_for_cbar = sc_int
        else:
            sc_for_cbar = ax.scatter(
                self.sites[:, 0],
                self.sites[:, 1],
                c=conds_plot,
                s=marker_size,
                cmap=cmap,
                norm=norm,
            )

        if show_colorbar:
            cbar = plt.colorbar(sc_for_cbar, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=11)

        ax.set_aspect('equal')
        ax.set_xlabel('x [ξ]')
        ax.set_ylabel('y [ξ]')
        ax.set_title(f'Condition numbers: {matrix_type}', fontsize=13, fontweight='bold')

        if show_stats:
            # Статистика отдельно для границы и внутренности
            bnd_mean = conds[is_boundary].mean() if is_boundary.any() else 0
            int_mean = conds[~is_boundary].mean() if (~is_boundary).any() else 0
            bnd_max = conds[is_boundary].max() if is_boundary.any() else 0

            # Топ-5 худших точек
            worst_idx = np.argsort(conds)[-5:][::-1]
            worst_text = "\n".join([
                f"  #{idx}: cond={conds[idx]:.2e} at ({self.sites[idx, 0]:.2f}, {self.sites[idx, 1]:.2f})"
                + (" [BND]" if is_boundary[idx] else "")
                for idx in worst_idx
            ])

            stats_text = (
                f"Min: {conds.min():.2e}   Max: {conds.max():.2e}\n"
                f"Mean (int): {int_mean:.2e}   Mean (bnd): {bnd_mean:.2e}\n"
                f"Max (bnd): {bnd_max:.2e}\n"
                f"── Top-5 worst ──\n{worst_text}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=8, family='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            )

        return fig, ax

    def plot_condition_comparison(self, matrix_types=None, figsize=(18, 6)):
        """
        Сравнивает condition numbers для нескольких типов матриц side-by-side.

        Args:
            matrix_types: Список типов ('psi', 'psi_gamma', 'mu').
                         Если None, использует все доступные.
            figsize: Размер фигуры.
        """
        import matplotlib.pyplot as plt

        if matrix_types is None:
            matrix_types = list(self.cond_stats.keys())

        available = [m for m in matrix_types if m in self.cond_stats]
        if not available:
            raise ValueError(
                f"No condition numbers computed for {matrix_types}.\n"
                f"Available: {list(self.cond_stats.keys())}."
            )

        fig, axes = plt.subplots(1, len(available), figsize=figsize, constrained_layout=True)
        if len(available) == 1:
            axes = [axes]

        for ax, mtype in zip(axes, available):
            self.plot_condition_numbers(matrix_type=mtype, ax=ax, show_stats=True)

        return fig, axes

    def plot_condition_vs_distance(
            self,
            matrix_type: str = 'psi',
            figsize=(12, 5),
    ):
        """
        Строит зависимость condition number от расстояния до границы.
        Показывает, портится ли обусловленность у границы.

        Args:
            matrix_type: Тип матрицы.
            figsize: Размер фигуры.

        Returns:
            fig, axes (2 subplot'а: scatter + binned mean)
        """
        import matplotlib.pyplot as plt

        if matrix_type not in self.cond_stats:
            raise ValueError(f"Condition numbers for '{matrix_type}' not computed.")

        conds = self.cond_stats[matrix_type]

        # Расстояние до ближайшей граничной точки
        bnd_tree = KDTree(self.sites[self.boundary_indices])
        dist_to_bnd, _ = bnd_tree.query(self.sites)

        is_boundary = np.zeros(self.n_sites, dtype=bool)
        is_boundary[self.boundary_indices] = True

        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # === Левый subplot: scatter plot ===
        ax1 = axes[0]
        sc = ax1.scatter(
            dist_to_bnd, conds,
            c=np.where(is_boundary, 'red', 'steelblue'),
            s=3, alpha=0.6,
        )
        ax1.set_yscale('log')
        ax1.set_xlabel('Distance to boundary [ξ]')
        ax1.set_ylabel(f'cond(SᵀWS) for {matrix_type}')
        ax1.set_title('Condition number vs distance to boundary')
        ax1.grid(True, alpha=0.3)

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Boundary'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=6, label='Interior'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # === Правый subplot: binned mean ===
        ax2 = axes[1]
        n_bins = 30
        bins = np.linspace(0, dist_to_bnd.max(), n_bins + 1)
        bin_idx = np.digitize(dist_to_bnd, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_means = np.zeros(n_bins)
        bin_stds = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                bin_means[b] = conds[mask].mean()
                bin_stds[b] = conds[mask].std()
                bin_counts[b] = mask.sum()

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        valid = bin_counts > 0

        ax2.semilogy(bin_centers[valid], bin_means[valid], 'o-', color='steelblue',
                     label='Mean cond')
        ax2.fill_between(
            bin_centers[valid],
            bin_means[valid] - bin_stds[valid],
            bin_means[valid] + bin_stds[valid],
            alpha=0.2, color='steelblue', label='±1 std'
        )
        ax2.set_xlabel('Distance to boundary [ξ]')
        ax2.set_ylabel(f'Mean cond(SᵀWS) for {matrix_type}')
        ax2.set_title('Binned condition number vs distance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        return fig, axes

    def save_top_bad_condition_numbers(
            self,
            filepath: str,
            matrix_type: str = 'psi',
            top_n: int = 10,
    ) -> None:
        """
        Сохраняет топ-N точек с худшими числами обусловленности в текстовый файл.

        Файл содержит:
        - Общую статистику (min/mean/max отдельно для границы и внутренности)
        - Топ-N худших точек в целом
        - Топ-N худших точек ТОЛЬКО на границе
        - Топ-N худших точек ТОЛЬКО внутри области

        Args:
            filepath: Путь к выходному .txt файлу.
            matrix_type: Тип матрицы ('psi', 'psi_gamma', 'mu').
            top_n: Количество худших точек для вывода (по умолчанию 10).
        """
        from pathlib import Path

        if matrix_type not in self.cond_stats:
            raise ValueError(
                f"Condition numbers for '{matrix_type}' not computed.\n"
                f"Available: {list(self.cond_stats.keys())}.\n"
                f"Создайте LSFD_operators с check_condition_number=True."
            )

        conds = self.cond_stats[matrix_type]

        # Определяем границу и внутренность
        is_boundary = np.zeros(self.n_sites, dtype=bool)
        is_boundary[self.boundary_indices] = True

        # Расстояние до границы для каждой точки
        bnd_tree = KDTree(self.sites[self.boundary_indices])
        dist_to_bnd, _ = bnd_tree.query(self.sites)

        # Тип точки
        point_type = np.where(is_boundary, "BND", "INT")

        # === Сборка общей таблицы данных ===
        # Структура: (index, x, y, cond, dist_to_bnd, type)
        data = []
        for i in range(self.n_sites):
            data.append({
                'idx': i,
                'x': self.sites[i, 0],
                'y': self.sites[i, 1],
                'cond': conds[i],
                'dist': dist_to_bnd[i],
                'type': point_type[i],
            })

        # Сортировки
        sorted_all = sorted(data, key=lambda d: d['cond'], reverse=True)[:top_n]
        sorted_bnd = sorted([d for d in data if d['type'] == "BND"],
                            key=lambda d: d['cond'], reverse=True)[:top_n]
        sorted_int = sorted([d for d in data if d['type'] == "INT"],
                            key=lambda d: d['cond'], reverse=True)[:top_n]

        # === Форматирование и запись в файл ===
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Статистика
        cond_bnd = conds[is_boundary]
        cond_int = conds[~is_boundary]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write(f" CONDITION NUMBER ANALYSIS: matrix_type = '{matrix_type}'\n")
            f.write(
                f" Total sites: {self.n_sites}  |  Boundary: {is_boundary.sum()}  |  Interior: {(~is_boundary).sum()}\n")
            f.write("=" * 90 + "\n\n")

            # Общая статистика
            f.write("GLOBAL STATISTICS:\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'Category':<15} {'Min':>12} {'Mean':>12} {'Median':>12} {'Max':>12} {'Count':>8}\n")
            f.write("-" * 90 + "\n")
            f.write(f"{'ALL':<15} {conds.min():>12.2e} {conds.mean():>12.2e} "
                    f"{np.median(conds):>12.2e} {conds.max():>12.2e} {len(conds):>8d}\n")
            if len(cond_bnd) > 0:
                f.write(f"{'BOUNDARY':<15} {cond_bnd.min():>12.2e} {cond_bnd.mean():>12.2e} "
                        f"{np.median(cond_bnd):>12.2e} {cond_bnd.max():>12.2e} {len(cond_bnd):>8d}\n")
            if len(cond_int) > 0:
                f.write(f"{'INTERIOR':<15} {cond_int.min():>12.2e} {cond_int.mean():>12.2e} "
                        f"{np.median(cond_int):>12.2e} {cond_int.max():>12.2e} {len(cond_int):>8d}\n")
            f.write("\n")
            f.write(f"Ratio Max(BND)/Max(INT): {cond_bnd.max() / (cond_int.max() + 1e-15):.2f}\n")
            f.write(f"Ratio Mean(BND)/Mean(INT): {cond_bnd.mean() / (cond_int.mean() + 1e-15):.2f}\n\n")

            # Вспомогательная функция для вывода таблицы
            def write_table(title, data_list):
                f.write(f"\n{title} (top {top_n}):\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Rank':<5} {'Idx':<7} {'Type':<5} "
                        f"{'X [ξ]':>10} {'Y [ξ]':>10} {'Dist→Bnd':>10} {'Cond':>14}\n")
                f.write("-" * 90 + "\n")
                for rank, d in enumerate(data_list, start=1):
                    f.write(f"{rank:<5} {d['idx']:<7} {d['type']:<5} "
                            f"{d['x']:>10.4f} {d['y']:>10.4f} {d['dist']:>10.4f} {d['cond']:>14.2e}\n")
                f.write("-" * 90 + "\n")

            write_table("TOP WORST CONDITION NUMBERS (ALL SITES)", sorted_all)
            write_table("TOP WORST CONDITION NUMBERS (BOUNDARY ONLY)", sorted_bnd)
            write_table("TOP WORST CONDITION NUMBERS (INTERIOR ONLY)", sorted_int)

            f.write("\n" + "=" * 90 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 90 + "\n")

        print(f"✅ Condition number report saved to: {filepath}")


    def analyze_boundary_stencil(
            self,
            K_list=(30, 50, 70, 100, 170),
            h_main: float = 0.5,
            figsize=(16, 10),
            save=None,
    ):
        """
        Анализ структуры стенсила на граничных точках для разных K.

        Для каждого K (не больше сохранённого n_lsfd_neighbors) вычисляет:
          - domain radius d0 = расстояние до K-го соседа (vs координата θ и vs K);
          - число соседей-граничных (n_bnd) и соседей-внутренних (n_int);
          - число "основных" точек (dist < h_main): всего / граничных / внутренних.

        Цель — понять, почему меньшее K улучшает границу: при большом K стенсил
        перенасыщается тангенциальными граничными точками, а нормальная
        (внутренняя) информация почти не прибавляется → растёт анизотропия фита.

        Returns:
            results: dict[K -> dict массивов по граничным точкам]
        """
        import matplotlib.pyplot as plt

        nb = self.mesh.lsfd_neighbors
        idx_full = nb.indices      # (N, Kmax), отсортированы по расстоянию
        dist_full = nb.distances   # (N, Kmax)
        Kmax = idx_full.shape[1]

        is_bnd = np.zeros(self.n_sites, dtype=bool)
        is_bnd[self.boundary_indices] = True
        b = self.boundary_indices

        # Сортировка вдоль дуги (для диска — по углу)
        theta = np.degrees(np.arctan2(self.sites[b, 1], self.sites[b, 0]))
        order = np.argsort(theta)
        theta_sorted = theta[order]
        b_sorted = b[order]

        results = {}
        summary_rows = []

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        for K in K_list:
            K = int(min(K, Kmax))
            if K != K and K > Kmax:
                print(f"Warning: K={K} > Kmax={Kmax}, clipped to {Kmax}")

            dK = dist_full[b_sorted, :K]     # (Nb, K)
            iK = idx_full[b_sorted, :K]      # (Nb, K)

            domain_radius = dK[:, -1]
            is_bnd_nb = is_bnd[iK]           # (Nb, K)
            n_bnd = is_bnd_nb.sum(axis=1)
            n_int = K - n_bnd

            main_mask = dK < h_main
            n_main = main_mask.sum(axis=1)
            n_main_bnd = (main_mask & is_bnd_nb).sum(axis=1)
            n_main_int = (main_mask & ~is_bnd_nb).sum(axis=1)

            results[K] = {
                'theta': theta_sorted,
                'domain_radius': domain_radius,
                'n_bnd': n_bnd,
                'n_int': n_int,
                'n_main': n_main,
                'n_main_bnd': n_main_bnd,
                'n_main_int': n_main_int,
            }

            # 1) domain radius vs θ
            axes[0, 0].plot(theta_sorted, domain_radius, '.-', ms=1.5, lw=0.6,
                            label=f'K={K}')
            # 2) доля граничных соседей среди K vs θ
            axes[0, 1].plot(theta_sorted, n_bnd / K, '.-', ms=1.5, lw=0.6,
                            label=f'K={K}')
            # 4) внутренние "основные" точки vs θ (нормальная информация)
            axes[1, 1].plot(theta_sorted, n_main_int, '.-', ms=1.5, lw=0.6,
                            label=f'K={K}')

            summary_rows.append((
                K,
                domain_radius.mean(), domain_radius.min(), domain_radius.max(),
                n_bnd.mean(), n_int.mean(),
                n_main.mean(), n_main_bnd.mean(), n_main_int.mean(),
            ))

        axes[0, 0].set_title('domain radius $d_0$ vs θ')
        axes[0, 0].set_ylabel('$d_0$ [ξ]')
        axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

        axes[0, 1].set_title('доля граничных (тангенциальных) соседей среди K')
        axes[0, 1].set_ylabel('$n_{bnd}/K$')
        axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

        # 3) средние счётчики "основных" точек vs K
        Ks = [r[0] for r in summary_rows]
        axes[1, 0].plot(Ks, [r[6] for r in summary_rows], 'o-', label='main: всего (dist<h)')
        axes[1, 0].plot(Ks, [r[7] for r in summary_rows], 's-', label='main: граничные')
        axes[1, 0].plot(Ks, [r[8] for r in summary_rows], '^-', label='main: внутренние')
        axes[1, 0].set_title('среднее число "основных" точек (dist < h)')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()

        axes[1, 1].set_title('внутренние "основные" точки vs θ (нормальная инф.)')
        axes[1, 1].set_xlabel('θ [deg]')
        axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()

        if save:
            fig.savefig(save, dpi=150, bbox_inches='tight')

        # Текстовая сводка
        print(f"\n📐 Boundary stencil analysis (h_main = {h_main:.2f}):")
        print("=" * 105)
        print(f"{'K':>4} | {'d0 mean':>8} {'min':>7} {'max':>7} | "
              f"{'n_bnd':>6} {'n_int':>6} | {'main':>6} {'m_bnd':>6} {'m_int':>6}")
        print("-" * 105)
        for r in summary_rows:
            print(f"{r[0]:>4} | {r[1]:>8.3f} {r[2]:>7.3f} {r[3]:>7.3f} | "
                  f"{r[4]:>6.1f} {r[5]:>6.1f} | {r[6]:>6.1f} {r[7]:>6.1f} {r[8]:>6.1f}")
        print("=" * 105 + "\n")

        return results
