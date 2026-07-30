import warnings
from typing import Callable, Tuple, Union
import time
import numpy as np
import scipy.sparse as sp

from tdgl_LSFD_lib.mesh.mesh import Mesh

# ----------------------------------------------------------------
# 1. Finite Volume operators for numerical integration and flux calculation
# ----------------------------------------------------------------
class FVMIntegrator:
    """
    Finite Volume Method integrator for flux and integral computations.

    Provides 5 methods to compute integrals and verify conservation laws:
    1. Surface flux through boundary (vector fields)
    2. Volume integral via divergence (div * Voronoi areas)
    3. Flux through triangle normals
    4. Scalar integrals via Simpson quadrature on triangles
    5. Scalar integrals via Voronoi cells
    """

    def __init__(self, mesh: Mesh):
        """
        Initialize FVM integrator with mesh data.

        Args:
            mesh: Mesh object containing sites, elements, dual_mesh, etc.
        """
        self.mesh = mesh
        self.sites = mesh.sites
        self.triangles = mesh.tri_mesh.triangles
        self.boundary_indices = mesh.boundary_indices

        # TriMesh data
        self.edges = mesh.tri_mesh.edges
        self.edge_lengths = mesh.tri_mesh.edge_lengths
        self.edge_directions = mesh.tri_mesh.edge_directions
        self.normalized_edge_directions = self.edge_directions / self.edge_lengths[:, np.newaxis]
        self.boundary_edge_indices = mesh.tri_mesh.boundary_edge_indices
        self.boundary_edges = self.edges[self.boundary_edge_indices]

        self.tri_areas = mesh.tri_mesh.tri_areas
        self.tri_centroids = mesh.tri_mesh.tri_centroids
        self.tri_to_edges = mesh.tri_mesh.tri_to_edges
        self.tri_edge_normals = mesh.tri_mesh.tri_edge_normals
        self.boundary_site_indices = mesh.tri_mesh.boundary_site_indices
        self.boundary_site_normals = mesh.tri_mesh.boundary_site_normals
        self.boundary_edge_normals = mesh.tri_mesh.boundary_edge_normals

        # === DualMesh data ===
        self.voronoi_areas = mesh.dual_mesh.dual_areas
        self.voronoi_polygons = mesh.dual_mesh.voronoi_polygons
        self.dual_edge_lengths = mesh.dual_mesh.dual_edge_lengths
        self.dual_edge_directions = mesh.dual_mesh.dual_edge_directions

        self.boundary_sites_weights = self.compute_edge_lenghts_for_boundary_sites()
        self.voronoi_flux_matrix = self.build_voronoi_flux_matrix()
        self.edge_interp_matrix = self.build_edge_interpolation()
        self.tri_flux_matrix = self.build_triangle_flux_matrix()
        self.tri_vertex_weights = self.build_tri_vertex_weights()

        self.boundary_flux_from_cell = np.zeros(len(self.sites), dtype=self.sites.dtype)

    def compute_edge_lenghts_for_boundary_sites(self):

        L = np.zeros(len(self.boundary_indices), dtype= self.sites.dtype)
        boundary_edges = self.edges[self.boundary_edge_indices]
        boundary_edge_lengths = self.edge_lengths[self.boundary_edge_indices]

        np.add.at(L, boundary_edges[:, 0], 0.5 * boundary_edge_lengths)
        np.add.at(L, boundary_edges[:, 1], 0.5 * boundary_edge_lengths)

        return 0.5 * L

    def build_voronoi_flux_matrix(self):

        # Indices for each edge
        edge_indices = np.arange(len(self.edges))  # возвращает массив индексов ребер
        # Compute the weights for each edge
        weights = self.dual_edge_lengths  # возвращает значения длин сторон ячейки вороного; eij - ребро триангуляции между i и j вершинами
        # Rows and cols to update                                                                sij - сторона ячейки Вороного пересекающая, ребро eij
        edges0 = self.edges[:, 0]  # массив i вершин для ребер e_ij
        edges1 = self.edges[:, 1]  # массив j вершин для ребер e_ij
        rows = np.concatenate([edges0, edges1])
        cols = np.concatenate([edge_indices, edge_indices])
        values = np.concatenate(
            [weights, -weights]
        )
        return sp.csr_array(
            (values, (rows, cols)), shape=(len(self.sites), len(self.edges))
        )

    def build_edge_interpolation(self) -> sp.csr_array:
        """
        Матрица интерполяции J из вершин на рёбра.

        J_edge = EdgeInterp @ J_vertex

        Returns:
            EdgeInterp: sparse (E, N) — среднее значение на ребре
        """
        E = len(self.edges)
        edges0 = self.edges[:, 0]
        edges1 = self.edges[:, 1]

        # Для каждого ребра: J_edge = 0.5 * (J_v0 + J_v1)
        rows = np.concatenate([np.arange(E), np.arange(E)])
        cols = np.concatenate([edges0, edges1])
        weights = 0.5 * np.ones(E)
        values = np.concatenate([weights, weights])

        return sp.csr_array((values, (rows, cols)), shape=(E, len(self.sites)))

    def build_tri_vertex_weights(self) -> np.ndarray:
        """
        Предвычисляет вес каждой вершины для интеграла по треугольникам.

        weight[v] = Σ (A_tri / 3) для всех треугольников содержащих вершину v

        Returns:
            vertex_weights: (N,) — вес для каждой вершины
        """
        N = len(self.sites)

        # Метод 1: np.add.at (быстро!)
        vertex_weights = np.zeros(N, dtype=self.sites.dtype)
        tri_area_third = self.tri_areas / 3  # (M,)

        # Каждая вершина получает A_tri/3 от каждого треугольника
        np.add.at(vertex_weights, self.triangles[:, 0], tri_area_third)
        np.add.at(vertex_weights, self.triangles[:, 1], tri_area_third)
        np.add.at(vertex_weights, self.triangles[:, 2], tri_area_third)

        return vertex_weights

    def build_triangle_flux_matrix(self) -> sp.csr_array:
        """
        Матрица потока через рёбра треугольников.

        flux_tri = TriFlux @ J_edge

        где J_edge = [Jx_edge, Jy_edge] (2E,)

        Returns:
            TriFlux: sparse (M, 2*E) — поток через каждое ребро треугольника
        """
        M = len(self.triangles)
        E = len(self.edges)

        # Индексы рёбер для каждого треугольника: (M, 3)
        tri_to_edges = self.tri_to_edges

        # Нормали для каждого ребра треугольника: (M, 3, 2)
        normals = self.tri_edge_normals

        # Длины рёбер: (E,)
        edge_lengths = self.edge_lengths

        # Для каждого треугольника и каждого из 3 рёбер:
        #   flux = (Jx_edge * n_x + Jy_edge * n_y) * |e|

        rows = np.repeat(np.arange(M), 3)  # (3M,)
        cols_edges = tri_to_edges.ravel()  # (3M,)

        # Коэффициенты для Jx и Jy
        coeff_x = normals[:, :, 0].ravel() * edge_lengths[cols_edges]  # (3M,)
        coeff_y = normals[:, :, 1].ravel() * edge_lengths[cols_edges]  # (3M,)

        # Строим матрицу для [Jx_edge, Jy_edge] → (M, 2*E)
        cols_x = cols_edges  # Индексы для Jx_edge
        cols_y = cols_edges + E  # Индексы для Jy_edge

        rows_combined = np.concatenate([rows, rows])
        cols_combined = np.concatenate([cols_x, cols_y])
        values_combined = np.concatenate([coeff_x, coeff_y])

        return sp.csr_array(
            (values_combined, (rows_combined, cols_combined)),
            shape=(M, 2 * E)
        )



# Public functions
#
    # ========================================================================
    # 1. SURFACE FLUX THROUGH BOUNDARY
    # ========================================================================

    def compute_surface_flux_edges(self, J_x: np.ndarray, J_y: np.ndarray) -> float:
        """
        Compute total flux through boundary: ∮ (J·n) dl

        Args:
            J_x: (N,) x-component of vector field at vertices
            J_y: (N,) y-component of vector field at vertices

        Returns:
            Total flux through boundary
        """
        # Interpolate J to boundary edges (average of endpoints)
        Jx0 = J_x[self.boundary_edges[:, 0]]
        Jx1 = J_x[self.boundary_edges[:, 1]]
        Jy0 = J_y[self.boundary_edges[:, 0]]
        Jy1 = J_y[self.boundary_edges[:, 1]]

        J_dot_n = (Jx0 + Jx1) * self.boundary_edge_normals[:, 0] + (Jy0 + Jy1) * self.boundary_edge_normals[:, 1]

        return np.sum(0.5 * J_dot_n * self.edge_lengths[self.boundary_edge_indices])

    def compute_surface_flux_sites(self, J_x: np.ndarray, J_y: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute total flux through boundary: ∮ (J·n) dl

        Args:
            J_x: (N,) x-component of vector field at vertices
            J_y: (N,) y-component of vector field at vertices

        Returns:
            Total flux through boundary
        """
        # Interpolate J to boundary edges (average of endpoints)
        Jx = J_x[self.boundary_indices]
        Jy = J_y[self.boundary_indices]
        J_dot_n = Jx * self.boundary_site_normals[:, 0] + Jy * self.boundary_site_normals[:, 1]

        return float(np.sum(J_dot_n * self.boundary_sites_weights)), float(np.max(J_dot_n)), float(np.mean(J_dot_n))

    # ========================================================================
    # 2. FLUX THROUGH VORONOI CELLS
    # ========================================================================

    def compute_voronoi_flux(self, J_x: np.ndarray, J_y: np.ndarray) -> np.ndarray:

        Jx_edge = self.edge_interp_matrix @ J_x
        Jy_edge = self.edge_interp_matrix @ J_y

        Je = Jx_edge * self.normalized_edge_directions[:, 0] + Jy_edge * self.normalized_edge_directions[:, 1]

        Jx = J_x[self.boundary_indices]
        Jy = J_y[self.boundary_indices]
        J_dot_n = Jx * self.boundary_site_normals[:, 0] + Jy * self.boundary_site_normals[:, 1]

        inner_flux_from_cell = self.voronoi_flux_matrix @ Je
        self.boundary_flux_from_cell[self.boundary_indices] = J_dot_n * self.boundary_sites_weights

        total_flux = inner_flux_from_cell + self.boundary_flux_from_cell

        return total_flux

    # ========================================================================
    # 3. FLUX THROUGH TRIANGLES
    # ========================================================================

    def compute_triangle_flux(self, J_x: np.ndarray, J_y: np.ndarray) -> np.ndarray:
        """
        Вычисляет поток через КАЖДЫЙ треугольник (если нужно).

        Args:
            J_x: (N,) x-компонента тока в вершинах
            J_y: (N,) y-компонента тока в вершинах

        Returns:
            tri_flux: (M,) поток через каждый треугольник
        """
        # 1. Интерполяция на рёбра
        Jx_edge = self.edge_interp_matrix @ J_x  # (E,)
        Jy_edge = self.edge_interp_matrix @ J_y  # (E,)

        # 2. Поток через треугольники
        J_edge_vec = np.concatenate([Jx_edge, Jy_edge])  # (2*E,)
        tri_flux = self.tri_flux_matrix @ J_edge_vec  # (M,)

        return tri_flux

    # ========================================================================
    # 4. VOLUME INTEGRAL VIA DIVERGENCE
    # ========================================================================

    def compute_divergence_integral(self, div_J: np.ndarray,
                                    method: str = 'voronoi') -> float:
        """
        Compute ∫ div(J) dS using Voronoi cells or triangles.

        Args:
            div_J: (N,) divergence at vertices
            method: 'voronoi' or 'triangles'

        Returns:
            Volume integral of divergence
        """
        if method == 'voronoi':
            # ∫ div(J) dS ≈ Σ div_J[i] * A_voronoi[i]
            return np.sum(div_J * self.voronoi_areas)

        elif method == 'triangles':
            # ∫ div(J) dS ≈ Σ (div_J[v0] + div_J[v1] + div_J[v2]) / 3 * A_tri
            tri_integral = np.sum(div_J * self.tri_vertex_weights)
            return np.sum(tri_integral)

        else:
            raise ValueError(f"Unknown method: {method}")

    # ========================================================================
    # 5. CHECK CONSERVATION
    # ========================================================================

    def global_conservation_check(self, J_x: np.ndarray, J_y: np.ndarray, div_J: np.ndarray) -> Tuple[float, float, float, float, float, float]:

        # 1. Поток через границу edges
        surface_flux_edges = self.compute_surface_flux_edges(J_x, J_y)

        # 2. Поток через границу sites
        surface_flux_sites, Jn_max, Jn_mean = self.compute_surface_flux_sites(J_x, J_y)

        # 3. Интеграл от дивергенции по всей области (Вороной)
        div_voronoi = self.compute_divergence_integral(div_J, method='voronoi')

        # 4. Интеграл от дивергенции по всей области (треугольники)
        div_triangles  = self.compute_divergence_integral(div_J, method='triangles')

        return surface_flux_edges, surface_flux_sites, div_voronoi, div_triangles, Jn_mean, Jn_max

    def local_conservation_check(self, J_x: np.ndarray, J_y: np.ndarray, div_J: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Вычисляет ВСЕ 5 интегралов и проверяет сохранение.

        Args:
            J_x, J_y: Компоненты тока в вершинах
            div_J: Дивергенция в вершинах
            tolerance: Порог для проверки сохранения

        Returns:
            results: dict с всеми значениями и флагом passed
        """

        # 1. Поток через ячейки Вороного
        voronoi_flux = self.compute_voronoi_flux(J_x, J_y)
        # 2. Поток через треугольники
        tri_flux = self.compute_triangle_flux(J_x, J_y)
        # 3. Интеграл от дивергенции по всей области (Вороной)
        div_voronoi = div_J * self.voronoi_areas
        # 4. Интеграл от дивергенции по всей области (треугольники)
        div_triangles = (1/3) *  np.sum(div_J[self.triangles], axis=1) * self.tri_areas

        error_triangle_flux = abs(tri_flux - div_triangles) / np.max(abs(div_triangles))
        error_voronoi_flux = abs(voronoi_flux - div_voronoi) / np.max(abs(div_voronoi))

        max_triangle_error = np.max(error_triangle_flux)
        mean_triangle_error = np.mean(error_triangle_flux)
        max_voronoi_error = np.max(error_voronoi_flux)
        mean_voronoi_error = np.mean(error_voronoi_flux)

        return max_triangle_error, mean_triangle_error, max_voronoi_error, mean_voronoi_error


    def Energy_conservation_check(self, psi: np.ndarray, psi_derivatives: np.ndarray, s_applied: np.ndarray,
                          B:float, eta:float, gamma:float):

        s_x, s_y = s_applied
        sq_psi = psi * psi.conjugate()
        sq_Dx_psi = psi_derivatives[:, 0] * psi_derivatives[:, 0].conjugate()
        sq_Dy_psi = psi_derivatives[:, 1] * psi_derivatives[:, 1].conjugate()
        s_grad_psi = (s_x * (psi_derivatives[:, 0] + psi_derivatives[:, 0].conjugate())
                    + s_y * (psi_derivatives[:, 1] + psi_derivatives[:, 1].conjugate()))

        F = - sq_psi + 0.5 * sq_psi**2 + sq_Dx_psi + sq_Dy_psi + eta * s_grad_psi + B**2

        F_voronoi = self.compute_divergence_integral(F, method='voronoi')
        F_triangles  = self.compute_divergence_integral(F, method='triangles')

        return F_voronoi, F_triangles
