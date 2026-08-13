from typing import Sequence, Tuple, Union, Optional, List
from dataclasses import dataclass
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from typing import Dict  # если ещё не импортирован
import logging
logger = logging.getLogger(__name__)
from .tri_mesh import TriMesh
from .dual_mesh import DualMesh
from .util import (
    get_edges,
    triangle_areas,
    generate_voronoi_vertices,
    get_voronoi_polygon_indices,
    orient_convex_polygon,
    get_oriented_boundary,
    compute_voronoi_polygon_areas,
    convex_polygon_centroid,
)

from tqdm import tqdm
from meshpy import triangle as mpt
from shapely.geometry import Polygon as ShapelyPolygon

from .geometry import (
    close_curve,
)

@dataclass
class LSFDNeighbors:
    """Данные о соседях для метода LSFD."""
    indices: np.ndarray  # (N, K)
    coords: np.ndarray  # (N, K, 2)
    distances: np.ndarray  # (N, K)
    lsfd_edge_vectors: np.ndarray # (N, K, 2)
    indices_with_ghosts: np.ndarray
    coords_with_ghosts: np.ndarray
    distances_with_ghosts: np.ndarray
    lsfd_edge_vectors_with_ghosts: np.ndarray
    ghost_dist: float
    ghost_coords: np.ndarray
    indices_of_own_ghost_point_in_sites_with_ghosts: np.ndarray

class Mesh:
    def __init__(
            self,
            sites: np.ndarray,
            elements: np.ndarray,
            boundary_indices: np.ndarray,
            # Остальные параметры опциональны
            tri_mesh: Optional[TriMesh] = None,
            dual_mesh: Optional[DualMesh] = None,
            lsfd_neighbors: Optional[LSFDNeighbors] = None,
            n_lsfd_neighbors: int = 15,
            ghost_coeff: float = 1,
    ):
        self.sites = np.asarray(sites, dtype=np.float64).squeeze()
        self.elements = np.asarray(elements, dtype=np.int64).squeeze()
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)

        self.tri_mesh = tri_mesh
        self.dual_mesh = dual_mesh
        self.lsfd_neighbors = lsfd_neighbors
        self.n_lsfd_neighbors = n_lsfd_neighbors
        self.ghost_coeff = ghost_coeff

        self._center_of_mass: Optional[Tuple[float, float]] = None

    # ========================================================================
    # БАЗОВЫЕ СВОЙСТВА
    # ========================================================================

    @property
    def x(self) -> np.ndarray:
        return self.sites[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.sites[:, 1]

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def n_elements(self) -> int:
        return len(self.elements)


    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """Центр масс области."""
        if self._center_of_mass is None:
            tri_areas = triangle_areas(self.sites, self.elements)
            tri_centroids = self.sites[self.elements].mean(axis=1)
            com = np.average(tri_centroids, axis=0, weights=tri_areas)
            self._center_of_mass = tuple(com)
        return self._center_of_mass

    # ========================================================================
    # УТИЛИТЫ
    # ========================================================================

    def closest_site(self, xy: Tuple[float, float]) -> int:
        """Индекс ближайшей вершины."""
        return np.argmin(np.linalg.norm(self.sites - np.atleast_2d(xy), axis=1))

    def check_quality(self) -> Dict[str, float]:
        """Расширенная проверка: отдельно для границы и внутренности, плюс анализ углов."""
        sites = self.sites
        elements = self.elements
        edges, is_boundary_edge = get_edges(elements)
        edge_lengths = np.linalg.norm(
            sites[edges[:, 1]] - sites[edges[:, 0]], axis=1
        )

        boundary = self.boundary_indices
        is_boundary_site = np.zeros(len(sites), dtype=bool)
        is_boundary_site[boundary] = True

        # Рёбра: граничные, внутренние, переходные
        both_boundary = is_boundary_site[edges[:, 0]] & is_boundary_site[edges[:, 1]]
        both_interior = ~is_boundary_site[edges[:, 0]] & ~is_boundary_site[edges[:, 1]]
        mixed = ~both_boundary & ~both_interior  # одно ребро на границе, другое внутри

        # === 1. Углы в треугольниках ===
        A = sites[elements[:, 0]]
        B = sites[elements[:, 1]]
        C = sites[elements[:, 2]]

        AB = B - A
        AC = C - A
        BC = C - B

        len_AB = np.linalg.norm(AB, axis=1)
        len_AC = np.linalg.norm(AC, axis=1)
        len_BC = np.linalg.norm(BC, axis=1)

        # Скалярные произведения для теоремы косинусов
        dot_A = np.sum(AB * AC, axis=1)
        dot_B = np.sum(-AB * BC, axis=1)
        dot_C = np.sum(AC * BC, axis=1)  # эквивалентно dot(-AC, -BC)

        # Косинусы углов (с защитой от деления на ноль)
        cos_A = dot_A / (len_AB * len_AC + 1e-15)
        cos_B = dot_B / (len_AB * len_BC + 1e-15)
        cos_C = dot_C / (len_AC * len_BC + 1e-15)

        # Клиппинг для защиты от ошибок floating point в arccos
        cos_A = np.clip(cos_A, -1.0, 1.0)
        cos_B = np.clip(cos_B, -1.0, 1.0)
        cos_C = np.clip(cos_C, -1.0, 1.0)

        # Углы в градусах
        ang_A = np.degrees(np.arccos(cos_A))
        ang_B = np.degrees(np.arccos(cos_B))
        ang_C = np.degrees(np.arccos(cos_C))

        angles = np.column_stack((ang_A, ang_B, ang_C))
        min_angles = np.min(angles, axis=1)
        max_angles = np.max(angles, axis=1)

        # Пороги для "плохих" треугольников (slivers)
        small_angle_thresh = 15.0  # градусов
        large_angle_thresh = 150.0  # градусов

        n_bad_small = int(np.sum(min_angles < small_angle_thresh))
        n_bad_large = int(np.sum(max_angles > large_angle_thresh))
        n_bad_total = int(np.sum((min_angles < small_angle_thresh) | (max_angles > large_angle_thresh)))

        # === 2. Формирование словаря результатов ===
        result = {
            # Глобальные
            'edge_ratio_global': float(np.max(edge_lengths) / (np.min(edge_lengths) + 1e-15)),

            # Граничные рёбра
            'edge_len_boundary_mean': float(np.mean(edge_lengths[both_boundary])),
            'edge_len_boundary_min': float(np.min(edge_lengths[both_boundary])),
            'edge_len_boundary_max': float(np.max(edge_lengths[both_boundary])),
            'edge_ratio_boundary': float(
                np.max(edge_lengths[both_boundary]) / (np.min(edge_lengths[both_boundary]) + 1e-15)),

            # Внутренние рёбра
            'edge_len_interior_mean': float(np.mean(edge_lengths[both_interior])),
            'edge_len_interior_min': float(np.min(edge_lengths[both_interior])),
            'edge_len_interior_max': float(np.max(edge_lengths[both_interior])),
            'edge_ratio_interior': float(
                np.max(edge_lengths[both_interior]) / (np.min(edge_lengths[both_interior]) + 1e-15)),

            # Переходные рёбра (граница ↔ внутренность) — САМЫЕ ВАЖНЫЕ!
            'edge_len_mixed_mean': float(np.mean(edge_lengths[mixed])),
            'edge_len_mixed_min': float(np.min(edge_lengths[mixed])),
            'edge_len_mixed_max': float(np.max(edge_lengths[mixed])),
            'edge_ratio_mixed': float(np.max(edge_lengths[mixed]) / (np.min(edge_lengths[mixed]) + 1e-15)),

            # Отношение граница/внутренность
            'ratio_boundary_to_interior': float(
                np.mean(edge_lengths[both_interior]) / (np.mean(edge_lengths[both_boundary]) + 1e-15)
            ),

            # Число рёбер каждого типа
            'n_boundary_edges': int(np.sum(both_boundary)),
            'n_interior_edges': int(np.sum(both_interior)),
            'n_mixed_edges': int(np.sum(mixed)),

            # Статистика углов
            'angle_global_min': float(np.min(min_angles)),
            'angle_global_max': float(np.max(max_angles)),
            'angle_mean_min': float(np.mean(min_angles)),
            'angle_mean_max': float(np.mean(max_angles)),

            # Плохие треугольники
            'n_bad_small_angle': n_bad_small,
            'n_bad_large_angle': n_bad_large,
            'n_bad_total': n_bad_total,
            'n_triangles': int(len(elements)),
        }

        return result

    def print_quality(self, label: str = "") -> None:
        """Подробный вывод качества сетки."""
        q = self.check_quality()
        prefix = f"[{label}]\n" if label else ""

        print(f"{prefix}"
              f"  Global edge ratio: {q['edge_ratio_global']:.1f}\n"
              f"  Boundary edges:  len=[{q['edge_len_boundary_min']:.4f}, {q['edge_len_boundary_max']:.4f}], "
              f"mean={q['edge_len_boundary_mean']:.4f}, ratio={q['edge_ratio_boundary']:.2f} "
              f"(n={q['n_boundary_edges']})\n"
              f"  Interior edges:  len=[{q['edge_len_interior_min']:.4f}, {q['edge_len_interior_max']:.4f}], "
              f"mean={q['edge_len_interior_mean']:.4f}, ratio={q['edge_ratio_interior']:.2f} "
              f"(n={q['n_interior_edges']})\n"
              f"  Mixed edges:     len=[{q['edge_len_mixed_min']:.4f}, {q['edge_len_mixed_max']:.4f}], "
              f"mean={q['edge_len_mixed_mean']:.4f}, ratio={q['edge_ratio_mixed']:.2f} "
              f"(n={q['n_mixed_edges']})\n"
              f"  Boundary/Interior ratio: {q['ratio_boundary_to_interior']:.1f}\n"
              f"  Angles (deg):    min={q['angle_global_min']:.2f}, max={q['angle_global_max']:.2f}, "
              f"mean_min={q['angle_mean_min']:.2f}, mean_max={q['angle_mean_max']:.2f}\n"
              f"  Bad triangles:   {q['n_bad_total']} / {q['n_triangles']} "
              f"(small < 15°: {q['n_bad_small_angle']}, large > 150°: {q['n_bad_large_angle']})"
              )

    def smooth(
            self,
            iterations: int = 1,
            create_submesh: bool = True,
            verbose: bool = True,
    ) -> "Mesh":
        """
        Laplacian smoothing: перемещает внутренние вершины в среднее соседей.

        Args:
            iterations: Число итераций сглаживания.
            create_submesh: Если False, возвращает минимальный Mesh (только sites/elements)
                           для использования внутри make_mesh(). Если True — создаёт полный Mesh.
            verbose: Если True, выводит метрики качества на каждом шаге.

        Returns:
            Mesh с обновлёнными координатами.
        """
        sites = self.sites.copy()
        elements = self.elements
        edges, _ = get_edges(elements)
        n = len(sites)
        boundary = self.boundary_indices

        # === Проверка качества ДО сглаживания ===
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Laplacian smoothing: {iterations} iterations")
            print(f"{'=' * 70}")

            # Создаём временный Mesh для проверки (только sites/elements)
            temp_mesh = Mesh(
                sites=sites, elements=elements, boundary_indices=boundary,
                n_lsfd_neighbors=self.n_lsfd_neighbors,
            )
            temp_mesh.print_quality(label="Before smooth")

        # === История метрик (для анализа) ===
        quality_history = []

        # === Сглаживание: только координаты ===
        for it in range(iterations):
            num_neighbors = np.bincount(edges.ravel(), minlength=n)

            new_sites = np.zeros_like(sites)
            for dim in range(2):
                vals = sites[edges[:, 1], dim]
                new_sites[:, dim] += np.bincount(edges[:, 0], weights=vals, minlength=n)
                vals = sites[edges[:, 0], dim]
                new_sites[:, dim] += np.bincount(edges[:, 1], weights=vals, minlength=n)

            new_sites /= num_neighbors[:, np.newaxis]
            new_sites[boundary] = sites[boundary]  # Фиксация границы
            sites = new_sites

            # === Проверка качества на каждом шаге ===
            if verbose and ((it + 1) % max(1, iterations // 10) == 0 or it == iterations - 1):
                temp_mesh = Mesh(
                    sites=sites, elements=elements, boundary_indices=boundary,
                    n_lsfd_neighbors=self.n_lsfd_neighbors,
                )
                temp_mesh.print_quality(label=f"Iter {it + 1}/{iterations}")
                quality_history.append(temp_mesh.check_quality())

        # === Возвращаем результат ===
        if create_submesh:
            result = Mesh.from_triangulation(
                sites, elements, n_lsfd_neighbors=self.n_lsfd_neighbors
            )
        else:
            result = Mesh(
                sites=sites, elements=elements, boundary_indices=boundary,
                n_lsfd_neighbors=self.n_lsfd_neighbors,
            )

        # === Финальная проверка качества ===
        if verbose:
            result.print_quality(label="After smooth")
            print(f"{'=' * 70}\n")


        return result

    @staticmethod
    def shoelace_centroid(poly: np.ndarray) -> Tuple[float, float]:
        """Центроид выпуклого многоугольника по формуле шнуровки."""
        x, y = poly[:, 0], poly[:, 1]
        cr = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = 0.5 * cr.sum()
        if abs(a) < 1e-15:
            return float(x.mean()), float(y.mean())
        cx = ((x + np.roll(x, -1)) * cr).sum() / (6 * a)
        cy = ((y + np.roll(y, -1)) * cr).sum() / (6 * a)
        return float(cx), float(cy)

    def _retriangulate(
            self,
            sites: np.ndarray,
            elements: np.ndarray,
            min_angle: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Перестраивает constrained Delaunay триангуляцию после сдвига узлов.
        Сохраняет границу и отверстия, не добавляет Steiner-точки на границе.
        """
        edges, is_bnd_edge = get_edges(elements)
        bnd_edges = edges[is_bnd_edge]
        loops = get_oriented_boundary(sites, bnd_edges)

        if not loops:
            return sites, elements

        # Внешний контур — петля с максимальной площадью
        loop_areas = []
        for loop in loops:
            p = sites[loop]
            area = 0.5 * np.abs(np.sum(
                p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
            ))
            loop_areas.append(area)
        outer_idx = int(np.argmax(loop_areas))

        facets_list, hole_pts = [], []
        for k, loop in enumerate(loops):
            idx = np.asarray(loop, dtype=int)
            facets_list.append(np.column_stack([idx, np.roll(idx, -1)]))
            if k != outer_idx:
                try:
                    poly = ShapelyPolygon(sites[loop])
                    hole_pts.append(np.array(poly.representative_point().coords[0]))
                except Exception:
                    pass

        info = mpt.MeshInfo()
        info.set_points(sites)
        info.set_facets(np.vstack(facets_list))
        if hole_pts:
            info.set_holes(hole_pts)

        try:
            mesh = mpt.build(
                info,
                min_angle=min_angle,
                quality=True,
                allow_boundary_steiner=False,
            )
            return np.asarray(mesh.points, dtype=float), np.asarray(mesh.elements, dtype=int)
        except Exception as exc:
            # Fallback без quality
            try:
                mesh = mpt.build(info, allow_boundary_steiner=False)
                return np.asarray(mesh.points, dtype=float), np.asarray(mesh.elements, dtype=int)
            except Exception:
                # Если ничего не работает — возвращаем исходные данные
                print(f"Warning: _retriangulate failed: {exc}. Keeping old triangulation.")
                return sites, elements

    def smooth_lloyd(
            self,
            iterations: int = 20,
            step: float = 0.5,
            max_disp_factor: float = 0.3,
            convergence_tol: float = 1e-3,
            min_angle: float = 30.0,
            create_submesh: bool = True,
            verbose: bool = True,
    ) -> "Mesh":
        """
        Lloyd's algorithm (CVT) с автоматическим критерием сходимости.

        Критерий сходимости (останавливается досрочно, если оба выполнены):
          1. Относительное изменение средней длины mixed edges < convergence_tol
          2. Максимальное смещение узлов за итерацию < max_disp_factor * h_bnd * convergence_tol

        Args:
            iterations: Максимальное число итераций.
            step: Доля смещения (0 < step <= 1). 0.5 — безопасно.
            max_disp_factor: Макс. смещение за шаг в долях h_bnd.
            convergence_tol: Порог сходимости (относительный).
            min_angle: Минимальный угол при перетриангуляции.
            create_submesh: Если False, возвращает минимальный Mesh.
            verbose: Выводить метрики.

        Returns:
            Mesh с обновлёнными координатами.
        """
        sites = self.sites.copy()
        elements = self.elements.copy()
        boundary = self.boundary_indices.copy()
        is_int = np.ones(len(sites), dtype=bool)
        is_int[boundary] = False
        int_idx = np.where(is_int)[0]

        # Характерный размер граничного ребра
        edges, is_bnd_edge = get_edges(elements)
        bnd_edges = edges[is_bnd_edge]
        if len(bnd_edges) > 0:
            h_bnd = float(np.mean(
                np.linalg.norm(sites[bnd_edges[:, 0]] - sites[bnd_edges[:, 1]], axis=1)
            ))
        else:
            h_bnd = 0.1
        max_disp = max_disp_factor * h_bnd

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Lloyd's (CVT) smoothing: max {iterations} iterations, step={step}, "
                  f"tol={convergence_tol}")
            print(f"{'=' * 70}")
            temp_mesh = Mesh(
                sites=sites, elements=elements, boundary_indices=boundary,
                n_lsfd_neighbors=self.n_lsfd_neighbors,
            )
            temp_mesh.print_quality(label="Before Lloyd")

        prev_mixed_mean = None
        pbar = tqdm(range(iterations), desc="Lloyd smoothing", disable=not verbose)

        for it in pbar:
            # 1. Ячейки Вороного из ТЕКУЩЕЙ триангуляции
            verts = generate_voronoi_vertices(sites, elements)  # (M, 2)
            polys = get_voronoi_polygon_indices(elements, len(sites))

            # 2. Шаг Ллойда: только внутренние узлы
            target = sites.copy()
            for i in int_idx:
                cell = orient_convex_polygon(verts[polys[i]])
                if len(cell) >= 3:
                    cx, cy = self.shoelace_centroid(cell)
                    target[i] = [cx, cy]

            # 3. Смещение с ограничением
            disp = step * (target - sites)
            disp[boundary] = 0.0  # граница фиксирована

            # Ограничение максимального смещения
            dn = np.linalg.norm(disp, axis=1)
            scale = np.where(dn > max_disp, max_disp / (dn + 1e-15), 1.0)
            actual_disp = disp * scale[:, np.newaxis]
            sites += actual_disp

            max_actual_disp = float(np.max(dn))

            # 4. Перетриангуляция
            sites, elements = self._retriangulate(sites, elements, min_angle=min_angle)
            boundary = Mesh._find_boundary_indices(elements)

            # 5. Критерий сходимости
            edges_new, is_bnd_new = get_edges(elements)
            is_bnd_site = np.zeros(len(sites), dtype=bool)
            is_bnd_site[boundary] = True
            mixed = ~is_bnd_site[edges_new[:, 0]] & ~is_bnd_site[edges_new[:, 1]] == False
            mixed = ~(is_bnd_site[edges_new[:, 0]] & is_bnd_site[edges_new[:, 1]]) & \
                    ~(~is_bnd_site[edges_new[:, 0]] & ~is_bnd_site[edges_new[:, 1]])
            mixed_lens = np.linalg.norm(
                sites[edges_new[mixed, 1]] - sites[edges_new[mixed, 0]], axis=1
            )

            curr_mixed_mean = float(np.mean(mixed_lens)) if len(mixed_lens) > 0 else 0.0
            curr_mixed_cv = float(np.std(mixed_lens) / (curr_mixed_mean + 1e-15)) if len(mixed_lens) > 0 else 0.0

            # Обновление прогресс-бара
            pbar.set_postfix({
                'max_disp': f"{max_actual_disp:.4f}",
                'mixed_mean': f"{curr_mixed_mean:.4f}",
                'mixed_cv': f"{curr_mixed_cv:.3f}",
            })

            # Проверка сходимости
            if prev_mixed_mean is not None:
                rel_change = abs(curr_mixed_mean - prev_mixed_mean) / (prev_mixed_mean + 1e-15)
                if rel_change < convergence_tol and max_actual_disp < max_disp * convergence_tol:
                    if verbose:
                        print(f"\nConverged at iteration {it + 1}: "
                              f"rel_change={rel_change:.2e}, max_disp={max_actual_disp:.2e}")
                    break
            prev_mixed_mean = curr_mixed_mean

        pbar.close()

        # === Возвращаем результат ===
        if create_submesh:
            result = Mesh.from_triangulation(
                sites, elements, n_lsfd_neighbors=self.n_lsfd_neighbors
            )
        else:
            result = Mesh(
                sites=sites, elements=elements, boundary_indices=boundary,
                n_lsfd_neighbors=self.n_lsfd_neighbors,
            )

        if verbose:
            result.print_quality(label="After Lloyd")
            print(f"{'=' * 70}\n")

        return result

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            show_sites: bool = True,
            show_triangles: bool = False,
            show_voronoi: bool = True,
            site_color: Optional[str] = None,
            triangle_color: Optional[str] = "k",
            voronoi_color: Optional[str] = "gray",
            linewidth: float = 0.75,
            marker: str = ".",
    ) -> plt.Axes:
        """Визуализация сетки."""
        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect("equal")

        x, y = self.sites.T

        if show_triangles:
            ax.triplot(x, y, self.elements, color=triangle_color, lw=linewidth)

        if show_voronoi:
            for poly in self.dual_mesh.voronoi_polygons:
                ax.plot(*close_curve(poly).T, color=voronoi_color, lw=linewidth)

        if show_sites:
            ax.plot(x, y, marker=marker, ls="", color=site_color)

        return ax

    # ========================================================================
    # HDF5: ТОЛЬКО ЗДЕСЬ (убрано из TriMesh/DualMesh)
    # ========================================================================

    def to_hdf5(self, h5group: h5py.Group, compress: bool = True, **meta) -> None:
        """Сохранить Mesh в HDF5. Каждая группа создаётся ровно ОДИН раз."""
        compression = "gzip" if compress else None

        # === Базовые данные ===
        h5group.create_dataset("sites", data=self.sites, compression=compression)
        h5group.create_dataset("elements", data=self.elements, compression=compression)
        h5group.create_dataset("boundary_indices", data=self.boundary_indices, compression=compression)

        # === Глобальные параметры ===
        h5group.attrs["n_lsfd_neighbors"] = self.n_lsfd_neighbors
        h5group.attrs["ghost_coeff"] = self.ghost_coeff
        h5group.attrs["h_boundary"] = float(np.mean(
            self.tri_mesh.edge_lengths[self.tri_mesh.boundary_edge_indices]))
        for k, v in meta.items():
            h5group.attrs[k] = v

        # === TriMesh ===
        tri_grp = h5group.create_group("tri_mesh")
        for name in ["edges", "boundary_edge_indices", "edge_lengths", "edge_directions",
                     "edge_midpoints", "tri_areas", "tri_centroids", "tri_to_edges",
                     "edges_to_tri", "tri_edge_normals", "boundary_site_normals",
                     "boundary_site_indices", "boundary_edge_normals"]:
            tri_grp.create_dataset(name, data=getattr(self.tri_mesh, name), compression=compression)

        # === DualMesh ===
        dual_grp = h5group.create_group("dual_mesh")
        for name in ["dual_sites", "dual_areas", "dual_edge_lengths", "dual_edge_directions"]:
            dual_grp.create_dataset(name, data=getattr(self.dual_mesh, name), compression=compression)
        if self.dual_mesh.voronoi_polygons:
            polygons_flat = np.concatenate(self.dual_mesh.voronoi_polygons, axis=0)
            split_indices = np.cumsum([len(p) for p in self.dual_mesh.voronoi_polygons[:-1]])
            dual_grp.create_dataset("voronoi_polygons_flat", data=polygons_flat, compression=compression)
            dual_grp.create_dataset("voronoi_split_indices", data=split_indices, compression=compression)

        # === LSFD Neighbors (ПОЛНОСТЬЮ, один раз) ===
        lsfd_grp = h5group.create_group("lsfd_neighbors")
        nb = self.lsfd_neighbors
        lsfd_grp.create_dataset("indices", data=nb.indices, compression=compression)
        lsfd_grp.create_dataset("coords", data=nb.coords, compression=compression)
        lsfd_grp.create_dataset("distances", data=nb.distances, compression=compression)
        lsfd_grp.create_dataset("lsfd_edge_vectors", data=nb.lsfd_edge_vectors, compression=compression)
        lsfd_grp.create_dataset("indices_with_ghosts", data=nb.indices_with_ghosts, compression=compression)
        lsfd_grp.create_dataset("coords_with_ghosts", data=nb.coords_with_ghosts, compression=compression)
        lsfd_grp.create_dataset("distances_with_ghosts", data=nb.distances_with_ghosts, compression=compression)
        lsfd_grp.create_dataset("lsfd_edge_vectors_with_ghosts",
                                data=nb.lsfd_edge_vectors_with_ghosts, compression=compression)
        lsfd_grp.create_dataset("ghost_coords", data=nb.ghost_coords, compression=compression)
        lsfd_grp.create_dataset("indices_of_own_ghost",
                                data=nb.indices_of_own_ghost_point_in_sites_with_ghosts,
                                compression=compression)
        lsfd_grp.attrs["ghost_dist"] = float(nb.ghost_dist)

        # === Метаданные ===
        h5group.attrs["n_sites"] = self.n_sites
        h5group.attrs["n_elements"] = self.n_elements

    @classmethod
    def from_hdf5(cls, h5group: h5py.Group,
                  n_lsfd_neighbors: Optional[int] = None,
                  ghost_coeff: Optional[float] = None) -> "Mesh":
        """Загрузить Mesh. Если K/ghost изменены или файл старый — пересобрать соседей."""
        for name in ["sites", "elements", "boundary_indices"]:
            if name not in h5group:
                raise IOError(f"Missing required dataset: {name}")
        sites = np.array(h5group["sites"])
        elements = np.array(h5group["elements"], dtype=np.int64)
        boundary_indices = np.array(h5group["boundary_indices"], dtype=np.int64)

        saved_K = int(h5group.attrs.get("n_lsfd_neighbors", 15))
        saved_gc = float(h5group.attrs.get("ghost_coeff", 1.0))
        K = int(n_lsfd_neighbors) if n_lsfd_neighbors is not None else saved_K
        gc = float(ghost_coeff) if ghost_coeff is not None else saved_gc

        # === TriMesh ===
        tri_grp = h5group["tri_mesh"]
        tri_mesh = TriMesh.__new__(TriMesh)
        tri_mesh.sites = sites
        tri_mesh.triangles = elements
        for name in ["edges", "boundary_edge_indices", "edge_lengths", "edge_directions",
                     "edge_midpoints", "tri_areas", "tri_centroids", "tri_to_edges",
                     "edges_to_tri", "tri_edge_normals", "boundary_site_normals",
                     "boundary_site_indices"]:
            setattr(tri_mesh, name, np.array(tri_grp[name]))
        if "boundary_edge_normals" in tri_grp:
            tri_mesh.boundary_edge_normals = np.array(tri_grp["boundary_edge_normals"])
        else:
            _, _, tri_mesh.boundary_edge_normals = tri_mesh.compute_boundary_normals()
        tri_mesh.normalized_edge_directions = \
            tri_mesh.edge_directions / tri_mesh.edge_lengths[:, np.newaxis]

        # === DualMesh ===
        dual_grp = h5group["dual_mesh"]
        dual_mesh = DualMesh.__new__(DualMesh)
        for name in ["dual_sites", "dual_areas", "dual_edge_lengths", "dual_edge_directions"]:
            setattr(dual_mesh, name, np.array(dual_grp[name]))
        dual_mesh.tri_mesh = tri_mesh
        if "voronoi_polygons_flat" in dual_grp:
            polygons_flat = np.array(dual_grp["voronoi_polygons_flat"])
            split_indices = np.array(dual_grp["voronoi_split_indices"])
            dual_mesh.voronoi_polygons = np.split(polygons_flat, split_indices)
        else:
            dual_mesh.voronoi_polygons = []

        # === LSFD Neighbors: прочитать ИЛИ пересобрать ===
        lsfd_grp = h5group["lsfd_neighbors"]
        need_rebuild = (K != saved_K) or (gc != saved_gc) or ("indices_with_ghosts" not in lsfd_grp)
        if need_rebuild:
            logger.info(f"Mesh.from_hdf5: rebuilding LSFD neighbors (K={K}, saved K={saved_K})")
            lsfd_neighbors = cls.build_lsfd_neighbors(sites, boundary_indices, tri_mesh, K, gc)
        else:
            lsfd_neighbors = LSFDNeighbors(
                indices=np.array(lsfd_grp["indices"]),
                coords=np.array(lsfd_grp["coords"]),
                distances=np.array(lsfd_grp["distances"]),
                lsfd_edge_vectors=np.array(lsfd_grp["lsfd_edge_vectors"]),
                indices_with_ghosts=np.array(lsfd_grp["indices_with_ghosts"]),
                coords_with_ghosts=np.array(lsfd_grp["coords_with_ghosts"]),
                distances_with_ghosts=np.array(lsfd_grp["distances_with_ghosts"]),
                lsfd_edge_vectors_with_ghosts=np.array(lsfd_grp["lsfd_edge_vectors_with_ghosts"]),
                ghost_dist=float(lsfd_grp.attrs["ghost_dist"]),
                ghost_coords=np.array(lsfd_grp["ghost_coords"]),
                indices_of_own_ghost_point_in_sites_with_ghosts=np.array(lsfd_grp["indices_of_own_ghost"]),
            )

        return cls(sites=sites, elements=elements, boundary_indices=boundary_indices,
                   tri_mesh=tri_mesh, dual_mesh=dual_mesh, lsfd_neighbors=lsfd_neighbors,
                   n_lsfd_neighbors=K, ghost_coeff=gc)

    # ========================================================================
    # УДОБНЫЕ МЕТОДЫ
    # ========================================================================

    def save(self, filepath: str, compress: bool = True) -> None:
        """Сохранить Mesh в файл."""
        with h5py.File(filepath, "w") as f:
            self.to_hdf5(f, compress=compress)

    @classmethod
    def load(cls, filepath: str) -> "Mesh":
        """Загрузить Mesh из файла."""
        with h5py.File(filepath, "r") as f:
            return cls.from_hdf5(f)

    @staticmethod
    def build_lsfd_neighbors(
            sites: np.ndarray,
            boundary_indices: np.ndarray,
            tri_mesh: "TriMesh",
            n_lsfd_neighbors: int,
            ghost_coeff: float = 1.0,
            use_ghost_points: bool = True,
            verbose: bool = True,  # ← печатать отчёт
            validate: bool = False,
            tol: float = 1e-8,
    ) -> LSFDNeighbors:
        """Построить LSFD-соседей. use_ghost_points=False -> только обычные соседи."""
        sites = np.asarray(sites, dtype=np.float64)

        # --- обычные соседи (всегда) ---
        kdtree = KDTree(sites)
        distances, indices = kdtree.query(sites, k=n_lsfd_neighbors + 1)
        distances, indices = distances[:, 1:], indices[:, 1:]   # убираем саму точку
        coords = sites[indices] - sites[:, np.newaxis, :]

        # --- ghost points ---
        n_dist = ghost_coeff * float(np.mean(tri_mesh.edge_lengths[tri_mesh.boundary_edge_indices]))
        ghost_coords = sites[boundary_indices] + tri_mesh.boundary_site_normals * n_dist
        sites_with_ghosts = np.concatenate([ghost_coords, sites])  # (G+N, 2)

        kdtree_g = KDTree(sites_with_ghosts)
        dwg, iwg = kdtree_g.query(sites_with_ghosts, k=n_lsfd_neighbors + 1)
        dwg = dwg[len(ghost_coords):, 1:]     # только реальные строки, без self
        iwg = iwg[len(ghost_coords):, 1:]
        cwg = sites_with_ghosts[iwg] - sites[:, np.newaxis, :]

        I = iwg[boundary_indices]             # (B, K)
        own_ghost = np.argmax(I == np.arange(len(boundary_indices))[:, None], axis=1)


        nb = LSFDNeighbors(
            indices=indices, coords=coords, distances=distances,
            lsfd_edge_vectors=coords,
            indices_with_ghosts=iwg, coords_with_ghosts=cwg,
            distances_with_ghosts=dwg, lsfd_edge_vectors_with_ghosts=cwg,
            ghost_dist=n_dist, ghost_coords=ghost_coords,
            indices_of_own_ghost_point_in_sites_with_ghosts=own_ghost,
        )

        # --- валидация применяется СРАЗУ к готовому LSFDNeighbors ---
        if validate:
            report = Mesh.validate_lsfd_neighbors(
                nb, sites, boundary_indices, tri_mesh, tol=tol)
            failed = {k: v for k, v in report.items() if v > tol}

            if verbose:
                status = "FAIL ❌" if failed else "OK ✅"
                print(f"\n🔍 LSFD neighbors validation: {status} "
                      f"(K={n_lsfd_neighbors}, tol={tol:.0e})")
                for k, v in report.items():
                    mark = "❌" if v > tol else "  "
                    print(f"  {mark} {k:<26s} {v:.3e}")

            if failed:
                raise ValueError(f"LSFDNeighbors validation FAILED: {failed}")

        return nb

    @staticmethod
    def validate_lsfd_neighbors(
            nb: "LSFDNeighbors",
            sites: np.ndarray,
            boundary_indices: np.ndarray,
            tri_mesh: "TriMesh",
            tol: float = 1e-8,
    ) -> Dict[str, float]:
        """
        Валидация LSFDNeighbors напрямую (без operators).
        Возвращает словарь {проверка: величина ошибки}; 0.0 = пройдено.
        Логические проверки: 0.0 — ок, 1.0 — нарушено.
        """
        sites = np.asarray(sites, dtype=np.float64)
        boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        N, B = len(sites), len(boundary_indices)
        r = {}

        # ---------- обычные соседи ----------
        # coords согласованы с sites
        r['coords_vs_sites'] = float(np.max(np.abs(
            (sites[nb.indices] - sites[:, None, :]) - nb.coords)))
        # distances = |coords|
        r['dist_vs_coords'] = float(np.max(np.abs(
            np.linalg.norm(nb.coords, axis=2) - nb.distances)))
        # точка не сосед сама себе
        r['self_in_neighbors'] = float(np.any(nb.indices == np.arange(N)[:, None]))

        # ---------- ghost-часть (если построена) ----------
        if nb.indices_with_ghosts is not None:
            G = len(nb.ghost_coords)
            swg = np.concatenate([nb.ghost_coords, sites])  # (G+N, 2)

            # индексы в диапазоне [0, G+N), без самой точки (self = G + i)
            r['iwg_range'] = float(
                (nb.indices_with_ghosts.min() < 0) or
                (nb.indices_with_ghosts.max() >= G + N))
            r['self_in_ghost_neighbors'] = float(np.any(
                nb.indices_with_ghosts == (np.arange(N)[:, None] + G)))

            # coords/distances согласованы с sites_with_ghosts
            r['cwg_vs_swg'] = float(np.max(np.abs(
                (swg[nb.indices_with_ghosts] - sites[:, None, :]) - nb.coords_with_ghosts)))
            r['dwg_vs_cwg'] = float(np.max(np.abs(
                np.linalg.norm(nb.coords_with_ghosts, axis=2) - nb.distances_with_ghosts)))

            # ghost стоит на нормали на расстоянии ghost_dist
            ghost_expected = sites[boundary_indices] + tri_mesh.boundary_site_normals * nb.ghost_dist
            r['ghost_position'] = float(np.max(np.abs(ghost_expected - nb.ghost_coords)))
            r['ghost_dist'] = float(np.max(np.abs(
                np.linalg.norm(nb.ghost_coords - sites[boundary_indices], axis=1) - nb.ghost_dist)))

            # у каждой граничной точки ровно один собственный ghost, и own_ghost_col верен
            I = nb.indices_with_ghosts[boundary_indices]                      # (B, K)
            counts = np.sum(I == np.arange(B)[:, None], axis=1)
            r['own_ghost_unique'] = float(np.any(counts != 1))
            r['own_ghost_col'] = float(np.any(
                I[np.arange(B), nb.indices_of_own_ghost_point_in_sites_with_ghosts] != np.arange(B)))

            # вектор в колонке own_ghost = вектор к ghost
            own_vec = nb.coords_with_ghosts[
                boundary_indices, nb.indices_of_own_ghost_point_in_sites_with_ghosts]
            r['own_ghost_vec'] = float(np.max(np.abs(
                own_vec - (nb.ghost_coords - sites[boundary_indices]))))

        return r

    @staticmethod
    def from_triangulation(
            sites: Sequence[Tuple[float, float]],
            elements: Sequence[Tuple[int, int, int]],
            n_lsfd_neighbors: int = 15,  # ← глобальный параметр
            ghost_coeff: float = 1.0,
    ) -> "Mesh":
        """
        Создать Mesh из триангуляции.

        Args:
            sites: Координаты вершин (N, 2).
            elements: Индексы вершин треугольников (M, 3).
            n_lsfd_neighbors: Число соседей для LSFD (сохраняется в сетке).
        """
        sites = np.asarray(sites, dtype=np.float64).squeeze()
        elements = np.asarray(elements, dtype=np.int64).squeeze()

        if sites.ndim != 2 or sites.shape[1] != 2:
            raise ValueError(f"sites must be (N, 2), got {sites.shape}")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(f"elements must be (M, 3), got {elements.shape}")

        # === 1. Граничные вершины ===
        boundary_indices = Mesh._find_boundary_indices(elements)

        # === 2. TriMesh ===
        tri_mesh = TriMesh(sites, elements)

        # === 3. DualMesh ===
        dual_mesh = DualMesh(tri_mesh)

        # === 4. lsfd_neighbors ===
        lsfd_neighbors = Mesh.build_lsfd_neighbors(
            sites, boundary_indices, tri_mesh, n_lsfd_neighbors)

        return Mesh(
            sites=sites,
            elements=elements,
            boundary_indices=boundary_indices,
            tri_mesh=tri_mesh,
            dual_mesh=dual_mesh,
            lsfd_neighbors=lsfd_neighbors,
            n_lsfd_neighbors=n_lsfd_neighbors,  # ← передаём в __init__
        )

    @staticmethod
    def _find_boundary_indices(elements: np.ndarray) -> np.ndarray:
        """Найти индексы вершин на границе."""
        edges, is_boundary = get_edges(elements)
        boundary_edges = edges[is_boundary]
        return np.unique(boundary_edges.flatten())

    def __repr__(self):
        return (f"Mesh(sites={self.n_sites}, triangles={self.n_elements}, "
                f"LSFD_neighbors={self.n_lsfd_neighbors})")