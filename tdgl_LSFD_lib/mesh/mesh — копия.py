from typing import Sequence, Tuple, Union, Optional, List
from dataclasses import dataclass
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from .tri_mesh import TriMesh
from .dual_mesh import DualMesh
from .util import (
    get_edges,
    triangle_areas,
    generate_voronoi_vertices,
    get_voronoi_polygon_indices,
    compute_voronoi_polygon_areas,
    convex_polygon_centroid,
)

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

    def smooth(self, iterations: int = 1, create_submesh: bool = True) -> "Mesh":
        """
        Laplacian smoothing: перемещает внутренние вершины в среднее соседей.

        Args:
            iterations: Число итераций сглаживания.
            create_submesh: Если False, возвращает минимальный Mesh (только sites/elements)
                           для использования внутри make_mesh(). Если True — создаёт полный Mesh.

        Returns:
            Mesh с обновлёнными координатами.
        """
        sites = self.sites.copy()
        elements = self.elements
        edges, _ = get_edges(elements)
        n = len(sites)
        boundary = self.boundary_indices

        # === Сглаживание: только координаты ===
        for _ in range(iterations):
            num_neighbors = np.bincount(edges.ravel(), minlength=n)

            new_sites = np.zeros_like(sites)
            for dim in range(2):
                # Соседи через edges[:, 1] для edges[:, 0]
                vals = sites[edges[:, 1], dim]
                new_sites[:, dim] += np.bincount(edges[:, 0], weights=vals, minlength=n)
                # Соседи через edges[:, 0] для edges[:, 1]
                vals = sites[edges[:, 0], dim]
                new_sites[:, dim] += np.bincount(edges[:, 1], weights=vals, minlength=n)

            new_sites /= num_neighbors[:, np.newaxis]
            new_sites[boundary] = sites[boundary]  # Фиксация границы
            sites = new_sites

        # === Возвращаем результат ===
        if create_submesh:
            # Полный Mesh (для пользователя)
            return Mesh.from_triangulation(sites, elements, n_lsfd_neighbors=self.n_lsfd_neighbors)
        else:
            # Минимальный Mesh (для внутреннего использования в make_mesh)
            return Mesh(
                sites=sites,
                elements=elements,
                boundary_indices=boundary,
                n_lsfd_neighbors=self.n_lsfd_neighbors,
            )

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

    def to_hdf5(self, h5group: h5py.Group, compress: bool = True) -> None:
        """Сохранить Mesh в HDF5."""
        compression = "gzip" if compress else None

        # === Базовые данные ===
        h5group.create_dataset("sites", data=self.sites, compression=compression)
        h5group.create_dataset("elements", data=self.elements, compression=compression)
        h5group.create_dataset("boundary_indices", data=self.boundary_indices, compression=compression)

        # === Глобальный параметр LSFD ===
        h5group.attrs["n_lsfd_neighbors"] = self.n_lsfd_neighbors  # ← СОХРАНЯЕМ

        # === TriMesh (только основные атрибуты) ===
        tri_grp = h5group.create_group("tri_mesh")
        tri_grp.create_dataset("edges", data=self.tri_mesh.edges, compression=compression)
        tri_grp.create_dataset("boundary_edge_indices", data=self.tri_mesh.boundary_edge_indices,
                               compression=compression)
        tri_grp.create_dataset("edge_lengths", data=self.tri_mesh.edge_lengths, compression=compression)
        tri_grp.create_dataset("edge_directions", data=self.tri_mesh.edge_directions, compression=compression)
        tri_grp.create_dataset("edge_midpoints", data=self.tri_mesh.edge_midpoints, compression=compression)
        tri_grp.create_dataset("tri_areas", data=self.tri_mesh.tri_areas, compression=compression)
        tri_grp.create_dataset("tri_centroids", data=self.tri_mesh.tri_centroids, compression=compression)
        tri_grp.create_dataset("tri_to_edges", data=self.tri_mesh.tri_to_edges, compression=compression)
        tri_grp.create_dataset("edges_to_tri", data=self.tri_mesh.edges_to_tri, compression=compression)
        tri_grp.create_dataset("tri_edge_normals", data=self.tri_mesh.tri_edge_normals, compression=compression)
        tri_grp.create_dataset("boundary_site_normals", data=self.tri_mesh.boundary_site_normals,
                               compression=compression)
        tri_grp.create_dataset("boundary_site_indices", data=self.tri_mesh.boundary_site_indices,
                               compression=compression)

        # === DualMesh ===
        dual_grp = h5group.create_group("dual_mesh")
        dual_grp.create_dataset("dual_sites", data=self.dual_mesh.dual_sites, compression=compression)
        dual_grp.create_dataset("dual_areas", data=self.dual_mesh.dual_areas, compression=compression)
        dual_grp.create_dataset("dual_edge_lengths", data=self.dual_mesh.dual_edge_lengths, compression=compression)
        dual_grp.create_dataset("dual_edge_directions", data=self.dual_mesh.dual_edge_directions,
                                compression=compression)

        # Полигоны Вороного (ragged array)
        if self.dual_mesh.voronoi_polygons:
            polygons_flat = np.concatenate(self.dual_mesh.voronoi_polygons, axis=0)
            split_indices = np.cumsum([len(p) for p in self.dual_mesh.voronoi_polygons[:-1]])
            dual_grp.create_dataset("voronoi_polygons_flat", data=polygons_flat, compression=compression)
            dual_grp.create_dataset("voronoi_split_indices", data=split_indices, compression=compression)

        # === LSFD Neighbors ===
        lsfd_grp = h5group.create_group("lsfd_neighbors")
        lsfd_grp.create_dataset("indices", data=self.lsfd_neighbors.indices, compression=compression)
        lsfd_grp.create_dataset("coords", data=self.lsfd_neighbors.coords, compression=compression)
        lsfd_grp.create_dataset("distances", data=self.lsfd_neighbors.distances, compression=compression)

        # === Метаданные ===
        h5group.attrs["n_sites"] = self.n_sites
        h5group.attrs["n_elements"] = self.n_elements

    @classmethod
    def from_hdf5(cls, h5group: h5py.Group) -> "Mesh":
        """Загрузить Mesh из HDF5."""
        # Проверка обязательных данных
        for name in ["sites", "elements", "boundary_indices"]:
            if name not in h5group:
                raise IOError(f"Missing required dataset: {name}")

        sites = np.array(h5group["sites"])
        elements = np.array(h5group["elements"], dtype=np.int64)
        boundary_indices = np.array(h5group["boundary_indices"], dtype=np.int64)

        # === Загружаем глобальный параметр ===
        n_lsfd_neighbors = int(h5group.attrs.get("n_lsfd_neighbors", 15))  # ← ЗАГРУЖАЕМ

        # === TriMesh ===
        tri_grp = h5group["tri_mesh"]
        tri_mesh = TriMesh.__new__(TriMesh)
        tri_mesh.sites = sites
        tri_mesh.triangles = elements
        tri_mesh.edges = np.array(tri_grp["edges"])
        tri_mesh.boundary_edge_indices = np.array(tri_grp["boundary_edge_indices"], dtype=np.int64)
        tri_mesh.edge_lengths = np.array(tri_grp["edge_lengths"])
        tri_mesh.edge_directions = np.array(tri_grp["edge_directions"])
        tri_mesh.edge_midpoints = np.array(tri_grp["edge_midpoints"])
        tri_mesh.tri_areas = np.array(tri_grp["tri_areas"])
        tri_mesh.tri_centroids = np.array(tri_grp["tri_centroids"])
        tri_mesh.tri_to_edges = np.array(tri_grp["tri_to_edges"], dtype=np.int64)
        tri_mesh.edges_to_tri = np.array(tri_grp["edges_to_tri"], dtype=np.int64)
        tri_mesh.tri_edge_normals = np.array(tri_grp["tri_edge_normals"])
        tri_mesh.boundary_site_normals = np.array(tri_grp["boundary_site_normals"])
        tri_mesh.boundary_site_indices = np.array(tri_grp["boundary_site_indices"])  # ← ДОБАВИТЬ!
        # Пересчитываем normalized_edge_directions
        tri_mesh.normalized_edge_directions = tri_mesh.edge_directions / tri_mesh.edge_lengths[:, np.newaxis]

        # === DualMesh ===
        dual_grp = h5group["dual_mesh"]
        dual_mesh = DualMesh.__new__(DualMesh)
        dual_mesh.dual_sites = np.array(dual_grp["dual_sites"])
        dual_mesh.dual_areas = np.array(dual_grp["dual_areas"])
        dual_mesh.dual_edge_lengths = np.array(dual_grp["dual_edge_lengths"])
        dual_mesh.dual_edge_directions = np.array(dual_grp["dual_edge_directions"])
        dual_mesh.tri_mesh = tri_mesh

        # Загружаем полигоны Вороного
        if "voronoi_polygons_flat" in dual_grp:
            polygons_flat = np.array(dual_grp["voronoi_polygons_flat"])
            split_indices = np.array(dual_grp["voronoi_split_indices"])
            dual_mesh.voronoi_polygons = np.split(polygons_flat, split_indices)
        else:
            dual_mesh.voronoi_polygons = []

        # === LSFD Neighbors ===
        lsfd_grp = h5group["lsfd_neighbors"]
        lsfd_neighbors = LSFDNeighbors(
            indices=np.array(lsfd_grp["indices"]),
            coords=np.array(lsfd_grp["coords"]),
            distances=np.array(lsfd_grp["distances"]),
        )

        return cls(
            sites=sites,
            elements=elements,
            boundary_indices=boundary_indices,
            tri_mesh=tri_mesh,
            dual_mesh=dual_mesh,
            lsfd_neighbors=lsfd_neighbors,
            n_lsfd_neighbors=n_lsfd_neighbors,  # ← передаём в __init__
        )

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

        # === 4. LSFD Neighbors ===
        kdtree = KDTree(sites)
        distances, indices = kdtree.query(sites, k=n_lsfd_neighbors + 1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        coords = sites[indices] - sites[:, np.newaxis, :]

        lsfd_edge_vectors = coords   # (N, K, 2)

        # ==== 5. LSFD Neighbors with ghost points ===

        n_dist = ghost_coeff #np.mean(tri_mesh.edge_lengths[tri_mesh.boundary_edge_indices])

        ghost_coords = sites[boundary_indices] + tri_mesh.boundary_site_normals * n_dist

        sites_with_ghosts = np.concatenate([ghost_coords, sites]) # (N+G, 2)
        kdtree_with_ghosts = KDTree(sites_with_ghosts)
        distances_with_ghosts, indices_with_ghosts = kdtree_with_ghosts.query(sites_with_ghosts, k=n_lsfd_neighbors + 1) # (N+G, K+1)

        distances_with_ghosts = distances_with_ghosts[len(ghost_coords):] # (N, K+1)
        distances_with_ghosts = distances_with_ghosts[:, 1:] # (N, K)

        indices_with_ghosts = indices_with_ghosts[len(ghost_coords):] # (N, K+1)
        indices_with_ghosts = indices_with_ghosts[:, 1:] # (N, K)

        coords_with_ghosts = sites_with_ghosts[indices_with_ghosts] - sites[:, np.newaxis, :]

        lsfd_edge_vectors_with_ghosts = coords_with_ghosts  # (N, K, 2)

        I = indices_with_ghosts[boundary_indices]  # (N_bnd, K)

        # Векторизованный поиск: для каждой строки j ищем столбец, где значение == j
        indices_of_own_ghost_point_in_sites_with_ghosts = np.argmax(I == np.arange(len(boundary_indices))[:, None], axis=1)


        lsfd_neighbors = LSFDNeighbors(
            indices=indices,
            coords=coords,
            distances=distances,
            lsfd_edge_vectors = lsfd_edge_vectors,
            indices_with_ghosts = indices_with_ghosts,
            coords_with_ghosts=coords_with_ghosts,
            distances_with_ghosts=distances_with_ghosts,
            lsfd_edge_vectors_with_ghosts = lsfd_edge_vectors_with_ghosts,
            ghost_dist = n_dist,
            ghost_coords = ghost_coords,
            indices_of_own_ghost_point_in_sites_with_ghosts = indices_of_own_ghost_point_in_sites_with_ghosts,
        )


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