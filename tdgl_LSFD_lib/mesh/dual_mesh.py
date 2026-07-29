from typing import Sequence, Tuple
import numpy as np

from .util import (
    generate_voronoi_vertices,
    get_dual_edge_lengths,
    get_voronoi_polygon_indices,
    compute_voronoi_polygon_areas,
)

class DualMesh:
    """Dual (Voronoi) mesh for finite volume methods.

    Created from a TriMesh instance. Contains:
    - dual_sites: Voronoi vertices (circumcenters of triangles)
    - dual_areas: Area of each Voronoi cell
    - dual_edge_lengths: Length of dual edges (between Voronoi vertices)
    - dual_edge_directions: Direction of dual edges
    - voronoi_polygons: Vertices of each Voronoi cell (for visualization)
    """

    def __init__(self, tri_mesh):
        """
        Create DualMesh from TriMesh.

        Args:
            tri_mesh: TriMesh instance with all geometric data.
        """
        # === From TriMesh ===
        sites = tri_mesh.sites
        triangles = tri_mesh.triangles
        edges = tri_mesh.edges
        boundary_indices = tri_mesh.boundary_site_indices
        boundary_edge_indices = tri_mesh.boundary_edge_indices
        edge_midpoints = tri_mesh.edge_midpoints
        edges_to_tri = tri_mesh.edges_to_tri  # можно использовать вместо adj матрицы

        # === 1. Dual sites (центры описанных окружностей = вершины Вороного) ===
        self.dual_sites = generate_voronoi_vertices(sites, triangles)  # (M, 2)

        # === 2. Индексы полигонов Вороного для каждой вершины ===
        voronoi_polygon_indices = get_voronoi_polygon_indices(triangles, len(sites))

        # === 3. Площади и полигоны Вороного ===
        dual_areas, voronoi_polygons = compute_voronoi_polygon_areas(
            sites=sites,
            dual_sites=self.dual_sites,
            boundary=boundary_indices,
            edges=edges,
            boundary_edge_indices=boundary_edge_indices,
            polygons=voronoi_polygon_indices,
        )
        self.dual_areas = dual_areas  # (N,)
        self.voronoi_polygons = voronoi_polygons  # List[np.ndarray]

        # === 4. Длины и направления двойственных рёбер ===
        dual_edge_lengths, dual_edge_directions = get_dual_edge_lengths(
            edge_centers=edge_midpoints,
            elements=triangles,
            dual_sites=self.dual_sites,
            edges=edges,
            num_sites=len(sites),
        )
        self.dual_edge_lengths = dual_edge_lengths  # (E,)
        self.dual_edge_directions = dual_edge_directions  # (E, 2)

        # === Ссылка на исходный TriMesh (опционально) ===
        self.tri_mesh = tri_mesh