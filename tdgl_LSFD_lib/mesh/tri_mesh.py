import numpy as np
from typing import Sequence, Tuple

from .util import  get_edge_lengths, get_edges, build_edge_triangle_mapping, triangle_areas


class TriMesh:
    """Parameters of Delaunay Triangulation mesh.

    Contains geometric data for each triangle: areas, centroids,
    edge lengths, normals, and boundary information.
    """

    def __init__(
        self,
        sites: Sequence[Tuple[float, float]], # coordinates of vertices
        triangles: Sequence[Tuple[int, int, int]], # triangles of Delaunay triangulation as triples of vertex indices
    ):
        self.sites = np.asarray(sites).squeeze()
        self.triangles = np.asarray(triangles, dtype=np.int64)

        # === 1. Edges ===
        self.edges, is_boundary = get_edges(self.triangles) # shape (E, 2) array of site indices for each edge
        self.boundary_edge_indices = np.where(is_boundary)[0] # boolean array indicating which edges are on the boundary
        self.edge_lengths = get_edge_lengths(self.sites, self.triangles) # shape (E,) array of lengths for each edge
        self.edge_directions = np.diff(self.sites[self.edges], axis=1).squeeze()  # shape (E, 2) array of vectors pointing from the first site to the second site of each edge
        self.normalized_edge_directions = self.edge_directions / self.edge_lengths[:, np.newaxis] # shape (E, 2) array of unit vectors in the direction of each edge
        self.edge_midpoints = self.sites[self.edges].mean(axis=1)  # shape (E, 2) array of midpoints for each edge

        # === 2. Triangles ===
        self.tri_areas = triangle_areas(self.sites, self.triangles) # shape (M,) array of areas for each triangle
        self.tri_centroids = self.sites[self.triangles].mean(axis=1) # shape (M, 2) array of centroids for each triangle

        #= 3. Edge-Triangle Mapping ===
        self.tri_to_edges, self.edges_to_tri = build_edge_triangle_mapping(self.triangles, self.edges)
        # tri_to_edges: shape (M, 3) array of edge indices for each triangle; edges_to_tri: shape (E, 2) array of triangle indices for each edge (with -1 for boundary edges)

        # === 4. Normals ===
        self.tri_edge_normals = self.compute_tri_edge_normals() # shape (M, 3, 2) array of normals for each edge of each triangle, oriented outward from the triangle's centroid
        self.boundary_site_indices, self.boundary_site_normals, self.boundary_edge_normals = self.compute_boundary_normals() # shape (B, 2) array of normals for each boundary vertex

        return

    def compute_boundary_normals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normal for a boundary edge is the normal of the triangle it belongs to, oriented outward.
        Normal for boundary site is normalized sum of normals of the normals to boundary edges that meet at that site.
        """

        boundary_edge_normals = np.zeros( (len(self.boundary_edge_indices), 2))

        for i, edge_idx in enumerate(self.boundary_edge_indices):
            tri_idx = self.edges_to_tri[edge_idx, 0]
            #Search for local index of edge in triangle
            local_idx = np.where(self.tri_to_edges[tri_idx] == edge_idx)[0][0]
            boundary_edge_normals[i] = self.tri_edge_normals[tri_idx, local_idx]

        boundary_edges = self.edges[self.boundary_edge_indices]
        boundary_vertices = np.unique(boundary_edges.flatten())

        # === ИСПРАВЛЕНИЕ: создаём массив на ВСЕ вершины ===
        site_normals = np.zeros((len(self.sites), 2), dtype=np.float64)

        for i, (s1, s2) in enumerate(boundary_edges):
            site_normals[s1] += boundary_edge_normals[i]
            site_normals[s2] += boundary_edge_normals[i]

        # === Берём только граничные вершины ===
        boundary_site_normals = site_normals[boundary_vertices]

        # === Нормализуем (с защитой от 0) ===
        norms = np.linalg.norm(boundary_site_normals, axis=1)
        boundary_site_normals = boundary_site_normals / norms[:, np.newaxis]

        norms = np.linalg.norm(boundary_edge_normals, axis=1)
        boundary_edge_normals = boundary_edge_normals / norms[:, np.newaxis]

        return boundary_vertices, boundary_site_normals, boundary_edge_normals

    def compute_tri_edge_normals(self) -> np.ndarray:
        """
        Calculate external normals to edges for each triangle.
        One edge has two normals (one for each adjacent triangle), but we want to assign a unique normal to each edge in the context of a triangle.
        We determine the direction of the normal by checking if it points away from the triangle's centroid.
        If the normal points towards the centroid, we flip it to ensure it points outward.

        Returns:
            tri_edge_normals: A shape (M, 3, 2) array where M is the number of triangles. Each entry
            tri_edge_normals[i, j] = the normal vector for the j-th edge of the i-th triangle, oriented to point outward from the triangle's centroid.
        """

        normal = np.stack([self.normalized_edge_directions[:, 1], -self.normalized_edge_directions[:, 0]], axis=1)
        tri_normal = normal[self.tri_to_edges]  # (M, 3, 2)
        to_centroid = self.edge_midpoints[self.tri_to_edges] - self.tri_centroids[:, np.newaxis, :]
        dot = np.sum(tri_normal * to_centroid, axis=2)  # (M, 3)

        # If dot > 0 → normal points outward (away from the centroid)
        mask = dot > 0  # True → tri_normal
        tri_edge_normals = np.where(mask[:, :, np.newaxis], tri_normal, -tri_normal)  # (M, 3, 2)
        return tri_edge_normals
