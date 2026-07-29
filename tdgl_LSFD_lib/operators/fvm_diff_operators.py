import warnings
from typing import Callable, Tuple, Union
import time
import numpy as np
import scipy.sparse as sp

from tdgl_LSFD_lib.mesh.mesh import Mesh

# ----------------------------------------------------------------
# 1. Finite Volume operators for numerical integration and flux calculation
# ----------------------------------------------------------------
class FVM_diff:

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

    def build_laplacian(self, A_on_sites: np.array = None) -> sp.csc_array:
        """Build a Laplacian matrix on a given mesh.

        The default boundary condition is homogenous Neumann conditions. To get
        Dirichlet conditions, add fixed sites. To get non-homogenous Neumann condition,
        the flux needs to be specified using a Neumann boundary Laplacian matrix.

        Args:
            mesh: The mesh.
            link_exponents: The value is integrated, exponentiated and used as a
                link variable.
            fixed_sites: The sites to hold fixed.
            fixed_sites_eigenvalues: The eigenvalues for the fixed sites.

        Returns:
            The Laplacian matrix and indices of non-fixed rows.
        """

        weights = self.dual_edge_lengths / self.edge_lengths

        if A_on_sites is None:
            link_variable_weights = np.ones(len(weights))
        else:
            link_variable_weights = self.set_link_exponents_on_edge(A_on_sites)

        edges0 = self.edges[:, 0]
        edges1 = self.edges[:, 1]
        rows = np.concatenate([edges0, edges1, edges0, edges1])
        cols = np.concatenate([edges1, edges0, edges0, edges1])
        areas0 = self.voronoi_areas[edges0]
        areas1 = self.voronoi_areas[edges1]
        values = np.concatenate(
            [
                weights * link_variable_weights / areas0,
                weights * link_variable_weights.conjugate() / areas1,
                -weights / areas0,
                -weights / areas1,
            ]
        )

        laplacian = sp.csc_array(
            (values, (rows, cols)), shape=(len(self.sites), len(self.sites))
        )
        return laplacian

    def set_link_exponents_on_edge(self, A_on_sites:  np.array):

        edges = self.mesh.tri_mesh.edges
        A_on_edges = 0.5 * (A_on_sites[edges[:, 0]] + A_on_sites[edges[:, 1]])
        A_dot_e = np.einsum("ij, ij -> i", A_on_edges, self.edge_directions)
        U_link = np.cos(A_dot_e) - 1j * np.sin(A_dot_e)
        return U_link