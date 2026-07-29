import logging
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, Delaunay, QhullError
from shapely.geometry import MultiLineString
from shapely.ops import orient, polygonize
from tqdm import tqdm

logger = logging.getLogger("tdgl.mesh")


def get_edges(elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the edges from a list of triangle indices.

    Args:
        elements: The triangle indices, shape ``(n, 3)``

    Returns:
        A tuple containing an integer array of edges and a boolean array
        indicating whether each edge on in the boundary.
    """
    edges = np.concatenate([elements[:, e] for e in [(0, 1), (1, 2), (2, 0)]])
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    return edges, counts == 1


def get_edge_lengths(points: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Returns the lengths of all edges in a triangulation.

    Args:
        points: Vertex coordinates
        elements: Triangle indices

    Returns:
        An array of edge lengths
    """
    edges, _ = get_edges(elements)
    return np.linalg.norm(np.diff(points[edges], axis=1), axis=2).squeeze()

def build_edge_triangle_mapping(triangles: np.ndarray, edges: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Построить сопоставление между треугольниками и рёбрами.

        Args:
            triangles: Индексы вершин треугольников, форма (M, 3).
            edges: Уникальные рёбра, форма (E, 2).

        Returns:
            tri_to_edges: Массив индексов рёбер для каждого треугольника, форма (M, 3).
            edges_to_tri: Массив индексов треугольников для каждого ребра, форма (E, 2). (каждое ребро входит в два
            или один треугольник - граничное ребро)
        """

        M = len(triangles)
        E = len(edges)

        # Словарь: (v1, v2) → edge_index (edges уже отсортированы!)
        edge_dict = {tuple(edge): idx for idx, edge in enumerate(edges)}

        # Массив результатов
        tri_to_edges = np.zeros((M, 3), dtype=np.int64)
        edges_to_tri = -1 * np.ones((E, 2), dtype=np.int64)

        for tri_idx, (v0, v1, v2) in enumerate(triangles):
            # Явно создаём кортежи из 2 элементов (PyCharm доволен)
            e0 = (v0, v1) if v0 < v1 else (v1, v0)
            e1 = (v1, v2) if v1 < v2 else (v2, v1)
            e2 = (v2, v0) if v2 < v0 else (v0, v2)

            tri_to_edges[tri_idx, 0] = edge_dict[e0]
            tri_to_edges[tri_idx, 1] = edge_dict[e1]
            tri_to_edges[tri_idx, 2] = edge_dict[e2]

            if edges_to_tri[edge_dict[e0]][0] < 0:
                edges_to_tri[edge_dict[e0]][0] = tri_idx
            else:
                edges_to_tri[edge_dict[e0]][1] = tri_idx

            if edges_to_tri[edge_dict[e1]][0] < 0:
                edges_to_tri[edge_dict[e1]][0] = tri_idx
            else:
                edges_to_tri[edge_dict[e1]][1] = tri_idx

            if edges_to_tri[edge_dict[e2]][0] < 0:
                edges_to_tri[edge_dict[e2]][0] = tri_idx
            else:
                edges_to_tri[edge_dict[e2]][1] = tri_idx

        return tri_to_edges, edges_to_tri



def get_max_edge_length(points: np.ndarray, elements: np.ndarray) -> float:
    """Returns the maximum edge length in a triangulation.

    Args:
        points: Vertex coordinates
        elements: Triangle indices

    Returns:
        The maximum edge length
    """
    edges = np.concatenate([elements[:, e] for e in [(0, 1), (1, 2), (2, 0)]])
    return np.linalg.norm(np.diff(points[edges], axis=1), axis=2).max()


def get_dual_edge_lengths(
    edge_centers: np.ndarray,
    elements: np.ndarray,
    dual_sites: np.ndarray,
    edges: np.ndarray,
    num_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the lengths of the dual edges.

    Args:
        edge_centers: The (x, y) coordinates of the edge centers.
        elements: The triangular elements in the tesselation.
        dual_sites: The (x, y) coordinates for the dual mesh (Voronoi sites).
        edges: The edges connecting the sites.
        num_sites: The number of sites in the mesh.

    Returns:
        An array of dual edge lengths.
    """
    # Create a dict with keys corresponding to the edges and values
    # corresponding to the triangle indices
    adj = make_adj_directed_tri_indices(elements, num_sites)
    edge_to_element = defaultdict(list)
    for i, j, v in zip(*sp.find(adj)):
        # The triangle index is the entry in the adjacency matrix minus 1
        edge_to_element[frozenset((i, j))].append(v - 1)
    edge_to_element = dict(edge_to_element)

    dual_lengths = np.zeros(len(edge_centers), dtype=float)
    dual_edge_directions = np.zeros((len(edge_centers),2), dtype=float)

    for i, edge in enumerate(edges):
        indices = edge_to_element[frozenset(edge)]
        if len(indices) == 1:  # Boundary edge
            vec = dual_sites[indices[0]] - edge_centers[i]
            dual_lengths[i] = np.linalg.norm(vec)
            dual_edge_directions[i] = vec / np.linalg.norm(vec)

        else:  # Inner edge
            vec = dual_sites[indices[0]] - dual_sites[indices[1]]
            dual_lengths[i] = np.linalg.norm(vec)
            dual_edge_directions[i] = vec / np.linalg.norm(vec)

    return dual_lengths, dual_edge_directions

def generate_voronoi_vertices(
    sites: np.ndarray, elements: np.ndarray
) -> np.ndarray:
    """Compute the vertices of the Voronoi lattice by computing the
    circumcenters of the triangles in the tesselation.

    Args:
        sites: The (x, y) coordinates of the tesselation.
        elements: The triangular elements in the tesselation.

    Returns:
        The x and y coordinates of the Voronoi vertices, as arrays.
    """
    # https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2
    # Get the triangle ABC
    # Convert to the coordinate system where A is in the origin
    A = sites[elements[:, 0]]
    B = sites[elements[:, 1]] - A
    C = sites[elements[:, 2]] - A
    # Compute the circumcenter
    D = 2 * B[:, 0] * C[:, 1] - 2 * B[:, 1] * C[:, 0]
    Ux = (C[:, 1] * (B**2).sum(axis=1) - B[:, 1] * (C**2).sum(axis=1)) / D
    Uy = (B[:, 0] * (C**2).sum(axis=1) - C[:, 0] * (B**2).sum(axis=1)) / D
    # Convert back to the initial coordinate system
    return np.array([Ux, Uy]).T + A


def make_adj_directed_tri_indices(elements: np.ndarray, num_sites: int) -> sp.csc_array:
    """Construct the directed adjacency matrix.

    Each element (i, j) represents an edge in the mesh, and the value at (i, j)
    is 1 + the index of a triangle containing that edge.

    Args:
        elements: The triangle indices, shape ``(m, 3)``
        num_sites: The number of sites in the mesh

    Returns:
        A directed adjacency matrix containing triangle indices + 1
    """
    t0 = elements[:, 0]
    t1 = elements[:, 1]
    t2 = elements[:, 2]
    i = np.column_stack([t0, t1, t2]).ravel()
    j = np.column_stack([t1, t2, t0]).ravel()
    # store triangle index + 1 (zero means no edge connecting i and j)
    data = np.repeat(np.arange(1, elements.shape[0] + 1), 3)
    return sp.csc_array((data, (i, j)), shape=(num_sites, num_sites))


def get_voronoi_polygon_indices(
    elements: np.ndarray, num_sites: int
) -> List[np.ndarray]:
    """Find the polygons surrounding each site.

    The indices of the Voronoi vertices surrounding each site are the
    same as the indices of the triangles adjacent to each site.

    Args:
        elements: The triangular elements in the tesselation.
        num_sites: The number of sites

    Returns:
        A list of arrays of Voronoi polygon indices.
    """
    adj = make_adj_directed_tri_indices(elements, num_sites).tolil()
    return [np.array(tri) - 1 for tri in adj.data]


def compute_voronoi_polygon_areas(
    sites: np.ndarray,
    dual_sites: np.ndarray,
    boundary: np.ndarray,
    edges: np.ndarray,
    boundary_edge_indices: np.ndarray,
    polygons: List[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute the areas of the surrounding polygons.

    Areas of boundary points are handled by adding additional points
    on the boundary to make a convex polygon.

    Args:
        sites: The (x, y) coordinates for the sites.
        dual_sites: The (x, y) coordinates of the dual (Voronoi) sites.
        boundary: An array containing all boundary points.
        edges: The edges of the triangles.
        boundary_edge_indices: The edge indices corresponding to the boundary.
        polygons: The polygons in Voronoi diagram.

    Returns:
        An array of areas for each site in the lattice, and a list of
        counterclockwise-oriented Voronoi polygon vertices.
    """

    boundary_set = set(boundary)
    boundary_edges = edges[boundary_edge_indices]
    areas = np.zeros(len(polygons), dtype=float)
    voronoi_sites = []
    warning_str = (
        "Malformed Voronoi cell surrounding boundary site {site}."
        " Try changing the number of boundary mesh sites using"
        " Polygon.resample() or Polygon.buffer(eps) where eps"
        " is 0 or a small positive float."
    )
    for site, polygon in enumerate(
        tqdm(polygons, desc="Constructing Voronoi polygons")
    ):
        # Get the polygon points
        poly = dual_sites[polygon]
        # # Polygon vertices may end up very close (e.g. delta = 2e-17) due to floating
        # # point errors, so we need to remove near-duplicate vertices.
        # _, unique = np.unique(poly.round(decimals=13), axis=0, return_index=True)
        # poly = poly[unique]
        if site not in boundary_set:
            areas[site], is_convex = get_convex_polygon_area(poly)
            if not is_convex:
                # All interior Voronoi cells must be convex.
                raise ValueError(warning_str.format(site=site))
            voronoi_sites.append(orient_convex_polygon(poly))
            continue
        # For points on the boundary, add vertices at the mesh site and the midpoints
        # of the two edges adjacent to the mesh site to complete the Voronoi polygon.
        connected_boundary_edges = boundary_edges[(boundary_edges == site).any(axis=1)]
        # The midpoints of the two edges adjacent to the site
        midpoints = sites[connected_boundary_edges].mean(axis=1)
        # Orient the convex hull of the polygon in a counterclockwise fashion
        coords = orient_convex_polygon(np.concatenate([poly, midpoints], axis=0))
        coords = [tuple(xy) for xy in coords]
        # Insert the central mesh site between the two boundary edge midpoints
        # to ensure the correct ordering of coordinates.
        indices = sorted([coords.index(tuple(mid)) for mid in midpoints])
        if indices[1] == indices[0] + 1:
            # The two boundary edge midpoints are adjacent in the list of coordinates,
            # so insert the mesh site between them.
            coords.insert(indices[1], sites[site])
        else:
            # The boundary edge midpoints are the first and last elements in the
            # list of coordinates, so append the central mesh site to the end.
            if indices[0] != 0:
                # TODO: Decide whether this should be an exception.
                logger.warning(warning_str.format(site=site))
            coords.append(sites[site])
        poly = np.array(coords)
        areas[site], is_convex = get_convex_polygon_area(poly)
        if not is_convex:
            # If the polygon is non-convex we need to subtract the area of the
            # concave part, which is the triangle formed by the mesh site and
            # the two adjacent boundary edge midpoints.
            triangle_area, is_convex = get_convex_polygon_area(
                np.concatenate([midpoints, [sites[site]]], axis=0)
            )
            assert is_convex  # This is just a triangle, so it must be convex.
            areas[site] -= triangle_area
        voronoi_sites.append(poly)
    return areas, voronoi_sites


def get_convex_polygon_area(coords: np.ndarray) -> Tuple[float, bool]:
    """Compute the area of a convex polygon or the area of its convex hull.

    Note: The vertices do not need to be stored in any specific order.

    Args:
        coords: The (x, y) coordinates of the vertices.

    Returns:
        The area of the polygon or the convex hull, and a bool indicating
        whether the polygon is convex.
    """
    try:
        hull = ConvexHull(coords)
    except QhullError:
        # Handle error when all points lie on a line
        return 0, True
    else:
        is_convex = len(hull.vertices) == len(coords)
        return hull.volume, is_convex


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculates the area of each triangle.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    xy = points[triangles]
    # s1 = xy[:, 2, :] - xy[:, 1, :]
    # s2 = xy[:, 0, :] - xy[:, 2, :]
    # s3 = xy[:, 1, :] - xy[:, 0, :]
    # which can be simplified to
    # s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]  # 3D
    s = xy[:, [2, 0]] - xy[:, [1, 2]]  # 2D
    a = np.linalg.det(s)
    return a * 0.5


def orient_convex_polygon(vertices: np.ndarray) -> np.ndarray:
    """Returns counterclockwise-oriented vertices for a convex polygon.

    Args:
        vertices: The vertex positions (x, y), shape ``(n, 2)``.

    Returns:
        The ``vertices`` sorted counterclockwise.
    """
    # Sort the vertices by the angle between each vertex and some point in the
    # interior of the polygon. Here we use the mean of the vertex positions.
    diffs = vertices - vertices.mean(axis=0)
    return vertices[np.argsort(np.arctan2(diffs[:, 1], diffs[:, 0]))]


def convex_polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Calculates the ``(x, y)`` position of the centroid of a convex polygon.

    Args:
        points: An array of vertex coordinates.

    Returns:
        The ``(x, y)`` position of the centroid of the polygon defined by ``points``.
    """
    # Find a Delaunay triangulation of the polygon
    triangles = Delaunay(points).simplices
    # Find the area and centroid of each triangle
    areas = triangle_areas(points, triangles)
    centroids = points[triangles].mean(axis=1)
    # Return the weighted average of the triangle centroids.
    return np.average(centroids, weights=areas, axis=0)


def get_oriented_boundary(
    points: np.ndarray, boundary_edges: np.ndarray
) -> List[np.ndarray]:
    """Returns arrays of boundary vertex indices, ordered counterclockwise.

    Args:
        points: Shape ``(n, 2)``, float array of vertex coordinates.
        boundary_edges: Shape ``(m, 2)`` integer array of boundary edges.

    Returns:
        A list of arrays of boundary vertex indices (ordered counterclockwise).
        The length of the list will be 1 plus the number of holes in the polygon,
        as each hole has a boundary.
    """
    points_list = [tuple(xy) for xy in points]
    edges = MultiLineString([points[edge, :] for edge in boundary_edges])
    polygons = list(polygonize(edges))
    polygon_indices = []
    for p in polygons:
        polygon = orient(p)
        indices = np.array([points_list.index(xy) for xy in polygon.exterior.coords])
        polygon_indices.append(indices[:-1])
    return polygon_indices

def check_boundary_connectivity(
        edge_indices: np.ndarray,  # Индексы выбранных рёбер (в терминале)
        edges: np.ndarray,  # Все рёбра сетки (E, 2)
) -> bool:
    """
    Проверить, что выбранные граничные рёбра образуют единую связную линию.

    Args:
        edge_indices: Индексы выбранных граничных рёбер (в терминале).
        edges: Все рёбра сетки (E, 2).

    Returns:
        True если рёбра образуют единую связную линию.
    """
    from collections import defaultdict, deque

    # === Случай 0 или 1 ребра ===
    if len(edge_indices) == 0:
        return False  # Нет рёбер → не связно
    if len(edge_indices) == 1:
        return True  # Одно ребро всегда связно

    # === Строим граф смежности для выбранных рёбер ===
    adj = defaultdict(list)
    for edge_idx in edge_indices:
        v1, v2 = edges[edge_idx]
        adj[v1].append(v2)
        adj[v2].append(v1)

    # === BFS из первой вершины первого ребра ===
    start = edges[edge_indices[0]][0]
    visited = set()
    queue = deque([start])

    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for neighbor in adj[v]:
            if neighbor not in visited:
                queue.append(neighbor)

    # === Проверяем, что все вершины выбранных рёбер посещены ===
    all_vertices = set()
    for edge_idx in edge_indices:
        all_vertices.update(edges[edge_idx])

    return visited == all_vertices
