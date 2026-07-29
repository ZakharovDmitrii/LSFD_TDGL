from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union, Literal
from collections import defaultdict, deque

import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from pathlib import Path
from matplotlib.path import Path as MplPath
from matplotlib.tri import Triangulation
from shapely import affinity
from shapely.geometry import Point

from .polygon import Polygon
from ..mesh.mesh import Mesh
from .meshing import generate_mesh
from ..mesh.util import check_boundary_connectivity  # ← Импортируем проверку связности

logger = logging.getLogger(__name__)


@dataclass
class TerminalData:
    """Данные токосъемного контакта."""
    name: str
    site_indices: np.ndarray  # (S,) — индексы граничных вершин в терминале
    edge_indices: np.ndarray  # (E,) — индексы граничных рёбер в терминале
    length: float  # Длина контакта [ξ] (сумма длин рёбер)
    center: np.ndarray  # (2,) — центр терминала
    terminal_type: Literal["source", "drain"]  # ← Тип терминала


class Device:
    """
    Безразмерное устройство для TDGL/LSFD расчётов.

    Все длины в единицах ξ (coherence length = 1).
    Содержит геометрию, сетку, терминалы и методы визуализации.
    """

    def __init__(
            self,
            name: str,
            film: Polygon,
            holes: List[Polygon] = None,
            terminals: List[Polygon] = None,
            probe_points: np.ndarray = None,
    ):
        self.name = name
        self.film = film
        self.holes = holes or []
        self.terminals = terminals or []

        # Проверка уникальности имен
        terminal_names = [t.name for t in self.terminals if t.name]
        if len(terminal_names) != len(set(terminal_names)):
            raise ValueError("Все терминалы должны иметь уникальные имена")

        # Проверка probe_points
        if probe_points is not None:
            probe_points = np.asarray(probe_points).squeeze()
            if probe_points.ndim != 2 or probe_points.shape[1] != 2:
                raise ValueError(f"probe_points должен иметь форму (N, 2), получил {probe_points.shape}")
            if not self.contains_points(probe_points).all():
                raise ValueError("Все probe_points должны лежать внутри пленки")
        self.probe_points = probe_points

        # Сетка и данные (создаются позже)
        self.mesh: Optional[Mesh] = None
        self._triangulation: Optional[Triangulation] = None
        self._terminal_data: Dict[str, TerminalData] = {}  # ← ОБЯЗАТЕЛЬНО объявите здесь!

    # ========================================================================
    # ГЕНЕРАЦИЯ СЕТКИ
    # ========================================================================

    def make_mesh(
            self,
            max_edge_length: float = 0.5,
            n_lsfd_neighbors: int = 15,
            smooth: int = 0,
            ghost_coeff: float = 1,
            **meshpy_kwargs,
    ) -> None:
        """Сгенерировать сетку и найти терминалы."""
        logger.info("Generating mesh...")

        # 1. Генерируем триангуляцию
        points, triangles = generate_mesh(
            self.film.points,
            hole_coords=[h.points for h in self.holes],
            max_edge_length=max_edge_length,
            **meshpy_kwargs,
        )

        # 2. Сглаживание (ЕСЛИ нужно) — ДО создания полной структуры!
        if smooth > 0:
            logger.info(f"Smoothing mesh ({smooth} iterations)...")
            temp_mesh = Mesh(
                sites=points,
                elements=triangles,
                boundary_indices=Mesh._find_boundary_indices(triangles),
                n_lsfd_neighbors=n_lsfd_neighbors,
                ghost_coeff = ghost_coeff
            )
            smoothed_mesh = temp_mesh.smooth(iterations=smooth, create_submesh=False)
            points = smoothed_mesh.sites
            triangles = smoothed_mesh.elements

        # 3. Создаём ПОЛНЫЙ Mesh ОДИН РАЗ
        logger.info("Creating full Mesh object...")
        self.mesh = Mesh.from_triangulation(
            points,
            triangles,
            n_lsfd_neighbors=n_lsfd_neighbors,
            ghost_coeff=ghost_coeff,
        )

        # 4. Находим терминалы
        self._compute_terminal_data()

        logger.info(f"Mesh created: {self.n_sites} sites, {self.n_elements} triangles")

    def _compute_terminal_data(self) -> None:
        """Найти граничные вершины/ребра для каждого терминала."""
        if self.mesh is None:
            raise RuntimeError("Сначала вызовите make_mesh()")

        mesh = self.mesh
        self._terminal_data = {}

        # Граничные вершины и ребра
        boundary_site_indices = mesh.boundary_indices
        boundary_sites = mesh.sites[boundary_site_indices]
        boundary_edge_indices = mesh.tri_mesh.boundary_edge_indices
        boundary_edge_midpoints = mesh.tri_mesh.edge_midpoints[boundary_edge_indices]
        boundary_edge_lengths = mesh.tri_mesh.edge_lengths[boundary_edge_indices]

        for terminal in self.terminals:
            if not terminal.name:
                continue

            # Вершины внутри терминала
            site_mask = terminal.contains_points(boundary_sites, radius=1e-8)
            site_indices_local = np.where(site_mask)[0]
            site_indices = boundary_site_indices[site_indices_local]

            # Ребра внутри терминала (по серединам)
            edge_mask = terminal.contains_points(boundary_edge_midpoints, radius=1e-8)
            edge_indices_local = np.where(edge_mask)[0]
            edge_indices = boundary_edge_indices[edge_indices_local]

            # === ПРОВЕРКА 1: хотя бы одно ребро ===
            if len(edge_indices) == 0:
                raise ValueError(
                    f"Терминал '{terminal.name}' не пересекает границу плёнки! "
                    f"Найдено 0 граничных рёбер внутри терминала. "
                    f"Увеличьте размер терминала или измените позицию."
                )

            # === ПРОВЕРКА 2: связность рёбер (только если рёбер >= 2) ===
            if len(edge_indices) >= 2:
                if not check_boundary_connectivity(edge_indices, mesh.tri_mesh.edges):
                    raise ValueError(
                        f"Терминал '{terminal.name}' содержит разрозненные группы граничных рёбер! "
                        f"Все граничные точки терминала должны образовывать единую линию. "
                        f"Проверьте позицию и размер терминала."
                    )

            # Длина контакта
            length = boundary_edge_lengths[edge_indices_local].sum()
            center = np.array(terminal.polygon.centroid.coords[0])

            # Определяем тип терминала по имени (упрощённо)
            terminal_type = "source" if "source" in terminal.name.lower() else "drain"

            self._terminal_data[terminal.name] = TerminalData(
                name=terminal.name,
                site_indices=site_indices,
                edge_indices=edge_indices,
                length=length,
                center=center,
                terminal_type=terminal_type,
            )

        # === ПРОВЕРКА 3: если есть drain, должен быть source ===
        self._validate_terminals()

    def _validate_terminals(self) -> None:
        """Проверить корректность терминалов."""
        has_source = any(t.terminal_type == "source" for t in self._terminal_data.values())
        has_drain = any(t.terminal_type == "drain" for t in self._terminal_data.values())

        if has_drain and not has_source:
            raise ValueError(
                "Обнаружен drain но нет source! "
                "Для протекания тока нужны как минимум один source и один drain."
            )

        if not has_source and not has_drain and self._terminal_data:
            logger.warning("Нет терминалов типа source или drain")

    # ========================================================================
    # ДОСТУП К ДАННЫМ
    # ========================================================================

    @property
    def terminal_data(self) -> Dict[str, TerminalData]:
        """Данные о терминалах (после make_mesh)."""
        if not self._terminal_data:
            raise RuntimeError("Сначала вызовите make_mesh()")
        return self._terminal_data

    @property
    def source_terminals(self) -> Dict[str, TerminalData]:
        """Только терминалы типа source."""
        return {name: data for name, data in self.terminal_data.items()
                if data.terminal_type == "source"}

    @property
    def drain_terminals(self) -> Dict[str, TerminalData]:
        """Только терминалы типа drain."""
        return {name: data for name, data in self.terminal_data.items()
                if data.terminal_type == "drain"}

    def get_terminal_total_length(self, terminal_type: str) -> float:
        """Суммарная длина всех терминалов заданного типа."""
        terminals = self.source_terminals if terminal_type == "source" else self.drain_terminals
        return sum(data.length for data in terminals.values())

    # ========================================================================
    # УТИЛИТЫ: Граничные индексы для source/drain
    # ========================================================================

    def get_boundary_site_indices(self, terminal_type: str) -> np.ndarray:
        """
        Получить все граничные вершины для терминалов заданного типа.

        Args:
            terminal_type: "source" или "drain".

        Returns:
            Массив индексов вершин (concatenated для всех терминалов типа).
        """
        if terminal_type == "source":
            terminals = self.source_terminals
        elif terminal_type == "drain":
            terminals = self.drain_terminals
        else:
            raise ValueError(f"terminal_type должен быть 'source' или 'drain', got {terminal_type}")

        if not terminals:
            return np.array([], dtype=np.int64)

        indices_list = [data.site_indices for data in terminals.values()]
        return np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int64)

    def get_boundary_edge_indices(self, terminal_type: str) -> np.ndarray:
        """
        Получить все граничные рёбра для терминалов заданного типа.

        Args:
            terminal_type: "source" или "drain".

        Returns:
            Массив индексов рёбер (concatenated для всех терминалов типа).
        """
        if terminal_type == "source":
            terminals = self.source_terminals
        elif terminal_type == "drain":
            terminals = self.drain_terminals
        else:
            raise ValueError(f"terminal_type должен быть 'source' или 'drain', got {terminal_type}")

        if not terminals:
            return np.array([], dtype=np.int64)

        indices_list = [data.edge_indices for data in terminals.values()]
        return np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int64)

    def get_all_boundary_site_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить все граничные вершины для source и drain.

        Returns:
            (source_indices, drain_indices) — два массива индексов вершин.
        """
        source_indices = self.get_boundary_site_indices("source")
        drain_indices = self.get_boundary_site_indices("drain")
        return source_indices, drain_indices

    def get_all_boundary_edge_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить все граничные рёбра для source и drain.

        Returns:
            (source_indices, drain_indices) — два массива индексов рёбер.
        """
        source_indices = self.get_boundary_edge_indices("source")
        drain_indices = self.get_boundary_edge_indices("drain")
        return source_indices, drain_indices

    # ========================================================================
    # СВОЙСТВА ДЛЯ БЫСТРОГО ДОСТУПА
    # ========================================================================

    @property
    def source_site_indices(self) -> np.ndarray:
        """Все граничные вершины source терминалов."""
        return self.get_boundary_site_indices("source")

    @property
    def drain_site_indices(self) -> np.ndarray:
        """Все граничные вершины drain терминалов."""
        return self.get_boundary_site_indices("drain")

    @property
    def source_edge_indices(self) -> np.ndarray:
        """Все граничные рёбра source терминалов."""
        return self.get_boundary_edge_indices("source")

    @property
    def drain_edge_indices(self) -> np.ndarray:
        """Все граничные рёбра drain терминалов."""
        return self.get_boundary_edge_indices("drain")

    @property
    def sites(self) -> Optional[np.ndarray]:
        """Координаты вершин [ξ]."""
        return self.mesh.sites if self.mesh else None

    @property
    def triangles(self) -> Optional[np.ndarray]:
        """Индексы треугольников."""
        return self.mesh.elements if self.mesh else None

    @property
    def n_sites(self) -> int:
        return len(self.mesh.sites) if self.mesh else 0

    @property
    def n_elements(self) -> int:
        return len(self.mesh.elements) if self.mesh else 0

    @property
    def triangulation(self) -> Optional[Triangulation]:
        """Matplotlib triangulation для plot."""
        if self.mesh is None or self._triangulation is not None:
            return self._triangulation
        self._triangulation = Triangulation(
            self.mesh.sites[:, 0], self.mesh.sites[:, 1], self.mesh.elements
        )
        return self._triangulation

    @property
    def polygons(self) -> Tuple[Polygon, ...]:
        """Все полигоны устройства."""
        return (self.film,) + tuple(self.holes) + tuple(self.terminals)

    # ========================================================================
    # ГЕОМЕТРИЧЕСКИЕ ОПЕРАЦИИ
    # ========================================================================

    def copy(self, with_mesh: bool = True) -> "Device":
        """Создать копию устройства."""
        device = Device(
            name=self.name,
            film=self.film.copy(),
            holes=[h.copy() for h in self.holes],
            terminals=[t.copy() for t in self.terminals],
            probe_points=self.probe_points.copy() if self.probe_points is not None else None,
        )
        if with_mesh and self.mesh is not None:
            device.mesh = self.mesh
            device._terminal_data = self._terminal_data.copy()
        return device

    def _warn_if_mesh_exists(self, method: str) -> None:
        if self.mesh is not None:
            logger.warning(
                f"device.{method}() called on device with existing mesh. "
                f"New device will have no mesh. Call new_device.make_mesh() to regenerate."
            )

    def scale(self, xfact: float = 1.0, yfact: float = 1.0, origin: Tuple[float, float] = (0, 0)) -> "Device":
        """Масштабировать устройство."""
        self._warn_if_mesh_exists("scale")
        device = self.copy(with_mesh=False)
        for polygon in device.polygons:
            polygon.scale(xfact=xfact, yfact=yfact, origin=origin, inplace=True)
        if device.probe_points is not None:
            points = [affinity.scale(Point(xy), xfact=xfact, yfact=yfact, origin=origin)
                      for xy in device.probe_points]
            device.probe_points = np.array([p.coords[0] for p in points])
        return device

    def rotate(self, degrees: float, origin: Tuple[float, float] = (0, 0)) -> "Device":
        """Повернуть устройство."""
        self._warn_if_mesh_exists("rotate")
        device = self.copy(with_mesh=False)
        for polygon in device.polygons:
            polygon.rotate(degrees, origin=origin, inplace=True)
        if device.probe_points is not None:
            points = [affinity.rotate(Point(xy), degrees, origin=origin)
                      for xy in device.probe_points]
            device.probe_points = np.array([p.coords[0] for p in points])
        return device

    def translate(self, dx: float = 0, dy: float = 0, inplace: bool = False) -> "Device":
        """Сдвинуть устройство."""
        if inplace:
            device = self
        else:
            self._warn_if_mesh_exists("translate")
            device = self.copy(with_mesh=False)

        for polygon in device.polygons:
            polygon.translate(dx, dy, inplace=True)
        if device.probe_points is not None:
            device.probe_points = device.probe_points + np.array([dx, dy])

        return device

    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """Центр масс пленки."""
        return tuple(self.film.polygon.centroid.coords[0])

    def contains_points(self, points: np.ndarray, index: bool = False, radius: float = 0) -> np.ndarray:
        """Проверить, лежат ли точки внутри устройства (учитывая отверстия)."""
        mask = self.film.contains_points(points, radius=radius)
        for hole in self.holes:
            mask &= ~hole.contains_points(points, radius=-radius)
        return np.where(mask)[0] if index else mask

    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================

    def plot(self, ax=None, legend: bool = True, figsize=None,
             mesh: bool = False, mesh_kwargs: dict = None,
             show_terminals: bool = True,  # ← НОВЫЙ ПАРАМЕТР
             terminal_colors: dict = None,  # ← НОВЫЙ ПАРАМЕТР: {name: color}
             **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Построить устройство и опционально сетку."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        if mesh_kwargs is None:
            mesh_kwargs = dict(color='k', lw=0.5)

        # Сетка
        if mesh and self.mesh is not None:
            ax.triplot(self.sites[:, 0], self.sites[:, 1], self.triangles, **mesh_kwargs)

        # Полигоны
        for polygon in self.polygons:
            ax = polygon.plot(ax=ax, **kwargs)

        # === НОВОЕ: Визуализация терминалов как выделенных линий на границе ===
        if show_terminals and self.mesh is not None and self._terminal_data:
            mesh = self.mesh
            all_edges = mesh.tri_mesh.edges

            # Отслеживаем, какие терминалы уже добавлены в легенду
            plotted_terminals = set()

            for name, term_data in self._terminal_data.items():
                # Цвет терминала (по умолчанию: source=зелёный, drain=красный)
                default_colors = {'source': 'green', 'drain': 'red'}
                color = terminal_colors.get(name, default_colors.get(term_data.terminal_type,
                                                                     'blue')) if terminal_colors else default_colors.get(
                    term_data.terminal_type, 'blue')

                # Рисуем рёбра терминала жирными линиями
                for edge_idx in term_data.edge_indices:
                    v1, v2 = all_edges[edge_idx]
                    p1 = mesh.sites[v1]
                    p2 = mesh.sites[v2]

                    # Добавляем в легенду только первый раз для каждого имени
                    if name not in plotted_terminals:
                        label = name
                        plotted_terminals.add(name)
                    else:
                        label = "_nolegend_"

                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=color, linewidth=3, label=label)

        # Probe points
        if self.probe_points is not None:
            ax.plot(*self.probe_points.T, 'ko', label='Probe points', markersize=3)

        if legend:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        ax.set_aspect('equal')
        ax.set_xlabel('x [ξ]')
        ax.set_ylabel('y [ξ]')

        return fig, ax

    def draw(self, ax=None, legend: bool = True, figsize=None,
             alpha: float = 0.5, exclude: Union[str, List[str]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Нарисовать устройство как заполненные полигоны."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.get_figure()

        exclude = [exclude] if isinstance(exclude, str) else (exclude or [])

        # Патчи для полигонов
        patches = self.patches()

        # Авто-масштаб
        x, y = self.film.points.T
        margin = 0.1
        dx, dy = np.ptp(x) * (1 + margin), np.ptp(y) * (1 + margin)
        x0, y0 = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2

        ax.set_xlim(x0 - dx / 2, x0 + dx / 2)
        ax.set_ylim(y0 - dy / 2, y0 + dy / 2)
        ax.set_aspect('equal')
        ax.grid(False)
        ax.set_xlabel('x [ξ]')
        ax.set_ylabel('y [ξ]')

        # Добавляем патчи
        handles, labels = [], []
        for i, (name, patch) in enumerate(patches.items()):
            if name in exclude:
                continue
            patch.set_alpha(alpha)
            patch.set_color(f'C{i % 10}')
            ax.add_artist(patch)
            handles.append(patch)
            labels.append(name)

        # Probe points
        if self.probe_points is not None:
            line, = ax.plot(*self.probe_points.T, 'ko', label='Probe points', markersize=3)
            handles.append(line)
            labels.append('Probe points')

        if legend and handles:
            ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left')

        return fig, ax

    def patches(self) -> Dict[str, PathPatch]:
        """Вернуть PathPatch для каждого полигона (для draw)."""
        hole_names = {h.name for h in self.holes}
        patches = {}

        for polygon in self.polygons:
            if polygon.name in hole_names:
                continue

            coords = polygon.points.tolist()
            codes = [MplPath.LINETO] * len(coords)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY

            poly = polygon.polygon
            for hole in self.holes:
                if poly.contains(hole.polygon):
                    hole_coords = hole.points.tolist()[::-1]
                    hole_codes = [MplPath.LINETO] * len(hole_coords)
                    hole_codes[0] = MplPath.MOVETO
                    hole_codes[-1] = MplPath.CLOSEPOLY
                    coords.extend(hole_coords)
                    codes.extend(hole_codes)

            patches[polygon.name] = PathPatch(MplPath(coords, codes))

        return patches

    # ========================================================================
    # HDF5: СОХРАНЕНИЕ / ЗАГРУЗКА
    # ========================================================================

    def save(self, filepath: str, save_mesh: bool = True, compress: bool = True) -> None:
        """Сохранить устройство в HDF5."""
        filepath = Path(filepath)
        if filepath.suffix != '.h5':
            filepath = filepath.with_suffix('.h5')
        filepath.parent.mkdir(parents=True, exist_ok=True)

        compression = 'gzip' if compress else None

        with h5py.File(filepath, 'w') as f:
            f.attrs['name'] = self.name

            # Геометрия
            self.film.to_hdf5(f.create_group('film'))

            if self.holes:
                holes_grp = f.create_group('holes')
                for i, hole in enumerate(self.holes):
                    hole.to_hdf5(holes_grp.create_group(f'hole_{i}'))

            if self.terminals:
                terms_grp = f.create_group('terminals')
                for term in self.terminals:
                    term.to_hdf5(terms_grp.create_group(term.name))

            if self.probe_points is not None:
                f.create_dataset('probe_points', data=self.probe_points, compression=compression)

            # Сетка
            if save_mesh and self.mesh is not None:
                self.mesh.to_hdf5(f.create_group('mesh'), compress=compress)

    @classmethod
    def load(cls, filepath: str, load_mesh: bool = True) -> "Device":
        """Загрузить устройство из HDF5."""
        filepath = Path(filepath)

        with h5py.File(filepath, 'r') as f:
            name = f.attrs['name']

            film = Polygon.from_hdf5(f['film'])

            holes = []
            if 'holes' in f:
                for grp in f['holes'].values():
                    holes.append(Polygon.from_hdf5(grp))

            terminals = []
            if 'terminals' in f:
                for grp in f['terminals'].values():
                    terminals.append(Polygon.from_hdf5(grp))

            probe_points = None
            if 'probe_points' in f:
                probe_points = np.array(f['probe_points'])

            device = cls(
                name=name,
                film=film,
                holes=holes,
                terminals=terminals,
                probe_points=probe_points,
            )

            if load_mesh and 'mesh' in f:
                device.mesh = Mesh.from_hdf5(f['mesh'])
                device._compute_terminal_data()

            return device

    # ========================================================================
    # УТИЛИТЫ
    # ========================================================================

    def closest_site(self, xy: Tuple[float, float]) -> int:
        """Индекс ближайшей вершины к точке (x, y)."""
        if self.mesh is None:
            raise RuntimeError("Сначала вызовите make_mesh()")
        return np.argmin(np.linalg.norm(self.mesh.sites - np.atleast_2d(xy), axis=1))

    def __repr__(self) -> str:
        parts = [
            f"name={self.name!r}",
            f"sites={self.n_sites}",
            f"triangles={self.n_elements}",
            f"terminals={len(self.terminals)}",
            f"holes={len(self.holes)}",
        ]
        return f"Device({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, Device):
            return False
        return (self.name == other.name and
                self.film == other.film and
                len(self.holes) == len(other.holes) and
                len(self.terminals) == len(other.terminals))