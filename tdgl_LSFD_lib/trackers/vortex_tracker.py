"""Dynamic phase-winding vortex IDs with buffered HDF5 trajectory tables."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .base_tracker import SimulationTracker
from ..solver.solver import StepResult, TDGLSolver


DYN_DTYPE = np.dtype([
    ("step", "<i8"), ("time", "<f8"), ("vortex_id", "<i8"),
    ("x", "<f8"), ("y", "<f8"), ("vx", "<f8"), ("vy", "<f8"),
    ("triangle_index", "<i8"),
])


STAT_DTYPE = np.dtype([
    ("vortex_id", "<i8"), ("charge", "<i8"), ("birth", "<f8"),
])


DEATH_DTYPE = np.dtype([
    ("vortex_id", "<i8"), ("death", "<f8"),
])


@dataclass
class VortexState:
    charge: int
    tri_index: int
    velocity: np.ndarray

    times: deque = field(default_factory=deque)
    positions: deque = field(default_factory=deque)


class VortexTracker(SimulationTracker):
    """Track all phase-zero cores with non-reusable, zero-based vortex IDs.

    Detection assumes valid psi on nondegenerate CCW triangles, without phase
    offsets, boundary/separation filters or root checks. Births and immutable
    charges are stored once, deaths separately, and positions/velocities at
    every observed step. Live regression histories survive HDF5 flushes.
    """

    def __init__(
        self,
        sites: np.ndarray,
        triangles: np.ndarray,
        max_match_distance=1.0,
        velocity_window=0.05,
        velocity_min_points=5,
        buffer_size: int = 100,
    ):

        if max_match_distance is None:
            max_match_distance = 1.0
        if not np.isfinite(max_match_distance) or max_match_distance <= 0:
            raise ValueError("max_match_distance must be finite and positive")
        if not np.isfinite(velocity_window) or velocity_window <= 0:
            raise ValueError("velocity_window must be finite and positive")
        if int(velocity_min_points) != velocity_min_points or velocity_min_points < 3:
            raise ValueError("velocity_min_points must be an integer >= 3")
        if int(buffer_size) != buffer_size or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer")

        self.sites = np.asarray(sites, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.max_match_distance = float(max_match_distance)
        self.velocity_window = float(velocity_window)
        self.velocity_min_points = int(velocity_min_points)
        self.buffer_size = int(buffer_size)
        self.reset()

    def reset(self):
        """Новый независимый прогон; после reset нужен новый HDF5-файл/группа."""
        self.active = {}
        self.next_id = 0
        self.buffer_deaths = []
        self.buffer_dyn = []
        self.buffer_stat = []
        self._idx = 0  # Число шагов, а не число строк samples; контракт Runner.
        self.flush_requested = False

    def on_step(
        self,
        step: int,
        t: float,
        dt: float,
        result: StepResult,
        solver: TDGLSolver,
    ) -> Dict[str, Any]:
        """Detect cores, continue IDs, estimate velocities and buffer one step."""
        if self._idx >= self.buffer_size:
            raise BufferError("Tracker buffer is full; call flush_to_hdf5 before the next step")

        detected = self.detect_vortices(result.psi)
        positions = detected["positions"]
        charges = detected["charges"]
        triangle_indices = detected["triangle_indices"]

        self.match_to_previous(t, positions, charges, triangle_indices)
        samples = np.empty(len(self.active), dtype=DYN_DTYPE)

        for i, (vortex_id, state) in enumerate(self.active.items()):
            samples[i] = (
                step,
                t,
                vortex_id,
                state.positions[-1][0],
                state.positions[-1][1],
                state.velocity[0],
                state.velocity[1],
                state.tri_index,
            )

        self.buffer_dyn.append(samples)
        self._idx +=1
        self.flush_requested = bool(self.buffer_deaths)
        return {"vortex_count": len(samples), "vortices": samples}


    def detect_vortices(self, psi):
        """Обход фазы + линейный комплексный ноль внутри треугольника.

        q=+1 означает рост фазы на 2*pi при обходе CCW со стороны +z.
        Фильтров границы, физического расстояния и числа ядер здесь нет.
        """
        psi = np.asarray(psi, dtype=np.complex128)
        values = psi[self.triangles]
        phases = np.angle(values)
        differences = np.roll(phases, -1, axis=1) - phases
        wrapped = (differences + np.pi) % (2 * np.pi) - np.pi
        circulation = np.sum(wrapped, axis=1)

        triangle_charges = np.rint(circulation / (2 * np.pi)).astype(np.int64)
        confirmed = np.flatnonzero(triangle_charges != 0) # triangles with vortices

        positions = []
        triangle_indices = []
        charges = []

        for triangle_idx in confirmed:
            vertices = self.triangles[triangle_idx]
            z = values[triangle_idx]
            z_norm = z / np.max(np.abs(z))

            # Система для барицентрических координат
            matrix = np.array([
                [z_norm[0].real - z_norm[2].real,
                 z_norm[1].real - z_norm[2].real],
                [z_norm[0].imag - z_norm[2].imag,
                 z_norm[1].imag - z_norm[2].imag],
            ])
            ab = np.linalg.solve(matrix, -np.array([z_norm[2].real, z_norm[2].imag]))
            barycentric = np.array([ab[0], ab[1], 1.0 - ab.sum()])
            # Декартовы координаты центра вихря
            root = barycentric @ self.sites[vertices]

            positions.append(root)
            triangle_indices.append(triangle_idx)
            charges.append(triangle_charges[triangle_idx])

        return {
            "positions": np.asarray(positions, dtype=np.float64).reshape(-1, 2),
            "charges": np.asarray(charges, dtype=np.int64),
            "triangle_indices": np.asarray(triangle_indices, dtype=np.int64),
        }

    def match_to_previous(self, t: float, positions: np.ndarray, charges: np.ndarray, triangle_indices: np.ndarray):
        """Однозначные ближайшие пары одного заряда с порогом расстояния."""

        pairs = []
        for vortex_id, state in self.active.items():
            distances = np.linalg.norm(positions - state.positions[-1], axis=1)
            candidates = np.flatnonzero((distances <= self.max_match_distance)
                                        & (charges == state.charge))
            pairs.extend((distances[j], vortex_id, j)for j in candidates)

        matched, used = {}, set()
        for _, vortex_id, j in sorted(pairs):
            if vortex_id not in matched and j not in used:
                matched[vortex_id] = j
                self.active[vortex_id].tri_index = int(triangle_indices[j])
                self.active[vortex_id].times.append(t)
                self.active[vortex_id].positions.append(positions[j].copy())
                self.compute_smoothed_velocities(vortex_id)

                used.add(j)

        disappeared = sorted( set(self.active.keys()) - matched.keys() )
        if disappeared:
            samples_death = np.empty(len(disappeared), dtype=DEATH_DTYPE)
            samples_death["vortex_id"] = disappeared
            samples_death["death"] = t
            for vortex_id in disappeared:
                del self.active[vortex_id]

            self.buffer_deaths.append(samples_death)

        appeared = [j for j in range(len(positions)) if j not in used]
        if appeared:
            app_charges = []
            app_ides = []

            for i in appeared:
                self.active[self.next_id] = VortexState(charge=charges[i], tri_index=triangle_indices[i],
                                                    velocity= np.full(2, np.nan))
                self.active[self.next_id].times.append(t)
                self.active[self.next_id].positions.append(positions[i].copy())
                app_charges.append(charges[i])
                app_ides.append(self.next_id)
                self.next_id += 1

            samples_stat = np.empty(len(appeared), dtype=STAT_DTYPE)
            samples_stat["vortex_id"] = app_ides
            samples_stat["charge"] = app_charges
            samples_stat["birth"] = t

            self.buffer_stat.append(samples_stat)

        return


    def compute_smoothed_velocities(self, vortex_id: int):
        """Квадратичная регрессия отдельно по непрерывной истории каждого ID."""

        state = self.active[vortex_id]
        time = state.times[-1]

        tolerance = 16 * np.finfo(float).eps * max(1.0, abs(time))
        while state.times[0] < time - self.velocity_window - tolerance:
            state.times.popleft()
            state.positions.popleft()

        if len(state.times) < self.velocity_min_points:
            state.velocity = np.full(2, np.nan)

        else:
            u = (np.asarray(state.times) - time) / self.velocity_window
            matrix = np.column_stack( (np.ones_like(u), u, u ** 2) )
            xy = np.asarray(state.positions) - state.positions[-1]
            coefficients = np.linalg.lstsq(matrix, xy, rcond=None)[0]
            state.velocity = coefficients[1] / self.velocity_window

        return


    def flush_to_hdf5(self, hdf5_group):
        """Append v2 tables after Runner's clock, preserving all live histories."""
        if self._idx == 0:
            return
        if len(self.buffer_dyn) != self._idx:
            raise ValueError("buffer_dyn must contain one array per buffered step")
        previous_frames = (len(hdf5_group["vortices/frame_offsets"]) - 1
                           if "vortices/frame_offsets" in hdf5_group else 0)
        end_frame = previous_frames + self._idx
        if "time" not in hdf5_group or len(hdf5_group["time"]) < end_frame:
            raise ValueError("Write Runner time/step buffers before flushing the v2 tracker")
        group = hdf5_group.require_group("vortices")
        if "dyn_info" not in group:
            group.create_dataset("dyn_info", (0,), maxshape=(None,), dtype=DYN_DTYPE)
            group.create_dataset("stat_info", (0,), maxshape=(None,), dtype=STAT_DTYPE)
            group.create_dataset("deaths", (0,), maxshape=(None,), dtype=DEATH_DTYPE)
            group.create_dataset("frame_offsets", data=np.array([0], dtype=np.int64), maxshape=(None,))
            group.attrs["schema_version"] = 2
            group.attrs["length_unit"] = "xi"
            group.attrs["velocity_method"] = "local_quadratic_derivative"
            group.attrs["velocity_window"] = self.velocity_window
            group.attrs["velocity_min_points"] = self.velocity_min_points
            group.attrs["max_match_distance"] = self.max_match_distance
            group.attrs["charge_convention"] = "CCW phase winding viewed from +z"
            group.attrs["death_definition"] = "first frame without the ID; no death row means not observed to die"
            group.attrs["death_source"] = "buffer_deaths"

        offsets = group["frame_offsets"]
        death_rows = (np.concatenate(self.buffer_deaths) if self.buffer_deaths
                      else np.empty(0, dtype=DEATH_DTYPE))

        def append(dataset, rows):
            if len(rows):
                start = len(dataset)
                dataset.resize(start + len(rows), axis=0)
                dataset[start:] = rows

        stat_rows = (np.concatenate(self.buffer_stat) if self.buffer_stat
                     else np.empty(0, dtype=STAT_DTYPE))
        counts = np.array([len(rows) for rows in self.buffer_dyn], dtype=np.int64)
        append(group["stat_info"], stat_rows)
        append(group["dyn_info"], np.concatenate(self.buffer_dyn))
        append(group["deaths"], death_rows)
        append(offsets, int(offsets[-1]) + np.cumsum(counts))
        self.buffer_dyn.clear()
        self.buffer_stat.clear()
        self.buffer_deaths.clear()
        self._idx = 0
        if hasattr(self, "flush_requested"):
            self.flush_requested = False
