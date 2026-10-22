"""
dynamics_options.py — dynamics options of TDGL simulation
"""
from dataclasses import dataclass
from enum import Enum
from typing import Union
from pathlib import Path
from math import isfinite


class TimeScheme(Enum):
    EULER = "euler"  # Explicit Euler scheme with fixed time step (order 1)
    ADAPTIVE_EULER = "adaptive_euler"  # Explicit Euler scheme with adaptive time step (order 1)
    CRANK_NICOLSON = "crank_nicolson"  # Implicit Crank-Nicolson scheme with fixed time step (order 2)


class RunMode(Enum):
    """Simulation run mode."""
    PC = "pc"          # Interactive use: tqdm (progress bar) + pause with Ctrl+C
    CLUSTER = "cluster"  # Cluster use: log + immediate exit with Ctrl+C


@dataclass
class SolverOptions:
    """Dynamic parameters for TDGL solver."""

    # === Time ===
    solve_time: float   # Total solution time
    skip_time: float = 0.0  # Time to skip before starting to save results

    # === Time step ===
    dt_init: float = 1e-6  # Initital time step
    dt_max: float = 1e-2  #  Maximum time step
    dt_min: float = 1e-8  # Minimum time step
    adaptive_window: int = 10  # Averaging window for adaptive time step

    # === Time scheme ===
    time_scheme: Union[TimeScheme, str] = TimeScheme.EULER

    # === Parameters of Poisson solver ===
    poisson_tolerance_init: float = 1e-4      # Initial tolerance
    poisson_tolerance_max: float = 1e-3     # Maximum tolerance
    poisson_tolerance_min: float = 5e-6       # Minimum tolerance
    poisson_iterations: int = 5000                 # Maximum number of iterations
    poisson_adaptive: bool = True                  # Use adaptive tolerance

    # === Save ===
    output_file: Union[str, Path, None] = None # Name for output_file
    save_every: int = 100

    # === Angle update ===
    # Update G matrix for case with s(t) (rotation of ferromagnetic field), if s rotates on fixed angle
    update_G_angle_threshold: float = 0.05  # рад (~2.9°)

    # === Trackers ===
    # ConservationTracker: energy, current conservation, Poisson residual, boundary currents
    track_conservation: bool = True   # Enable ConservationTracker (auto-created by Runner)
    # PhysicalTracker: mu and phase at probe points (requires device.probe_points)
    track_physical: bool = False       # Enable PhysicalTracker (auto-created by Runner)

    track_vortices: bool = False      # Enable VortexTracker (auto-created by Runner)
    # Phase-zero detection on the physical CCW triangulation.
    vortex_max_match_distance: float | None = 1.0  # xi per step; None also means 1.0
    vortex_velocity_window: float = 0.05  # Trailing window for quadratic velocity regression
    vortex_velocity_min_points: int = 5  # At least 3 points for the quadratic fit
    # Tracker logging
    tracker_log_every: int = 1000     # Log diagnostics every N steps (None to disable)

    # RunMode parameters for interactive or cluster use
    run_mode: Union[RunMode, str] = RunMode.PC

    # log file and structure
    log_file: Union[str, Path, None] = None
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    def __post_init__(self):
        # Преобразуем строку в Enum
        if isinstance(self.run_mode, str):
            try:
                self.run_mode = RunMode(self.run_mode.lower())
            except ValueError:
                valid = [m.value for m in RunMode]
                raise ValueError(f"run_mode must be one of {valid!r}, got '{self.run_mode}'")

        # Устанавливаем флаги
        if self.run_mode == RunMode.CLUSTER:
            self.use_tqdm = False
            self.use_pause = False
        else:
            self.use_tqdm = True
            self.use_pause = True

        self.validate()

    def validate(self) -> None:
        """Validate parameter correctness."""
        if self.solve_time <= 0:
            raise ValueError(f"solve_time must be > 0 (got {self.solve_time})")
        if self.skip_time < 0:
            raise ValueError(f"skip_time must be >= 0 (got {self.skip_time})")

        if self.dt_init <= 0:
            raise ValueError(f"dt_init must be > 0 (got {self.dt_init})")
        if self.dt_max <= 0:
            raise ValueError(f"dt_max must be > 0 (got {self.dt_max})")
        if self.dt_min <= 0:
            raise ValueError(f"dt_min must be > 0 (got {self.dt_min})")
        if self.dt_init > self.dt_max:
            raise ValueError(f"dt_init ({self.dt_init}) must be <= dt_max ({self.dt_max})")
        if self.dt_min > self.dt_init:
            raise ValueError(f"dt_min ({self.dt_min}) must be <= dt_init ({self.dt_init})")
        if self.adaptive_window < 1:
            raise ValueError(f"adaptive_window must be >= 1 (got {self.adaptive_window})")

        if self.save_every < 1:
            raise ValueError(f"save_every must be >= 1 (got {self.save_every})")

        # Poisson solver validation
        if self.poisson_tolerance_min <= 0:
            raise ValueError(f"poisson_tolerance_min must be > 0 (got {self.poisson_tolerance_min})")
        if self.poisson_tolerance_init <= 0:
            raise ValueError(f"poisson_tolerance_init must be > 0 (got {self.poisson_tolerance_init})")
        if self.poisson_tolerance_max <= 0:
            raise ValueError(f"poisson_tolerance_max must be > 0 (got {self.poisson_tolerance_max})")
        if self.poisson_tolerance_min > self.poisson_tolerance_init:
            raise ValueError(
                f"poisson_tolerance_min ({self.poisson_tolerance_min}) must be <= "
                f"poisson_tolerance_init ({self.poisson_tolerance_init})"
            )
        if self.poisson_tolerance_init > self.poisson_tolerance_max:
            raise ValueError(
                f"poisson_tolerance_init ({self.poisson_tolerance_init}) must be <= "
                f"poisson_tolerance_max ({self.poisson_tolerance_max})"
            )
        if self.poisson_iterations < 1:
            raise ValueError(f"poisson_iterations must be >= 1 (got {self.poisson_iterations})")

        # Trackers validation
        if self.tracker_log_every is not None and self.tracker_log_every < 1:
            raise ValueError(
                f"tracker_log_every must be >= 1 or None (got {self.tracker_log_every})"
            )
        # !!!14.10_vort1!!! Optional cap for nearest-neighbor vortex identity matching.
        if self.vortex_max_match_distance is not None and (
                not isfinite(self.vortex_max_match_distance) or self.vortex_max_match_distance <= 0):
            raise ValueError(
                "vortex_max_match_distance must be finite and > 0 or None "
                f"(got {self.vortex_max_match_distance})"
            )
        if not isfinite(self.vortex_velocity_window) or self.vortex_velocity_window <= 0:
            raise ValueError(
                f"vortex_velocity_window must be finite and > 0 (got {self.vortex_velocity_window})"
            )
        if int(self.vortex_velocity_min_points) != self.vortex_velocity_min_points or self.vortex_velocity_min_points < 3:
            raise ValueError(
                "vortex_velocity_min_points must be an integer >= 3 "
                f"(got {self.vortex_velocity_min_points})"
            )

        # Convert time_scheme from string to Enum
        if isinstance(self.time_scheme, str):
            try:
                self.time_scheme = TimeScheme[self.time_scheme.upper()]
            except KeyError:
                valid = list(TimeScheme.__members__.keys())
                raise ValueError(f"time_scheme must be one of {valid!r}, got {self.time_scheme}")

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{self.log_level}'")

    def __repr__(self):
        return (
            f"SolverOptions(\n"
            f"  run_mode={self.run_mode.value},\n"
            f"  solve_time={self.solve_time},\n"
            f"  time_scheme={self.time_scheme.value},\n"
            f"  save_every={self.save_every},\n"
            f"  poisson_tolerance_init={self.poisson_tolerance_init:.2e},\n"
            f"  poisson_adaptive={self.poisson_adaptive},\n"
            f"  track_conservation={self.track_conservation},\n"
            f"  track_physical={self.track_physical},\n"
            f"  track_vortices={self.track_vortices},\n"
            f"  use_tqdm={self.use_tqdm},\n"
            f"  use_pause={self.use_pause}\n"
            f")"
        )
