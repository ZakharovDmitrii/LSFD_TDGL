"""
solution.py — Container for TDGL simulation results.

Loads data from HDF5 and provides convenient access to:
    - Spatial fields (psi, mu, currents) at specific time steps
    - Time series of scalar quantities (energy, fluxes, probe values, etc.)
    - Mesh geometry and metadata

HDF5 structure (new architecture):
    output.h5
    ├── mesh/                    — mesh geometry (once)
    ├── A_for_constant_Bz        — fixed field (once)
    ├── data/                    — spatial snapshots every save_every steps
    │   ├── 0/
    │   │   ├── psi, mu, supercurrent_x, ...
    │   │   └── attrs: step, time, dt
    │   └── 1/
    └── time_series/             — scalar time series (every step)
        ├── time, step, dt, poisson_tolerance   (from Runner)
        ├── energy_voronoi, flux_edges, ...     (from ConservationTracker)
        ├── mu_probe, phase_probe               (from PhysicalTracker)
        └── probe_points_coords, probe_points_indices
"""
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class Solution:
    """
    Container for TDGL simulation results.

    Provides convenient access to spatial fields and time series data
    stored in HDF5 format. All time series methods return tuples of
    (time_array, values_array) with matching lengths.

    Args:
        path: Path to HDF5 file with simulation results.

    # Example:
    #     >>> sol = Solution("output.h5")
    #     >>> print(sol)
    #     Solution(output.h5, steps=100, sites=50000)
    #
    #     >>> # Get spatial fields at last step
    #     >>> psi = sol.get_spatial_data(step=-1)['psi']
    #
    #     >>> # Get energy time series
    #     >>> time, energy = sol.get_energy_series()
    #     >>> plt.plot(time, energy)
    #
    #     >>> # List all available time series
    #     >>> print(sol.list_time_series())
    #     ['time', 'step', 'dt', 'energy_voronoi', ...]
    # """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        self._load_metadata()
        self._load_mesh()

    def _load_metadata(self) -> None:
        """Load simulation metadata and step range."""
        with h5py.File(self.path, 'r') as f:
            # Simulation parameters (if stored as attributes)
            self.solve_time = f.attrs.get('solve_time', None)
            self.skip_time = f.attrs.get('skip_time', None)
            self.dt_init = f.attrs.get('dt_init', None)
            self.save_every = f.attrs.get('save_every', 100)

            # Step range from data group
            if 'data' in f:
                steps = sorted([int(k) for k in f['data'].keys()])
                self.step_min = min(steps)
                self.step_max = max(steps)
                self.n_saved_steps = len(steps)
                self._saved_steps = steps
            else:
                self.step_min = self.step_max = 0
                self.n_saved_steps = 0
                self._saved_steps = []

            # Time series keys
            if 'time_series' in f:
                self._time_series_keys = list(f['time_series'].keys())
            else:
                self._time_series_keys = []

    def _load_mesh(self) -> None:
        """Load mesh geometry."""
        with h5py.File(self.path, 'r') as f:
            if 'mesh' in f:
                mesh_grp = f['mesh']
                self.sites = np.array(mesh_grp['sites'])
                self.triangles = np.array(mesh_grp['elements'])
                if 'dual_mesh/dual_areas' in mesh_grp:
                    self.voronoi_areas = np.array(mesh_grp['dual_mesh/dual_areas'])
                else:
                    self.voronoi_areas = None
            else:
                self.sites = None
                self.triangles = None
                self.voronoi_areas = None

    # =========================================================================
    # Spatial data access
    # =========================================================================

    def get_spatial_data(self, step: int = -1) -> Dict[str, np.ndarray]:
        if step < 0:
            step = self._saved_steps[step]
        elif step not in self._saved_steps:
            raise ValueError(f"Step {step} not found. Available: {self._saved_steps}")
        with h5py.File(self.path, 'r') as f:
            grp = f['data'][str(step)]
            data = {
                'time': grp.attrs.get('time', 0.0),
                'dt': grp.attrs.get('dt', 0.0),
                'step': grp.attrs.get('step', step),
                'Bz': grp.attrs.get('Bz', None),
                'eta': grp.attrs.get('eta', None),
                'gamma': grp.attrs.get('gamma', None),
                's_applied': np.array(grp.attrs['s_applied']) if 's_applied' in grp.attrs else None,
            }
            for key in ['psi', 'mu', 'psi_derivatives', 'supercurrent_x', 'supercurrent_y',
                        'div_Js', 'normal_current', 'A_applied', 'J_boundary']:
                if key in grp:
                    data[key] = np.array(grp[key])
        return data

    # =========================================================================
    # Time series access
    # =========================================================================

    def get_time_series(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time series for a scalar quantity.

        Args:
            key: Name of the quantity (e.g., 'energy_voronoi', 'flux_edges', 'mu_probe').

        Returns:
            Tuple (time_array, values_array) with matching lengths.
            For vector quantities (e.g., mu_probe), values_array has shape (n_times, n_probes).

        Example:
        #    >>> time, energy = sol.get_time_series('energy_voronoi')
        #    >>> plt.plot(time, energy)
        """
        if key not in self._time_series_keys:
            raise KeyError(f"Time series '{key}' not found. Available: {self._time_series_keys}")

        with h5py.File(self.path, 'r') as f:
            time = np.array(f['time_series']['time'])
            values = np.array(f['time_series'][key])

        return time, values

    def list_time_series(self) -> List[str]:
        """List all available time series keys."""
        return self._time_series_keys.copy()

    # =========================================================================
    # Specialized time series methods
    # =========================================================================

    def get_energy_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get GL energy time series (both integration methods).

        Returns:
            Tuple (time, energy_voronoi, energy_triangles).
        """
        time = self.get_time_series('time')[0]
        _, e_vor = self.get_time_series('energy_voronoi')
        _, e_tri = self.get_time_series('energy_triangles')
        return time, e_vor, e_tri

    def get_conservation_series(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get current conservation time series.

        Returns:
            Dictionary with keys: 'flux_edges', 'flux_sites', 'div_voronoi', 'div_triangles'.
            Each value is a tuple (time, values).
        """
        result = {}
        for key in ['flux_edges', 'flux_sites', 'div_voronoi', 'div_triangles']:
            if key in self._time_series_keys:
                result[key] = self.get_time_series(key)
        return result

    def get_poisson_residual_series(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Poisson residual time series.

        Returns:
            Tuple (time, residual_max, residual_mean).
        """
        time = self.get_time_series('time')[0]
        _, res_max = self.get_time_series('poisson_residual_max')
        _, res_mean = self.get_time_series('poisson_residual_mean')
        return time, res_max, res_mean

    def get_currents_series(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get current diagnostics time series.

        Returns:
            Dictionary with keys: 'max_normal_current', 'min_supercurrent_mag',
            'mean_Jn_boundary', 'max_Jn_boundary'.
        """
        result = {}
        for key in ['max_normal_current', 'min_supercurrent_mag', 'mean_Jn_boundary', 'max_Jn_boundary']:
            if key in self._time_series_keys:
                result[key] = self.get_time_series(key)
        return result

    def get_probe_data(self) -> Dict[str, np.ndarray]:
        """
        Get probe point data (mu and phase at specified locations).

        Returns:
            Dictionary with keys:
                - 'time': time array
                - 'mu_probe': shape (n_times, n_probes)
                - 'phase_probe': shape (n_times, n_probes), phase in units of π
                - 'coords': shape (n_probes, 2), probe point coordinates [x, y]
                - 'indices': shape (n_probes,), mesh site indices
        """
        if 'mu_probe' not in self._time_series_keys:
            raise KeyError("No probe data found. PhysicalTracker was not enabled.")

        with h5py.File(self.path, 'r') as f:
            ts = f['time_series']
            result = {
                'time': np.array(ts['time']),
                'mu_probe': np.array(ts['mu_probe']),
                'phase_probe': np.array(ts['phase_probe']),
            }
            if 'probe_points_coords' in ts:
                result['coords'] = np.array(ts['probe_points_coords'])
            if 'probe_points_indices' in ts:
                result['indices'] = np.array(ts['probe_points_indices'])

        return result

    def get_solver_diagnostics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get solver diagnostic time series (dt and poisson_tolerance).

        Returns:
            Tuple (time, dt_array, poisson_tolerance_array).
        """
        time = self.get_time_series('time')[0]
        _, dt = self.get_time_series('dt')
        _, tol = self.get_time_series('poisson_tolerance')
        return time, dt, tol

    # =========================================================================
    # Utility methods
    # =========================================================================

    def __repr__(self) -> str:
        n_sites = len(self.sites) if self.sites is not None else 0
        return (f"Solution({self.path.name}, "
                f"steps={self.n_saved_steps}, "
                f"sites={n_sites})")