"""TDGL in a uniform perpendicular magnetic field with all three trackers.

Requires only the installed tdgl_LSFD_lib package and its dependencies.
The device and mesh are generated from scratch; no input HDF5 is read.
Parameters follow simulation_vers02.py, starting from psi=1 and mu=0.

    python examples/quickstart_field.py --smoke
    python examples/quickstart_field.py --no-show --no-animations

--smoke runs 0.001 tau0 with all trackers and PNG/JSON output, without GIFs.
Use --solve-time to change the duration and --output-dir for result files.
Short runs may have no vortices or finite velocity estimates yet.
"""
import numpy as np
import argparse
import json
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt
from tdgl_LSFD_lib.device.device import Device
from tdgl_LSFD_lib.device.geometry import circle
from tdgl_LSFD_lib.device.polygon import Polygon
from tdgl_LSFD_lib.external_fields.external_fields import ExternalFields
from tdgl_LSFD_lib.operators.operators import LSFD_operators
from tdgl_LSFD_lib.solver.dynamics_options import SolverOptions, TimeScheme
from tdgl_LSFD_lib.solver.solve import solve
from tdgl_LSFD_lib.post_processing.animation import make_video_from_solution
from tdgl_LSFD_lib.post_processing.plot_solution import (
    plot_conservation_summary,
    plot_probe_difference_graph,
    plot_summary,
    plot_vortex_tracker_graph,
    get_vortex_summary,
)


# Dimensionless parameters from simulation_vers02.py (magnetic field only).
R_val = 15
h_val = 0.5
Bz_amplitude = 0.1
I_boundary = 0.0
weight_function = "1r"
time_scheme = TimeScheme.ADAPTIVE_EULER
solve_time = 100
dt_init = 1e-4
K_val = 150
ghost_coeff = 1
vortex_max_match_distance = 1.0  # xi between consecutive steps; new IDs on larger jumps.
vortex_velocity_window = 0.10  # Trailing quadratic-regression window in simulation time.
vortex_velocity_min_points = 5

eta_val = 0.0
gamma_val = 0.1
h_direction = (0, -1)

HERE = Path(__file__).resolve().parent


def get_unique_filename(output_dir: Path, R, h, Bz, I_boundary, scheme, eta, gamma, h_dir, weight, time_val) -> str:
    scheme_str = scheme.value if hasattr(scheme, "value") else str(scheme).lower()
    h_dir_str = f"{h_dir[0]}{h_dir[1]}"
    weight_str = f"w{weight}"
    base_name = (
        f"R{R:.0f}_h{h:.2f}_Bz{Bz:.3f}_I_boundary{I_boundary:.2f}_{scheme_str}_"
        f"eta{eta:.1f}_gamma{gamma:.1f}_hdir{h_dir_str}_{weight_str}_"
        f"t{time_val:g}_field"
    )

    sim_id = 0
    while True:
        filename = f"{base_name}_sim{sim_id}.h5"
        if not (output_dir / filename).exists():
            return filename
        sim_id += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 0.001 tau0, headless, without GIFs")
    parser.add_argument("--solve-time", type=float, help="Override simulation duration")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot windows")
    parser.add_argument("--no-animations", action="store_true", help="Skip GIF generation")
    args = parser.parse_args(argv)
    # Smoke mode keeps the same geometry, mesh and physical parameters.
    run_duration = args.solve_time if args.solve_time is not None else (0.001 if args.smoke else solve_time)
    if run_duration <= 0 or not np.isfinite(run_duration):
        parser.error("--solve-time must be finite and positive")
    if args.no_show or args.smoke:
        matplotlib.use("Agg")
    OUTPUT_DIR = args.output_dir or (HERE / "test_output/quickstart_field_smoke"
                                     if args.smoke else HERE / "simulation_results/quickstart_field")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE_NAME = get_unique_filename(
        OUTPUT_DIR,
        R_val,
        h_val,
        Bz_amplitude,
        I_boundary,
        time_scheme,
        eta_val,
        gamma_val,
        h_direction,
        weight_function,
        run_duration,
    )
    OUTPUT_FILE = OUTPUT_DIR / OUTPUT_FILE_NAME
    BASE_NAME_NO_EXT = OUTPUT_FILE_NAME.replace(".h5", "")

    print(f"Output file: {OUTPUT_FILE_NAME}")

    # Always construct the device: no cached mesh or previous solution is needed.
    n_bound = int(20 * np.pi * R_val / h_val)
    film = Polygon(
        name="film", points=circle(radius=R_val, center=(0, 0)),
    ).resample(n_bound).buffer(0)
    # Horizontal pair (0, 1), vertical pair (2, 3); coordinates in xi.
    probes = np.array([
        [-R_val / 2, 0], [R_val / 2, 0],
        [0, -R_val / 2], [0, R_val / 2],
    ], dtype=np.float64)
    device = Device(name="field_disk", film=film, probe_points=probes)
    device.make_mesh(
        max_edge_length=h_val, n_lsfd_neighbors=K_val,
        smooth=100, graded_smooth=True, lloyd_step=0.3,
        ghost_coeff=ghost_coeff, layers=3,
    )
    device.save(str(OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_device.h5"),
                save_mesh=True, compress=True)
    device.plot(mesh=True)
    plt.savefig(OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_mesh.png", dpi=150)
    if args.no_show or args.smoke:
        plt.close("all")

    Ext = ExternalFields(
        Bz_amplitude=Bz_amplitude,
        I_amplitude_source=I_boundary,
        eta=eta_val,
        h_direction=h_direction,
        gamma=gamma_val,
    )

    LSFD = LSFD_operators(
        mesh=device.mesh,
        s_direction=Ext.s_constant,
        use_numba=True,
        num_threads=4,
        weight_function=weight_function,
        use_ghost_points=False,
        use_mirror_points=True,
        ghost_version="sym",
        check_condition_number=False,
    )

    options = SolverOptions(
        solve_time=run_duration,
        dt_init=dt_init,
        time_scheme=time_scheme,
        adaptive_window=10,
        save_every=3 if args.smoke else 100,
        run_mode="cluster" if args.smoke else "pc",
        poisson_adaptive=True,
        poisson_tolerance_init=1e-4,
        output_file=str(OUTPUT_FILE),
        track_conservation=True,
        track_physical=True,
        track_vortices=True,
        vortex_max_match_distance=vortex_max_match_distance,
        vortex_velocity_window=vortex_velocity_window,
        vortex_velocity_min_points=vortex_velocity_min_points,
        log_file=str(OUTPUT_DIR / f"{BASE_NAME_NO_EXT}.log"),
        log_level="INFO",
    )

    print(f"Starting simulation: {device.mesh.n_sites} sites, {run_duration} tau0")

    solution = solve(
        device=device,
        operators=LSFD,
        external_fields=Ext,
        options=options,
        psi_init=None,
        mu_init=None,
        seed_solution=None,  # Start from psi=1, mu=0; no seed file required.
        reset_clock=True,
    )

    print(f"Simulation finished. Solution: {solution.path}")

    summary_spatial_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_spatial_summary.png"
    fig1 = plot_summary(solution, step=-1, save_path=str(summary_spatial_path))
    print(f"Spatial summary: {summary_spatial_path.name}")

    summary_cons_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_conservation_summary.png"
    try:
        fig2 = plot_conservation_summary(solution, save_path=str(summary_cons_path))
        print(f"Conservation summary: {summary_cons_path.name}")
    except ValueError as e:
        print(f"Skip conservation summary: {e}")

    vortex_graph_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_vortex_tracker_graph.png"
    fig3 = plot_vortex_tracker_graph(
        solution,
        save_path=str(vortex_graph_path),
    )
    print(f"Vortex tracker graph: {vortex_graph_path.name}")
    (OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_vortex_summary.json").write_text(
        json.dumps(get_vortex_summary(solution), indent=2), encoding="utf-8")

    probe_diff_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_probe_difference_graph.png"
    fig4 = plot_probe_difference_graph(
        solution,
        horizontal_pair=(0, 1),
        vertical_pair=(2, 3),
        save_path=str(probe_diff_path),
    )
    print(f"Probe difference graph: {probe_diff_path.name}")

    if args.smoke or args.no_animations:
        if not (args.no_show or args.smoke):
            plt.show()
        plt.close("all")
        return solution

    print("Creating animations...")

    # GIFs use saved spatial frames; tracker data are saved at every time step.
    anim_psi_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_psi.gif"
    make_video_from_solution(
        solution=solution,
        fig_name=f"{BASE_NAME_NO_EXT}_psi",
        file_dir=str(OUTPUT_DIR),
        quantities="order_parameter",
        fps=15,
        figsize=(8, 6),
    )
    print(f"Order-parameter animation: {anim_psi_path.name}")

    anim_full_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_full_summary.gif"
    make_video_from_solution(
        solution=solution,
        fig_name=f"{BASE_NAME_NO_EXT}_full_summary",
        file_dir=str(OUTPUT_DIR),
        quantities=[
            "order_parameter", "phase", "mu",
            "div_Js", "supercurrent"
        ],
        fps=10,
        figsize=(14, 10),
    )
    print(f"Full summary animation: {anim_full_path.name}")

    if not args.no_show:
        plt.show()
    plt.close("all")
    return solution


if __name__ == "__main__":
    main()
