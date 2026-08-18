import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt

from tdgl_LSFD_lib.device.device import Device
from tdgl_LSFD_lib.device.geometry import circle
from tdgl_LSFD_lib.device.polygon import Polygon
from tdgl_LSFD_lib.external_fields.external_fields import ExternalFields
from tdgl_LSFD_lib.operators.operators import LSFD_operators
from tdgl_LSFD_lib.solver.dynamics_options import SolverOptions, TimeScheme
from tdgl_LSFD_lib.solver.solve import solve
from tdgl_LSFD_lib.post_processing.plot_solution import plot_summary, plot_conservation_summary
from tdgl_LSFD_lib.post_processing.animation import make_video_from_solution
from tdgl_LSFD_lib.mesh.mesh import Mesh
import h5py

def get_unique_filename(output_dir: Path, R, h, Bz, scheme, eta, h_dir, weight, time_val) -> str:
    """
    Генерирует уникальное имя файла, инкрементируя sim_id, если файл уже существует.
    """
    scheme_str = scheme.value if hasattr(scheme, 'value') else str(scheme).lower()
    h_dir_str = f"{h_dir[0]}{h_dir[1]}"
    weight_str = f"w{weight}"
    base_name = f"R{R:.0f}_h{h:.2f}_Bz{Bz:.3f}_{scheme_str}_eta{eta:.1f}_hdir{h_dir_str}_{weight_str}_t{time_val:.0f}"

    sim_id = 0
    while True:
        filename = f"{base_name}_sim{sim_id}.h5"
        if not (output_dir / filename).exists():
            return filename
        sim_id += 1

# ============================================================================
# 0. Input data
# ============================================================================
R_val = 15.0 # radius of film in units xi
h_val = 0.5 # max distance between sites in units xi
Bz_amplitude = 0.02 # uniform perpendicular magnetic field in units B_c2
weight_function = '1r' # weight function for LSFD matrix
time_scheme = TimeScheme.ADAPTIVE_EULER
solve_time = 100 # solution time
dt_init = 1e-4
K_val = 150 # neighbors amount for LSFD matrix
eta_val = 0.0 # linear diode coefficient
gamma_val = 0.1 # cubic diode coefficient
h_direction = (0, -1) # exchange field direction of ferromagnetic substate

# ============================================================================
# 1. Use seed data
# ============================================================================
# To start new simulation use "___.h5" or use previous results
SEED_FILE = "___.h5"  # Example: "simulation_results/R15_h0.25_Bz0.08_..._sim0.h5"

OUTPUT_DIR = Path("simulation_results") # directory for results
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 2. Generate file name
# ============================================================================
OUTPUT_FILE_NAME = get_unique_filename(
    OUTPUT_DIR, R_val, h_val, Bz_amplitude, time_scheme, eta_val, h_direction, weight_function, solve_time
)
OUTPUT_FILE = OUTPUT_DIR / OUTPUT_FILE_NAME
BASE_NAME_NO_EXT = OUTPUT_FILE_NAME.replace(".h5", "")

# ============================================================================
# 3. Geometry and mesh
# ============================================================================
n_bound = int(20 * np.pi * R_val / h_val) # amount of boundary points
film = Polygon(name="film", points=circle(radius=R_val, center=(0, 0))).resample(n_bound).buffer(0) # film generation
OUT = "device_mesh.h5" # name for saving device structure

# # # === 1. Download existed device with mesh ===
# device = Device.load(OUT, load_mesh=False)
# with h5py.File(OUT, "r") as f:
#     device.mesh = Mesh.from_hdf5(f["mesh"], n_lsfd_neighbors=K_val, layers=3)
# # # === 1. Download ===

# # === 2. Generate emesh ===
device = Device(name="test", film=film)
device.make_mesh(
    max_edge_length=h_val,
    n_lsfd_neighbors=K_val,
    smooth=100,
    graded_smooth=True,
    lloyd_step=0.3, # lloyd smooth for mesh
    layers = 3  # ← amount of layers for mirror points
)
device.save(OUT, save_mesh=True, compress=True)

# f_old, _ = device.plot(mesh=True) # plot mesh
# plt.show()

# ============================================================================
# 4. External fields and LSFD matrix
# ============================================================================

Ext = ExternalFields(Bz_amplitude=Bz_amplitude, eta=eta_val, h_direction=h_direction, gamma =  gamma_val)
LSFD = LSFD_operators(
    mesh=device.mesh,
    s_direction=Ext.s_constant,
    use_numba=True,
    num_threads=4,  # for parallel calculations
    weight_function=weight_function,
    use_ghost_points=False,
    use_mirror_points=True,
)

# ============================================================================
# 5. Simulation
# ============================================================================
options = SolverOptions(
    solve_time=solve_time,
    dt_init=dt_init,
    time_scheme=time_scheme,
    adaptive_window=10,
    save_every=100,
    poisson_adaptive=True,
    poisson_tolerance_init=1e-4,
    output_file=str(OUTPUT_FILE),
    track_conservation=True, # conservation tracker for energy and flux
    # === LOGGING ===
    log_file=str(OUTPUT_DIR / f"{BASE_NAME_NO_EXT}.log"),
    log_level="INFO"
)

seed_path = None  # download seed
if SEED_FILE != "____":
    seed_path_obj = OUTPUT_DIR / SEED_FILE if not Path(SEED_FILE).is_absolute() else Path(SEED_FILE)
    if seed_path_obj.exists():
        seed_path = str(seed_path_obj)
        print(f"\n🔄 Seed file: {seed_path_obj.name}")
    else:
        print(f"\n⚠️  Seed file doesn't exist: {seed_path_obj}")
        print("  Start with initial psi=1, mu=0")

print(f"\n🚀 Simulation starts: {device.mesh.n_sites} sites, {options.solve_time} τ₀")

solution = solve(
    device=device,
    operators=LSFD,
    external_fields=Ext,
    options=options,
    psi_init=None,
    mu_init=None,
    seed_solution=seed_path,
    reset_clock=True,
)

print(f"\n✅ Simulation is finished. Solution: {solution.path}")

# ============================================================================
# 8. Plot pictures
# ============================================================================
print("\n📊 Plot pictures...")

# 8.1. Spatial data (order parameter, currents, electical potential) from last step
summary_spatial_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_spatial_summary.png"
fig1 = plot_summary(solution, step=-1, save_path=str(summary_spatial_path))
print(f"   ✅ Spatial summary: {summary_spatial_path.name}")

# 8.2. Conservation tracker summary (energy. flux, poisson residual)
summary_cons_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_conservation_summary.png"
try:
    fig2 = plot_conservation_summary(solution, save_path=str(summary_cons_path))
    print(f"   ✅ Conservation summary: {summary_cons_path.name}")
except ValueError as e:
    print(f"   ⚠️ No Conservation summary: {e}")

# ============================================================================
# 9. Animation
# ============================================================================
anim_psi_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_psi.gif"
make_video_from_solution(
    solution=solution,
    fig_name=f"{BASE_NAME_NO_EXT}_psi",
    file_dir=str(OUTPUT_DIR),
    quantities="order_parameter", # animation for order parameter magnitude
    fps=15,
    figsize=(8, 6)
)

anim_full_path = OUTPUT_DIR / f"{BASE_NAME_NO_EXT}_full_summary.gif"
make_video_from_solution(
    solution=solution,
    fig_name=f"{BASE_NAME_NO_EXT}_full_summary",
    file_dir=str(OUTPUT_DIR),
    quantities=[
        "order_parameter", "phase", "mu",
        "div_Js", "supercurrent", "normal_current" # animation for all spatial data
    ],
    fps=10,
    figsize=(14, 10)
)
print(f"   ✅ Full animation: {anim_full_path.name}")


print("\n✨ Simulation is finished!")
