import numpy as np
import time
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

warnings.filterwarnings('ignore')

# ============================================================================
# 0. КОНФИГУРАЦИЯ
# ============================================================================
CORES_FOR_TDGL = 4

os.environ['OMP_NUM_THREADS'] = str(CORES_FOR_TDGL)
os.environ['MKL_NUM_THREADS'] = str(CORES_FOR_TDGL)
os.environ['OPENBLAS_NUM_THREADS'] = str(CORES_FOR_TDGL)

# ============================================================================
# 1. ИМПОРТЫ
# ============================================================================
try:
    from numba import njit, prange, set_num_threads, get_num_threads

    HAS_NUMBA = True
    set_num_threads(CORES_FOR_TDGL)
except ImportError:
    HAS_NUMBA = False

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ============================================================================
# 2. ПАРАМЕТРЫ
# ============================================================================
N = 39000
D = 15
K = 60
DTYPE = np.float32
ITER_COUNTS = [10, 50, 100]


# ============================================================================
# 3. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================================
def generate_test_data(N, D, K, dtype=DTYPE):
    np.random.seed(42)
    G = np.random.randn(N, D, K).astype(dtype)
    rhs = np.random.randn(N, K).astype(dtype)
    return G, rhs


# ============================================================================
# 4. МЕТОДЫ (ТОЛЬКО ЛУЧШИЕ)
# ============================================================================

def method_einsum_baseline(G, rhs):
    return np.einsum('nji,ni->nj', G, rhs, optimize='optimal')


# --- NUMBA PARALLEL ---
if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_parallel(G, rhs):
        N, D, K = G.shape
        result = np.empty((N, D), dtype=G.dtype)
        for n in prange(N):
            for d in range(D):
                s = 0.0
                for k in range(K):
                    s += G[n, d, k] * rhs[n, k]
                result[n, d] = s
        return result


    def method_numba_parallel(G, rhs):
        return _numba_parallel(G, rhs)


# --- THREADING ---
def _threading_worker(args):
    G_chunk, rhs_chunk = args
    return np.einsum('nji,ni->nj', G_chunk, rhs_chunk, optimize='optimal')


def method_threading(G, rhs, n_threads=None):
    if n_threads is None:
        n_threads = CORES_FOR_TDGL

    chunk_size = (G.shape[0] + n_threads - 1) // n_threads
    chunks = [(G[i:i + chunk_size], rhs[i:i + chunk_size])
              for i in range(0, G.shape[0], chunk_size)]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(_threading_worker, chunks))

    return np.vstack(results)


# --- JAX ---
if HAS_JAX:
    @jit
    def _jax_einsum(G, rhs):
        return jnp.einsum('nji,ni->nj', G, rhs)


    def method_jax(G, rhs):
        G_jax = jnp.asarray(G)
        rhs_jax = jnp.asarray(rhs)
        result = _jax_einsum(G_jax, rhs_jax)
        return np.asarray(result)

# --- PYTORCH ---
if HAS_TORCH:
    def method_pytorch(G, rhs):
        G_t = torch.from_numpy(G)
        rhs_t = torch.from_numpy(rhs)
        result = torch.matmul(G_t, rhs_t.unsqueeze(-1)).squeeze(-1)
        return result.numpy()


# ============================================================================
# 5. БЕНЧМАРК
# ============================================================================
def benchmark_method(func, G, rhs, name, iter_counts=ITER_COUNTS):
    results = {}

    for n_iter in iter_counts:
        # Warmup
        _ = func(G, rhs)

        # Измерение
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = func(G, rhs)
        t1 = time.perf_counter()

        total_time = t1 - t0
        avg_time = total_time / n_iter

        results[n_iter] = {
            'total_ms': total_time * 1000,
            'avg_ms': avg_time * 1000,
        }

    return results


def check_correctness(reference, result, name, tol=1e-10):
    diff = np.max(np.abs(reference - result))
    status = "✅" if diff < tol else "❌"
    print(f"   {status} {name}: max diff = {diff:.2e}")


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("БЕНЧМАРК: einsum('nji,ni->nj', G, rhs) — только лучшие методы")
    print("=" * 80)
    print(f"💻 Всего ядер: {cpu_count()}")
    print(f"🎯 Используем: {CORES_FOR_TDGL}")
    print(f"🔒 Оставлено: {cpu_count() - CORES_FOR_TDGL}")

    if HAS_NUMBA:
        print(f"✅ Numba: {get_num_threads()} потоков")
    if HAS_JAX:
        print(f"✅ JAX доступен")
    if HAS_TORCH:
        print(f"✅ PyTorch доступен")

    print(f"\n📊 Данные: N={N}, D={D}, K={K}")
    print(f"   Размер G: {N * D * K * 8 / 1024 / 1024:.1f} MB")
    print(f"   Размер rhs: {N * K * 8 / 1024 / 1024:.1f} MB")

    G, rhs = generate_test_data(N, D, K)

    # Собираем методы
    methods = [
        (method_einsum_baseline, "1. np.einsum (baseline)"),
    ]

    if HAS_NUMBA:
        methods.append((method_numba_parallel, "2. Numba parallel (prange)"))

    methods.append((lambda G, rhs: method_threading(G, rhs, CORES_FOR_TDGL),
                    f"3. Threading ({CORES_FOR_TDGL})"))

    if HAS_JAX:
        methods.append((method_jax, "4. JAX (JIT)"))

    if HAS_TORCH:
        methods.append((method_pytorch, "5. PyTorch (matmul)"))

    # Прогрев
    print("\n⏳ Прогрев (10 вызовов)...")
    for func, name in methods:
        try:
            for _ in range(10):
                _ = func(G, rhs)
            print(f"   ✓ {name}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")

    # Бенчмарк
    print(f"\n{'=' * 80}")
    print(f"ТЕСТ: {ITER_COUNTS} итераций")
    print("=" * 80)

    all_results = {}
    reference = None

    for func, name in methods:
        print(f"\n🔹 {name}")
        try:
            res = benchmark_method(func, G, rhs, name, ITER_COUNTS)
            all_results[name] = res

            result = func(G, rhs)
            if reference is None:
                reference = result
            else:
                check_correctness(reference, result, name)

            for n_iter in ITER_COUNTS:
                total = res[n_iter]['total_ms']
                avg = res[n_iter]['avg_ms']
                print(f"   {n_iter:>3d} итер: всего {total:>8.1f} ms, среднее {avg:>6.2f} ms/ит")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    # Итоговая таблица
    print("\n" + "=" * 100)
    print("ИТОГОВАЯ ТАБЛИЦА: Среднее время на 1 итерацию (ms)")
    print("=" * 100)

    header = f"{'Метод':<35s}"
    for n_iter in ITER_COUNTS:
        col_name = f"{n_iter} итер (avg)"
        header += f" | {col_name:<20s}"
    header += f" | {'Speedup (100)':<12s}"
    print(header)
    print("-" * 100)

    baseline_100 = all_results[list(all_results.keys())[0]][100]['avg_ms']

    for name, res in all_results.items():
        line = f"{name:<35s}"
        for n_iter in ITER_COUNTS:
            line += f" | {res[n_iter]['avg_ms']:>16.2f} ms    "
        speedup = baseline_100 / res[100]['avg_ms']
        line += f" | {speedup:>10.2f}x    "
        print(line)

    print("=" * 100)

    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    best_name = max(all_results.keys(),
                    key=lambda n: baseline_100 / all_results[n][100]['avg_ms'])
    best_speedup = baseline_100 / all_results[best_name][100]['avg_ms']
    print(f"   Лучший метод: {best_name} ({best_speedup:.2f}x)")

if __name__ == "__main__":
    main()