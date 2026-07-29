import numpy as np
import time
import os
import warnings
from multiprocessing import cpu_count
import psutil

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
from numba import njit, prange, set_num_threads, get_num_threads

set_num_threads(CORES_FOR_TDGL)

# Попытка импорта opt_einsum
try:
    import opt_einsum as oe

    HAS_OPT_EINSUM = True
    print("✅ opt_einsum доступен")
except ImportError:
    HAS_OPT_EINSUM = False
    print("⚠️ opt_einsum не установлен: conda install -c conda-forge opt_einsum")

# ============================================================================
# 2. ПАРАМЕТРЫ
# ============================================================================
N = 39000
D = 15
K = 100
ITER_COUNTS = [10, 50, 100]


# ============================================================================
# 3. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================================
def generate_test_data(N, D, K, dtype):
    np.random.seed(42)
    G = np.random.randn(N, D, K).astype(dtype)
    rhs = np.random.randn(N, K).astype(dtype)
    return G, rhs


# ============================================================================
# 4. МЕТОДЫ
# ============================================================================

def method_einsum_baseline(G, rhs):
    """Обычный np.einsum с optimize='optimal'"""
    return np.einsum('nji,ni->nj', G, rhs, optimize='optimal')


def method_opt_einsum(G, rhs):
    """opt_einsum с автоматической оптимизацией"""
    if not HAS_OPT_EINSUM:
        return None
    return oe.contract('nji,ni->nj', G, rhs, optimize='optimal')


def method_opt_einsum_greedy(G, rhs):
    """opt_einsum с greedy оптимизацией (быстрее для простых случаев)"""
    if not HAS_OPT_EINSUM:
        return None
    return oe.contract('nji,ni->nj', G, rhs, optimize='greedy')


@njit(parallel=True, fastmath=False, cache=True)
def _numba_parallel(G, rhs):
    """Numba parallel: prange по сайтам"""
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


def check_correctness(reference, result, name, tol=1e-6):
    diff = np.max(np.abs(reference - result))
    status = "✅" if diff < tol else "❌"
    print("   {} {}: max diff = {:.2e}".format(status, name, diff))


# ============================================================================
# 6. ЗАПУСК ТЕСТА ДЛЯ ОДНОГО DTYPE
# ============================================================================
def run_benchmark_for_dtype(dtype_name, dtype):
    print("\n" + "=" * 100)
    print("ТЕСТ ДЛЯ DTYPE: {} ({})".format(dtype_name, dtype))
    print("=" * 100)

    G, rhs = generate_test_data(N, D, K, dtype)
    bytes_per_elem = np.dtype(dtype).itemsize

    print("📊 Данные: N={}, D={}, K={}".format(N, D, K))
    print("   Размер G: {:.1f} MB".format(N * D * K * bytes_per_elem / 1024 / 1024))
    print("   Размер rhs: {:.1f} MB".format(N * K * bytes_per_elem / 1024 / 1024))

    methods = [
        (method_einsum_baseline, "1. np.einsum (baseline)"),
    ]

    if HAS_OPT_EINSUM:
        methods.extend([
            (method_opt_einsum, "2. opt_einsum (optimal)"),
            (method_opt_einsum_greedy, "3. opt_einsum (greedy)"),
        ])

    methods.append((method_numba_parallel, "4. Numba parallel (prange)"))

    # Прогрев
    print("\n⏳ Прогрев (10 вызовов)...")
    for func, name in methods:
        try:
            for _ in range(10):
                _ = func(G, rhs)
            print("   ✓ {}".format(name))
        except Exception as e:
            print("   ✗ {}: {}".format(name, e))

    # Бенчмарк
    print("\n" + "=" * 80)
    print("ТЕСТ: {} итераций".format(ITER_COUNTS))
    print("=" * 80)

    all_results = {}
    reference = None

    for func, name in methods:
        print("\n🔹 {}".format(name))
        try:
            res = benchmark_method(func, G, rhs, name, ITER_COUNTS)
            all_results[name] = res

            result = func(G, rhs)
            if reference is None:
                reference = result
            else:
                tol = 1e-6 if dtype == np.float32 else 1e-10
                check_correctness(reference, result, name, tol=tol)

            for n_iter in ITER_COUNTS:
                total = res[n_iter]['total_ms']
                avg = res[n_iter]['avg_ms']
                print("   {:>3d} итер: всего {:>8.1f} ms, среднее {:>6.2f} ms/ит".format(
                    n_iter, total, avg))
        except Exception as e:
            print("   ❌ Ошибка: {}".format(e))

    # Итоговая таблица
    print("\n" + "=" * 100)
    print("ИТОГОВАЯ ТАБЛИЦА ({}): Среднее время на 1 итерацию (ms)".format(dtype_name))
    print("=" * 100)

    header = "{:<35s}".format("Метод")
    for n_iter in ITER_COUNTS:
        col_name = "{} итер (avg)".format(n_iter)
        header += " | {:<20s}".format(col_name)
    header += " | {:<12s}".format("Speedup (100)")
    print(header)
    print("-" * 100)

    baseline_100 = all_results[list(all_results.keys())[0]][100]['avg_ms']

    for name, res in all_results.items():
        line = "{:<35s}".format(name)
        for n_iter in ITER_COUNTS:
            line += " | {:>16.2f} ms    ".format(res[n_iter]['avg_ms'])
        speedup = baseline_100 / res[100]['avg_ms']
        line += " | {:>10.2f}x    ".format(speedup)
        print(line)
    print("=" * 100)

    # Рекомендации
    best_name = max(all_results.keys(),
                    key=lambda n: baseline_100 / all_results[n][100]['avg_ms'])
    best_speedup = baseline_100 / all_results[best_name][100]['avg_ms']
    print("\n💡 Лучший метод: {} ({:.2f}x)".format(best_name, best_speedup))

    return all_results


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    print("=" * 100)
    print("БЕНЧМАРК: einsum('nji,ni->nj', G, rhs)")
    print("=" * 100)
    print("💻 Всего ядер: {}".format(cpu_count()))
    print("🎯 Используем: {}".format(CORES_FOR_TDGL))
    print("🔒 Оставлено: {}".format(cpu_count() - CORES_FOR_TDGL))
    print("✅ Numba: {} потоков".format(get_num_threads()))

    # Тест 1: float64
    results_f64 = run_benchmark_for_dtype("float64", np.float64)

    # Тест 2: float32
    results_f32 = run_benchmark_for_dtype("float32", np.float32)

    # Финальное сравнение
    print("\n" + "=" * 100)
    print("СРАВНЕНИЕ FLOAT64 vs FLOAT32 (100 итераций, среднее ms/ит)")
    print("=" * 100)
    print("{:<35s} | {:<15s} | {:<15s} | {:<12s}".format(
        "Метод", "float64", "float32", "F32 speedup"))
    print("-" * 100)

    for name in results_f64.keys():
        t64 = results_f64[name][100]['avg_ms']
        t32 = results_f32[name][100]['avg_ms']
        speedup = t64 / t32
        print("{:<35s} | {:>10.2f} ms  | {:>10.2f} ms  | {:>8.2f}x".format(
            name, t64, t32, speedup))
    print("=" * 100)


if __name__ == "__main__":
    main()