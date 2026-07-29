import numpy as np
import time
from numba import njit, prange, set_num_threads


# ============================================================================
# ФУНКЦИИ (Без time и print внутри @njit!)
# ============================================================================

def measure_memory_bandwidth():
    """Измеряет пропускную способность памяти"""
    size = 100000000  # 800 MB
    a = np.random.rand(size)
    b = np.random.rand(size)
    c = np.empty_like(a)

    t0 = time.perf_counter()
    np.add(a, b, out=c)  # In-place
    t1 = time.perf_counter()

    total_bytes = size * 8 * 3  # чтение a, b + запись c
    bandwidth = total_bytes / (t1 - t0) / 1e9
    print("💾 Пропускная способность памяти (v2): {:.2f} GB/s".format(bandwidth))
    return bandwidth


def compute_intensity(G, rhs):
    """Вычисляет интенсивность вычислений (FLOPs/байт)"""
    N, D, K = G.shape
    bytes_read = N * D * K * 8 + N * K * 8
    bytes_write = N * D * 8
    flops = N * D * K * 2

    intensity = flops / (bytes_read + bytes_write)
    print("📊 Интенсивность вычислений: {:.2f} FLOPs/байт".format(intensity))
    print("   Чтение: {:.1f} MB".format(bytes_read / 1024 / 1024))
    print("   Запись: {:.1f} MB".format(bytes_write / 1024 / 1024))
    print("   FLOPs: {:.1f} M".format(flops / 1e6))
    return intensity


@njit(parallel=True, fastmath=True, cache=True)
def run_parallel_benchmark(G, rhs):
    """Чисто вычислительная функция для Numba"""
    N, D, K = G.shape
    result = np.empty((N, D), dtype=G.dtype)

    for n in prange(N):
        for d in range(D):
            s = 0.0
            for k in range(K):
                s += G[n, d, k] * rhs[n, k]
            result[n, d] = s

    return result


# ============================================================================
# ЗАПУСК И ЗАМЕРЫ ВРЕМЕНИ (Здесь используем time и print)
# ============================================================================
print("=" * 80)
print("ДИАГНОСТИКА ПРОИЗВОДИТЕЛЬНОСТИ")
print("=" * 80)

N, D, K = 39000, 15, 60
G = np.random.randn(N, D, K)
rhs = np.random.randn(N, K)

measure_memory_bandwidth()
compute_intensity(G, rhs)

print("\n--- Тест с разным числом потоков ---")
for n_threads in [1, 2, 4, 8]:
    print("\n{} поток(а):".format(n_threads))
    set_num_threads(n_threads)

    # 1. Прогрев (обязательно для Numba, чтобы не замерять время JIT-компиляции)
    _ = run_parallel_benchmark(G, rhs)

    # 2. Замер времени
    t0 = time.perf_counter()
    result = run_parallel_benchmark(G, rhs)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    gflops = N * D * K * 2 / (t1 - t0) / 1e9

    print("🔧 Использовано потоков: {}".format(n_threads))
    print("⏱️  Время: {:.2f} ms".format(elapsed_ms))
    print("📈 Производительность: {:.2f} GFLOPs".format(gflops))