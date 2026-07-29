"""
numba_kernels.py — вычислительные ядра для LSFD
"""
import numpy as np

try:
    from numba import njit, prange, set_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@njit(parallel=True, cache=True)
def _numba_parallel(G, rhs):
    """Параллельное умножение: derivatives[n,d] = Σ_k G[n,d,k] * rhs[n,k]"""
    N, D, K = G.shape
    result = np.empty((N, D), dtype=rhs.dtype)
    for n in prange(N):
        for d in range(D):
            s = 0.0
            for k in range(K):
                s += G[n, d, k] * rhs[n, k]
            result[n, d] = s
    return result


def _einsum_wrapper(G, rhs):
    """Обычный einsum (fallback)"""
    return np.einsum('nji,ni->nj', G, rhs, optimize='optimal')


def get_batched_dot(use_numba=False, num_threads=4):
    """
    Выбрать функцию для пакетного умножения.

    Args:
        use_numba: False (по умолчанию) — использовать einsum.
                   True — использовать Numba (ошибка если не установлена).
        num_threads: число потоков для Numba (по умолчанию 4)

    Returns:
        (callable, str) — функция и название метода
    """
    if use_numba:
        if not HAS_NUMBA:
            raise ImportError(
                "use_numba=True, but numba isn't installed. "
                "Install: pip install numba"
            )
        set_num_threads(num_threads)
        return _numba_parallel, f"Numba parallel ({num_threads} threads)"
    else:
        return _einsum_wrapper, "np.einsum"