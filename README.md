# TDGL LSFD Solver

Решатель уравнений Time-Dependent Ginzburg-Landau (TDGL) с использованием метода наименьших квадратов конечных разностей (LSFD — Least Squares Finite Difference).

## Описание

Библиотека для численного моделирования динамики сверхпроводящих вихрей в тонких плёнках. Поддерживает:
- Генерацию неструктурированных треугольных сеток (через MeshPy)
- Вычисление производных высокого порядка методом LSFD
- Адаптивный шаг по времени
- Адаптивный tolerance для солвера Пуассона
- Сохранение результатов в HDF5 с возможностью продолжения симуляции
- Визуализацию устройств, сеток и результатов

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/tdgl_LSFD_dev.git
cd tdgl_LSFD_dev

# Установка в режиме разработки
pip install -e .
```

## Быстрый старт

```python
from tdgl_LSFD_lib import solve, Device, SolverOptions
from tdgl_LSFD_lib.device.polygon import Polygon

# Создание устройства
film = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
device = Device(name="square_film", film=film)
device.make_mesh(max_edge_length=0.5)

# Параметры симуляции
options = SolverOptions(
    solve_time=150.0,
    run_mode="pc",
    save_every=100,
    output_file="results/simulation.h5",
)

# Запуск симуляции
solution = solve(device=device, operators=operators, 
                 external_fields=fields, options=options)
```

## Структура проекта

```
tdgl_LSFD_dev/
├── tdgl_LSFD_lib/
│   ├── device/          # Геометрия устройства и генерация сетки
│   ├── mesh/            # Неструктурированные сетки (TriMesh, DualMesh)
│   ├── operators/       # LSFD операторы и FVM интегратор
│   ├── external_fields/ # Внешние поля (B, J, ферромагнетик)
│   ├── solver/          # TDGL солвер, runner, options
│   └── post_processing/ # Загрузка и анализ результатов
├── pyproject.toml
├── README.md
└── .gitignore
```

## Зависимости

- `numpy`, `scipy` — численные вычисления
- `h5py` — сохранение результатов в HDF5
- `tqdm` — прогресс-бар
- `shapely` — работа с геометрией
- `matplotlib` — визуализация
- `meshpy` — генерация треугольных сеток
- `numba` (опционально) — ускорение вычислений
- `opt_einsum` (опционально) — оптимизация тензорных операций

## Лицензия

MIT