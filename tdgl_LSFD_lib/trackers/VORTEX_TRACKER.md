# Динамический VortexTracker

В библиотеку перенесён алгоритм из `scratchers/votrex_scratcher_v2.py`.
Старая реализация сохранена без изменения в `vortex_tracker_archive.py`.
Детекция, matching и вычисление скорости новой версии совпадают с песочницей.

```python
options = SolverOptions(
    solve_time=100,
    track_vortices=True,
    save_every=100,
    vortex_max_match_distance=1.0,
    vortex_velocity_window=0.05,
    vortex_velocity_min_points=5,
)
```

Обычный TDGLRunner автоматически создаёт новый трекер. Число вихрей не ограничено
числом слотов. Параметр `max_vortices` оставлен для запуска старых скриптов,
но игнорируется. None для `vortex_max_match_distance` означает 1 ξ.
Порог — перемещение между соседними шагами; регрессия использует истинное время
солвера, включая адаптивный dt. Для оценки скорости в окне нужно минимум
`vortex_velocity_min_points` точек, иначе сохраняется NaN.

## Данные и жизненный цикл

У каждого вихря неповторяющийся ID, начиная с 0. Заряд фиксируется при рождении.
Сопоставление допускается только с тем же зарядом и расстоянием до порога.
Исчезнувший ID закрывается на первом шаге отсутствия, следующий вихрь получает
новый ID. Исчезновение запрашивает общую выгрузку Runner и всех трекеров.
Периодические выгрузки и остаток в конце работают по прежней схеме.
История живых вихрей не очищается при выгрузке; история погибшего удаляется.

HDF5 содержит общие числовые таблицы, без отдельной группы на каждый вихрь:

| Dataset в `/time_series/vortices` | Содержание |
|---|---|
| `stat_info` | `vortex_id, charge, birth`: одна запись при рождении |
| `dyn_info` | `step, time, vortex_id, x, y, vx, vy, triangle_index`: отсчёты живых ID |
| `deaths` | `vortex_id, death`: одна запись при исчезновении |
| `frame_offsets` | Границы dyn_info для каждого шага, включая пустые |

Runner сначала дописывает `/time_series/time`, `step`, `dt`, затем вызывает
`flush_to_hdf5`. В новой библиотечной версии deaths поступают из buffer_deaths;
восстановление событий по координатам, применявшееся в песочнице, не требуется.

## Графики и summary

```python
from tdgl_LSFD_lib.post_processing.solution import Solution
from tdgl_LSFD_lib.post_processing.plot_solution import (
    plot_vortex_tracker_graph, extract_vortex_trajectories, get_vortex_summary,
)

solution = Solution("solution.h5")
fig = plot_vortex_tracker_graph(solution, save_path="vortices.png")
# По умолчанию печатается короткая сводка по всему моделированию.
summary = get_vortex_summary(solution)
trajectories = extract_vortex_trajectories(solution)
t = trajectories[0]["time"]
x = trajectories[0]["x"]
# Для отдельных ID: vortex_ids=[0, 3]. Отключить печать: print_summary=False.
```

Четыре панели показывают x(t), y(t), vx(t), vy(t) всех наблюдавшихся ID, включая
завершённые траектории. Точки разных ID не соединяются. Вихрь с одним отсчётом
виден благодаря маркеру. NaN до накопления истории не заменяется нулём.

Сводка содержит:

- `final_positions`: ID, заряд и координаты центров фазовых нулей только тех
  вихрей, которые существуют на последнем шаге. Это координаты ядер из трекера,
  а не отдельный поиск узлового минимума |psi|.
- `center_of_mass`: среднее этих координат с равными весами, независимо от заряда.
  Если последний шаг пуст, значение None; старые координаты не подставляются.
- `lifetimes`: death-birth для каждого завершённого ID; `mean_lifetime` — их
  среднее. Живые в конце вихри не имеют измеренного death и в среднее не входят.
  Если завершённых нет, значение None, что явно указано в текстовой сводке.
- `min_speed`, `max_speed`: экстремумы hypot(vx, vy) среди конечных оценок по
  всему моделированию, с ID, временем и шагом. При отсутствии оценок — None.
- `total_vortices`: число уникальных ID, родившихся за весь расчёт.

`Solution.get_vortex_data()` возвращает исходные таблицы новой схемы. Старые
файлы с vortex_x/y/vx/vy по-прежнему читаются и строятся. Для них точное число
физических траекторий и времена жизни восстановить из переиспользуемых слотов
нельзя, поэтому эти величины не выдумываются и отображаются как недоступные.

## Проверка

```powershell
python -W error -m unittest discover -s tests -p 'test*.py' -v
python -W error -m unittest discover -s scratchers -p 'test*.py' -v
python -m examples.validate_dynamic_vortex_tracker
```

Последняя команда создаёт демонстрационные HDF5, PNG и summary.json в
`test_output/vortex_tracker_v2_integration`. Это проверка на синтетическом
движении ядер, найденных в исходном поле, без полного решения уравнений TDGL.
