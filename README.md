# Hogweed segmentation

Проект финального эксперимента по бинарной сегментации крупных очагов борщевика Сосновского на многоканальных растрах Sentinel-2.

## Состав

- `config.py` — пути и параметры эксперимента.
- `src/` — подготовка данных, разбиение, датасет, модель, обучение, метрики, полноразмерный прогноз и визуализации.
- `scripts/train_model.py` — обучение итоговой модели.
- `scripts/evaluate_model.py` — подбор порога и расчёт метрик на validation/test.
- `scripts/full_raster_report.py` — полноразмерный прогноз, объектные и буферизованные метрики, итоговые таблицы и рисунки.
- `scripts/make_rgb_dates_figure.py` — RGB-композиты выбранных дат.
- `notebooks/hogweed_unetpp_stratified_large_focus_tversky_executed.ipynb` — исходный исполненный ноутбук с сохранёнными output’ами.

## Запуск

```bash
pip install -r requirements.txt
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/full_raster_report.py
```

Перед запуском при необходимости меняются пути в `config.py`.

## Выходы

Таблицы, JSON-сводки, `.npy`-массивы и рисунки записываются в `outputs/`.
