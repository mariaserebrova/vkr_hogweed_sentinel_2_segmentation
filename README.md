# Поиск крупных очагов борщевика Сосновского по снимкам Sentinel-2

В репозитории собран код итогового эксперимента для выпускной квалификационной работы, посвящённой автоматическому выявлению крупных очагов борщевика Сосновского на многоспектральных спутниковых снимках Sentinel-2.

Задача решается как бинарная семантическая сегментация. На вход модели подаются многоканальные фрагменты спутникового растра, на выходе строится карта вероятности принадлежности пикселей классу «борщевик Сосновского».

Итоговая модель — U-Net++ с энкодером ResNet34.  
В финальной конфигурации используются 10 спектральных каналов Sentinel-2:

`B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11`.

---

## Структура репозитория

```text
hogweed_segmentation_project/
│
├── config.py
├── requirements.txt
│
├── src/
│   ├── reproducibility.py
│   ├── raster_io.py
│   ├── windows.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   ├── training.py
│   ├── inference.py
│   ├── visualization.py
│   └── pipeline.py
│
├── scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── full_raster_report.py
│   └── make_rgb_dates_figure.py
│
├── notebooks/
│   └── hogweed_unetpp_stratified_large_focus_tversky_executed.ipynb
│
└── outputs/
    ├── figures/
    ├── tables/
    └── checkpoints/
