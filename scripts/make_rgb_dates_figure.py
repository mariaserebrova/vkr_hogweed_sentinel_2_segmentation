import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.raster_io import discover_final_rasters
from src.visualization import plot_selected_dates_grid


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    date_rasters = discover_final_rasters(config.PREPARED_ROOT, config.RUN_ONLY_DATES)
    dates_to_show = ["0512", "0718", "0904", "0907", "0909"]
    date_labels = {
        "0512": "12.05.2025",
        "0718": "18.07.2025",
        "0904": "04.09.2024",
        "0907": "07.09.2024",
        "0909": "09.09.2024",
    }

    out_path = config.FIGURE_DIR / "selected_dates_rgb.png"
    fig = plot_selected_dates_grid(
        date_rasters=date_rasters,
        dates_to_show=dates_to_show,
        date_labels=date_labels,
        out_path=out_path,
        out_max_size=900,
        n_cols=5,
    )
    fig.clf()
    print(out_path)


if __name__ == "__main__":
    main()
