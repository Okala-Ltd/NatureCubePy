from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from naturecubepy import old_viz as vt


def _build_station_rows(site: str, n: int, lon: float, lat: float) -> list[dict[str, float | str]]:
    return [{"site": site, "lon": lon, "lat": lat} for _ in range(n)]


def test_plot_station_data_uses_bounded_sizes_and_rounded_legend(monkeypatch):
    monkeypatch.setattr(vt, "_add_satellite_basemap", lambda *args, **kwargs: True)

    stations = pd.DataFrame(
        _build_station_rows("A", 3, 12.10, -0.10)
        + _build_station_rows("B", 37, 12.15, -0.15)
        + _build_station_rows("C", 1789, 12.20, -0.20)
    )

    fig = vt.plot_station_data(
        stations,
        location_col="site",
        lon_col="lon",
        lat_col="lat",
        show_legend=True,
        size_scale=9.0,
    )

    try:
        ax = fig.axes[0]
        sizes = np.concatenate(
            [
                collection.get_sizes()
                for collection in ax.collections
                if hasattr(collection, "get_sizes") and collection.get_sizes().size > 0
            ]
        )
        assert sizes.max() <= 680.0

        legend = ax.get_legend()
        assert legend is not None
        labels = [int(text.get_text()) for text in legend.get_texts()]
        assert labels == sorted(set(labels))
        assert all(label > 0 for label in labels)
        assert all(label < 10 or (label % (10 ** (len(str(label)) - 1)) == 0) for label in labels)
    finally:
        plt.close(fig)


def test_plot_edna_iucn_by_class_species_expands_y_axis_to_fit_bars():
    rows: list[dict[str, str]] = []
    for i in range(8):
        rows.append({"class_": "Mammalia", "species": f"Mammalia species {i}", "iucn_redlist_status": "Least Concern"})
    for i in range(5):
        rows.append({"class_": "Aves", "species": f"Aves species {i}", "iucn_redlist_status": "Endangered"})

    df = pd.DataFrame(rows)

    fig = vt.plot_edna_iucn_by_class_species(
        df,
        class_col="class_",
        species_col="species",
        iucn_col="iucn_redlist_status",
        y_max=5,
        y_tick_step=5,
    )

    try:
        ax = fig.axes[0]
        y_top = ax.get_ylim()[1]
        assert y_top >= 10.0
    finally:
        plt.close(fig)
