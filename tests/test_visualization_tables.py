from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

import naturecubepy.viz as vt


def _build_station_rows(
    site: str,
    n: int,
    lon: float,
    lat: float,
    *,
    measurement_type: str = "camera",
) -> list[dict[str, float | str | int]]:
    return [
        {
            "site": site,
            "lon": lon,
            "lat": lat,
            "measurement_type": measurement_type,
            "record_count": n,
        }
    ]


def test_station_map_uses_bounded_sizes(monkeypatch):
    monkeypatch.setattr(vt, "_add_satellite_basemap", lambda *args, **kwargs: True)
    monkeypatch.setattr(vt, "_load_okala_logo", lambda *args, **kwargs: None)

    stations = pd.DataFrame(
        _build_station_rows("A", 3, 12.10, -0.10)
        + _build_station_rows("B", 37, 12.15, -0.15)
        + _build_station_rows("C", 1789, 12.20, -0.20)
    )

    fig = vt.station_map(stations)

    try:
        ax = fig.axes[0]
        sizes = np.concatenate(
            [
                collection.get_sizes()
                for collection in ax.collections
                if hasattr(collection, "get_sizes") and collection.get_sizes().size > 0
            ]
        )
        assert sizes.max() <= 400.0
        assert sizes.min() >= 50.0
    finally:
        plt.close(fig)


def test_station_map_uses_light_brand_tints_by_sensor(monkeypatch):
    monkeypatch.setattr(vt, "_add_satellite_basemap", lambda *args, **kwargs: True)
    monkeypatch.setattr(vt, "_load_okala_logo", lambda *args, **kwargs: None)

    stations = pd.DataFrame(
        _build_station_rows("A", 3, 12.10, -0.10, measurement_type="camera")
        + _build_station_rows("B", 4, 12.15, -0.15, measurement_type="bioacoustic")
        + _build_station_rows("C", 5, 12.20, -0.20, measurement_type="eDNA")
    )

    fig = vt.station_map(stations)

    try:
        marker_colors = {
            tuple(collection.get_facecolors()[0])
            for collection in fig.axes[0].collections
            if collection.get_facecolors().size
        }
        expected = {
            to_rgba(vt.SENSOR_COLORS["Camera"], alpha=0.9),
            to_rgba(vt.SENSOR_COLORS["Bioacoustic"], alpha=0.9),
            to_rgba(vt.SENSOR_COLORS["eDNA"], alpha=0.9),
        }
        assert marker_colors == expected
    finally:
        plt.close(fig)


def test_iucn_bar_plot_counts_unique_species_by_class():
    rows: list[dict[str, str]] = []
    for i in range(8):
        rows.append({"class": "Mammalia", "species": f"Mammalia species {i}", "iucn_status": "Least Concern"})
    for i in range(5):
        rows.append({"class": "Aves", "species": f"Aves species {i}", "iucn_status": "Endangered"})

    df = pd.DataFrame(rows)

    fig, summary = vt.iucn_bar_plot(df, return_summary=True)

    try:
        assert set(summary["class"]) == {"Mammalia", "Aves"}
        mammalia = summary.loc[summary["class"] == "Mammalia"].iloc[0]
        aves = summary.loc[summary["class"] == "Aves"].iloc[0]
        assert int(mammalia.get("Least Concern", 0)) == 8
        assert int(aves.get("Endangered", 0)) == 5

        ax = fig.axes[0]
        assert ax.get_ylim()[1] >= 8.0
    finally:
        plt.close(fig)
