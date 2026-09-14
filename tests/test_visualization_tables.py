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


def test_station_map_legend_layout_follows_aspect():
    assert vt._station_map_legend_layout(1000, 1000) == "side"
    assert vt._station_map_legend_layout(1000, 2000) == "side"
    assert vt._station_map_legend_layout(2000, 1000) == "bottom"
    assert vt._station_map_legend_layout(1550, 1000) == "bottom"


def test_station_map_puts_legend_below_wide_extent(monkeypatch):
    monkeypatch.setattr(vt, "_add_satellite_basemap", lambda *args, **kwargs: True)
    monkeypatch.setattr(vt, "_load_okala_logo", lambda *args, **kwargs: None)

    # ~2:1 east-west span → bottom legend strip.
    stations = pd.DataFrame(
        _build_station_rows("A", 3, 12.00, -0.10)
        + _build_station_rows("B", 37, 12.20, -0.10)
        + _build_station_rows("C", 50, 12.10, -0.12)
    )

    fig = vt.station_map(stations)
    try:
        assert len(fig.axes) == 2
        map_pos = fig.axes[0].get_position()
        leg_pos = fig.axes[1].get_position()
        assert leg_pos.y1 <= map_pos.y0 + 0.05
        assert leg_pos.width > map_pos.width * 0.6
        # Footer should sit close under the map (no tall letterbox gap).
        assert map_pos.y0 - leg_pos.y1 < 0.12
    finally:
        plt.close(fig)


def test_station_map_keeps_side_legend_for_compact_extent(monkeypatch):
    monkeypatch.setattr(vt, "_add_satellite_basemap", lambda *args, **kwargs: True)
    monkeypatch.setattr(vt, "_load_okala_logo", lambda *args, **kwargs: None)

    # Compact / taller extent → right-hand sidebar.
    stations = pd.DataFrame(
        _build_station_rows("A", 3, 12.10, -0.10)
        + _build_station_rows("B", 37, 12.12, -0.20)
        + _build_station_rows("C", 50, 12.11, -0.15)
    )

    fig = vt.station_map(stations)
    try:
        assert len(fig.axes) == 2
        map_pos = fig.axes[0].get_position()
        leg_pos = fig.axes[1].get_position()
        assert leg_pos.x0 >= map_pos.x1 - 0.05
        assert leg_pos.height > map_pos.height * 0.5
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


def test_edna_class_rank_summary_uses_taxonomy_not_bare_label():
    """Genus/family/order labels must not be counted as species-level IDs."""
    df = pd.DataFrame(
        [
            {
                "class_": "Aves",
                "order": "Passeriformes",
                "family": "Turdidae",
                "genus": "Turdus",
                "species": "Turdus merula",
                "label": "Turdus merula",
            },
            {
                "class_": "Aves",
                "order": "Passeriformes",
                "family": "Turdidae",
                "genus": "Turdus",
                "species": None,
                "label": "Turdus",
            },
            {
                "class_": "Actinopterygii",
                "order": "Cypriniformes",
                "family": "Cyprinidae",
                "genus": None,
                "species": None,
                "label": "Cyprinidae",
            },
            {
                "class_": "Aves",
                "order": "Passeriformes",
                "family": None,
                "genus": None,
                "species": None,
                "label": "Passeriformes",
            },
            {
                "class_": "Mammalia",
                "order": None,
                "family": None,
                "genus": None,
                "species": None,
                "label": "Mammalia",
            },
            {
                # Label-only binomial (no species column filled) still counts as species.
                "class_": "Mammalia",
                "order": "Carnivora",
                "family": "Mustelidae",
                "genus": "Martes",
                "species": None,
                "label": "Martes foina",
            },
        ]
    )

    summary = vt._edna_class_rank_summary(df)
    counts = {
        (row.resolved_class, row.resolved_rank): int(row.unique_taxa)
        for row in summary.itertuples(index=False)
    }
    assert counts[("Aves", "Species")] == 1
    assert counts[("Aves", "Genus")] == 1
    assert counts[("Aves", "Order")] == 1
    assert counts[("Actinopterygii", "Family")] == 1
    assert counts[("Mammalia", "Order")] == 1
    assert counts[("Mammalia", "Species")] == 1


def test_records_species_by_monitoring_type_keeps_species_detections_only():
    frames = {
        "camera": pd.DataFrame(
            [
                {"class": "Mammalia", "species": "Sus scrofa"},
                {"class": "Mammalia", "species": "Sus scrofa"},
                {"class": "Mammalia", "species": ""},  # higher taxon only
                {"class": "Unknown", "species": "Mystery sp"},
            ]
        ),
        "bioacoustic": pd.DataFrame(
            [
                {"class": "Aves", "species": "Turdus merula"},
                {"class": "Unknown", "species": ""},
            ]
        ),
    }
    counts = {
        key: vt._class_record_species_counts(df, species_detections_only=True)
        for key, df in frames.items()
    }
    cam = counts["camera"].set_index("class")
    bio = counts["bioacoustic"].set_index("class")
    assert list(cam.index) == ["Mammalia"]
    assert int(cam.loc["Mammalia", "number_of_records"]) == 2
    assert int(cam.loc["Mammalia", "number_of_species"]) == 1
    assert list(bio.index) == ["Aves"]
    assert int(bio.loc["Aves", "number_of_records"]) == 1
    assert "Unknown" not in set(cam.index) | set(bio.index)

