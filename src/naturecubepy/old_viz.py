"""Visualization and summary-table helpers for NatureCubePy tutorials."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import folium
import geopandas as gpd
import httpx
import matplotlib.pyplot as plt
import numpy as np


from naturecubepy.analysis import ObservationBundle
from naturecubepy.api import (
    build_iucn_map,
    enrich_with_iucn_status,
    get_audio_observation_data,
    get_camera_trap_data,
    get_edna_observation_data,
    get_project_labels,
    get_station_info,
    normalise_species_frame,
)
from naturecubepy.schema import SPECIES_OBS_CORE_COLUMNS

CONCERN_STATUSES = {
    "Critically Endangered",
    "Endangered",
    "Vulnerable",
    "Near Threatened",
}

PLOT_COLORS = ["#3E5859", "#D9ACDE", "#F6F6E2", "#9DEECF"]


def _normalise_sensor_label(value: object) -> str:
    mt = str(value).strip().lower()
    if mt == "camera":
        return "Camera"
    if mt in {"bioacoustic", "audio"}:
        return "Bioacoustic"
    if mt == "edna":
        return "eDNA"
    return "Unknown"


def _species_series(df: pd.DataFrame) -> pd.Series:
    if "species" in df.columns:
        return df["species"].fillna("").astype(str).str.strip()
    if "label" in df.columns:
        return df["label"].fillna("").astype(str).str.strip()
    return pd.Series(dtype="object")


def _taxonomic_class_series(df: pd.DataFrame) -> pd.Series:
    for col in ["class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
        if col in df.columns:
            out = df[col].fillna("Unknown").astype(str).str.strip()
            return out.mask(out == "", "Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in [
        "timestamp",
        "observation_timestamp",
        "segment_start_timestamp",
        "segment_timestamp",
        "media_file_timestamp",
        "project_system_record_start_timestamp",
    ]:
        if col in df.columns:
            return col
    return None


def _ensure_iucn_column(df: pd.DataFrame) -> pd.DataFrame:
    if "iucn_redlist_status" in df.columns:
        return df
    out = df.copy()
    out["iucn_redlist_status"] = pd.NA
    return out


def _empty_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "device_id",
            "measurement_type",
            "data_type",
            "common_name",
            "species",
            "label",
            "class",
            "taxonomic_class",
            "timestamp",
            "observation_timestamp",
            "segment_start_timestamp",
            "project_system_record_start_timestamp",
            "project_system_record_end_timestamp",
            "iucn_redlist_status",
        ]
    )


def _empty_stations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "device_id",
            "measurement_type",
            "data_type",
            "project_system_record_start_timestamp",
            "project_system_record_end_timestamp",
            "record_count",
            "latitude",
            "longitude",
        ]
    )

def plot_stations(geojson_response: gpd.GeoDataFrame) -> Any:
    """Plot station locations on an interactive Folium map.

    Circle markers are sized proportionally to the number of media records
    at each station.

    Parameters
    ----------
    geojson_response:
        A GeoDataFrame as returned by :func:`get_station_info`.

    Returns
    -------
    folium.Map
        An interactive map widget.

    Examples
    --------
    >>> m = plot_stations(stations)  # doctest: +SKIP
    """
    import folium

    print("Plotting stations")

    if geojson_response.crs is not None and geojson_response.crs.is_geographic:
        centroids = geojson_response.to_crs(epsg=3857).geometry.centroid.to_crs(geojson_response.crs)
    else:
        centroids = geojson_response.geometry.centroid

    m = folium.Map(
        location=[centroids.y.mean(), centroids.x.mean()],
        zoom_start=10,
        tiles="Esri WorldImagery",
    )

    record_counts = geojson_response.get("record_count", pd.Series([1] * len(geojson_response)))
    min_count = record_counts.min() if not record_counts.empty else 1
    max_count = record_counts.max() if not record_counts.empty else 1
    count_range = max_count - min_count

    def _rescale(value: float, new_min: float = 5.0, new_max: float = 15.0) -> float:
        if count_range == 0:
            return (new_min + new_max) / 2
        return new_min + (value - min_count) / count_range * (new_max - new_min)

    for idx, row in geojson_response.iterrows():
        popup_html = (
            f"Device ID: {row.get('device_id', '')}<br>"
            f"Measurement type: {row.get('measurement_type', '')}<br>"
            f"Data type: {row.get('data_type', '')}<br>"
            f"Start time: {row.get('project_system_record_start_timestamp', '')}<br>"
            f"End time: {row.get('project_system_record_end_timestamp', '')}<br>"
            f"No. media files: {row.get('record_count', 1)}<br>"
        )
        folium.CircleMarker(
            location=[centroids.loc[idx].y, centroids.loc[idx].x],
            radius=_rescale(row.get("record_count", 1)),
            tooltip=str(row.get("device_id", "")),
            popup=folium.Popup(popup_html, max_width=300),
            color="red",
            fill=True,
            fill_opacity=0.6,
            opacity=0.2,
        ).add_to(m)

    return m





def species_per_class_table(camera_df: pd.DataFrame, bio_df: pd.DataFrame) -> pd.DataFrame:
    """Create a table of unique species counts per taxonomic class and sensor type."""
    cam = pd.DataFrame(
        {
            "class": _taxonomic_class_series(camera_df),
            "species": _species_series(camera_df),
        }
    )
    cam = cam[cam["species"] != ""]
    cam["sensor_type"] = "Camera"

    bio = pd.DataFrame(
        {
            "class": _taxonomic_class_series(bio_df),
            "species": _species_series(bio_df),
        }
    )
    bio = bio[bio["species"] != ""]
    bio["sensor_type"] = "Bioacoustic"

    merged = pd.concat([cam, bio], ignore_index=True)
    out = (
        merged.drop_duplicates(["sensor_type", "class", "species"])
        .groupby(["sensor_type", "class"], as_index=False)
        .agg(number_of_species=("species", "count"))
        .sort_values(["sensor_type", "number_of_species"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out


def _class_record_species_counts(df: pd.DataFrame) -> pd.DataFrame:
    classes = _taxonomic_class_series(df)
    species = _species_series(df)
    work = pd.DataFrame({"class": classes, "species": species})

    records = work.groupby("class", as_index=False).size().rename(columns={"size": "number_of_records"})

    species_only = work[work["species"] != ""]
    if species_only.empty:
        records["number_of_species"] = 0
        return records.sort_values("number_of_records", ascending=False).reset_index(drop=True)

    species_counts = (
        species_only.drop_duplicates(["class", "species"])
        .groupby("class", as_index=False)
        .size()
        .rename(columns={"size": "number_of_species"})
    )
    out = records.merge(species_counts, on="class", how="left")
    out["number_of_species"] = out["number_of_species"].fillna(0).astype(int)
    return out.sort_values("number_of_records", ascending=False).reset_index(drop=True)


def _ordered_class_levels(camera_df: pd.DataFrame, bio_df: pd.DataFrame) -> list[str]:
    classes = pd.concat([
        _taxonomic_class_series(camera_df),
        _taxonomic_class_series(bio_df),
    ], ignore_index=True)
    classes = classes.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    unique_classes = classes.unique().tolist()

    def _rank(name: str) -> tuple[int, str]:
        lower = str(name).strip().lower()
        if any(token in lower for token in ["mammalia", "mammal"]):
            return (0, lower)
        if any(token in lower for token in ["aves", "bird", "avian"]):
            return (1, lower)
        return (2, lower)

    return sorted(unique_classes, key=_rank)


def _class_color_map(class_levels: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab20")
    colors: dict[str, object] = {}
    other_idx = 0

    for name in class_levels:
        lower = str(name).strip().lower()
        if any(token in lower for token in ["mammalia", "mammal"]):
            colors[name] = "#3E5859"
            continue
        if any(token in lower for token in ["aves", "bird", "avian"]):
            colors[name] = "#D9ACDE"
            continue

        # Skip first two tab20 colors so "other" classes don't collide with fixed mammal/aves colors.
        colors[name] = cmap((other_idx + 2) % 20)
        other_idx += 1

    return colors


def _style_class_axis(
    ax,
    *,
    values: np.ndarray,
    y_label: str,
    panel_letter: str,
    font_family: str = "Arial",
    y_tick_step: int = 0,
    x_label_rotation: float = 45.0,
) -> None:
    """Apply plot_edna_records-style axis formatting to a class bar chart axis."""
    def _nice_tick_step(max_value: float, n_ticks_target: int = 6) -> int:
        """Return a 1-2-5*10^k y-axis step close to max_value / n_ticks_target."""
        if max_value <= 0:
            return 1
        raw = max_value / max(1, n_ticks_target)
        exponent = int(math.floor(math.log10(raw)))
        base = 10 ** exponent
        scaled = raw / base
        if scaled <= 1:
            nice = 1
        elif scaled <= 2:
            nice = 2
        elif scaled <= 5:
            nice = 5
        else:
            nice = 10
        return int(nice * base)

    max_val = float(np.max(values)) if len(values) > 0 else 0.0
    step = y_tick_step if y_tick_step > 0 else _nice_tick_step(max_val)
    y_max = int(np.ceil(max_val / step) * step) if max_val > 0 else step
    y_lower_pad = max(0.8, step * 0.04)
    ax.set_ylim(-y_lower_pad, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))
    ax.spines["left"].set_bounds(0, y_max)

    ax.set_ylabel(y_label, fontfamily=font_family)
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", length=0, rotation=x_label_rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
        lbl.set_fontfamily(font_family)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(font_family)

    ax.text(
        -0.12, 1.06, panel_letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=14, fontweight="bold",
        fontfamily=font_family,
        clip_on=False,
    )


def _plot_species_per_class_for_sensor(
    df: pd.DataFrame,
    *,
    class_levels: list[str],
    class_colors: dict[str, object],
    font_family: str = "Arial",
    figsize: tuple[float, float] = (10.0, 3.8),
    x_label_rotation: float = 45.0,
):
    metrics = _class_record_species_counts(df)
    metrics = metrics.set_index("class").reindex(class_levels, fill_value=0).reset_index()
    metrics = metrics[(metrics["number_of_records"] > 0) | (metrics["number_of_species"] > 0)].reset_index(drop=True)

    if metrics.empty:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No class data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        return fig

    bar_colors = [class_colors[c] for c in metrics["class"].tolist()]

    fig, axes = plt.subplots(ncols=2, figsize=figsize, facecolor="white")
    fig.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")

    axes[0].bar(
        metrics["class"],
        metrics["number_of_records"],
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        clip_on=False,
    )
    _style_class_axis(
        axes[0],
        values=metrics["number_of_records"].to_numpy(dtype=float),
        y_label="Number of records",
        panel_letter="A",
        font_family=font_family,
        x_label_rotation=x_label_rotation,
    )

    axes[1].bar(
        metrics["class"],
        metrics["number_of_species"],
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        clip_on=False,
    )
    _style_class_axis(
        axes[1],
        values=metrics["number_of_species"].to_numpy(dtype=float),
        y_label="Number of species",
        panel_letter="B",
        font_family=font_family,
        x_label_rotation=x_label_rotation,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.94, wspace=0.32)
    return fig


def plot_species_per_class(camera_df: pd.DataFrame, bio_df: pd.DataFrame):
    """Plot A/B class histograms for each sensor type in separate figures.

    A = number of records per class, B = number of species per class.
    Returns a dictionary with ``camera`` and ``bioacoustic`` figure objects.
    """
    class_levels = _ordered_class_levels(camera_df, bio_df)
    class_colors = _class_color_map(class_levels)

    return {
        "camera": _plot_species_per_class_for_sensor(
            camera_df,
            class_levels=class_levels,
            class_colors=class_colors,
        ),
        "bioacoustic": _plot_species_per_class_for_sensor(
            bio_df,
            class_levels=class_levels,
            class_colors=class_colors,
        ),
    }


def _top_species_counts(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    species = _species_series(df)
    species = species[species != ""]
    counts = species.value_counts().head(top_n).sort_values(ascending=True)
    return counts.rename_axis("species").reset_index(name="observations")


def _display_species_series(df: pd.DataFrame) -> pd.Series:
    if "common_name" in df.columns:
        names = df["common_name"].fillna("").astype(str).str.strip()
        if (names != "").any():
            return names
    return _species_series(df)


def _apply_minimal_axes_style(ax, *, grid_axis: str | None = "y"):
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.25)
        ax.set_axisbelow(True)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


def plot_top_species_side_by_side(camera_df: pd.DataFrame, bio_df: pd.DataFrame, top_n: int = 15):
    """Plot top species by observation count with A/B side-by-side panels."""
    cam = _top_species_counts(camera_df, top_n=top_n)
    bio = _top_species_counts(bio_df, top_n=top_n)

    fig, axes = plt.subplots(ncols=2, figsize=(16, 8), sharex=False, sharey=False)

    axes[0].barh(cam["species"], cam["observations"], color=PLOT_COLORS[0])
    axes[0].text(
        -0.10,
        1.04,
        "A",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    axes[0].set_xlabel("Number of observations")
    axes[0].set_ylabel("Species")

    axes[1].barh(bio["species"], bio["observations"], color=PLOT_COLORS[1])
    axes[1].text(
        -0.10,
        1.04,
        "B",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    axes[1].set_xlabel("Number of observations")
    axes[1].set_ylabel("Species")

    for ax in axes:
        _apply_minimal_axes_style(ax, grid_axis="x")

    plt.tight_layout()
    return fig


def plot_top_species_mammal_bird_records(
    observations_df: pd.DataFrame,
    *,
    top_n: int = 10,
    class_col: str | None = None,
    species_col: str | None = None,
    figsize: tuple[float, float] = (14.0, 6.0),
    x_label_rotation: float = 45.0,
    font_family: str = "Arial",
):
    """Plot top mammal and bird species by number of records in A/B panels.

    Parameters
    ----------
    observations_df:
        Observation table containing class and species/common-name columns.
    top_n:
        Number of species to display per panel.
    class_col:
        Optional class column name. If omitted, resolves from class/class_.
    species_col:
        Optional species label column name. If omitted, resolves from
        common_name/species.
    figsize:
        Figure size in inches.
    x_label_rotation:
        Rotation angle for x-axis labels in degrees.
    font_family:
        Font family used to match NatureCubePy figure styling.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, pd.Series]]
        Figure and top-count series for mammals and birds.
    """
    df = observations_df.copy()

    if class_col is None:
        class_col = "class" if "class" in df.columns else "class_"
    if species_col is None:
        species_col = "common_name" if "common_name" in df.columns else "species"

    if class_col not in df.columns:
        raise KeyError("No class column found. Expected 'class' or 'class_'.")
    if species_col not in df.columns:
        raise KeyError("No species column found. Expected 'common_name' or 'species'.")

    df[class_col] = df[class_col].fillna("").astype(str).str.strip()
    df[species_col] = df[species_col].fillna("").astype(str).str.strip()

    mammal_mask = df[class_col].str.lower().isin(["mammalia", "mammal", "mammals"])
    bird_mask = df[class_col].str.lower().isin(["aves", "bird", "birds"])

    mammal_top = df.loc[mammal_mask & (df[species_col] != ""), species_col].value_counts().head(top_n)
    bird_top = df.loc[bird_mask & (df[species_col] != ""), species_col].value_counts().head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    def _plot_panel(ax, counts: pd.Series, *, color: str, edgecolor: str, panel_letter: str) -> None:
        if counts.empty:
            ax.text(
                0.5,
                0.5,
                "No records",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontfamily=font_family,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            return

        x_positions = np.arange(len(counts))
        ax.bar(x_positions, counts.values, color=color, edgecolor=edgecolor, linewidth=0.6, clip_on=False)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(counts.index)
        ax.set_xlim(-0.5, len(counts) - 0.5)

        _style_class_axis(
            ax,
            values=counts.to_numpy(dtype=float),
            y_label="Number of records",
            panel_letter=panel_letter,
            font_family=font_family,
            x_label_rotation=x_label_rotation,
        )

    _plot_panel(
        axes[0],
        mammal_top,
        color="#214f50",
        edgecolor="#4d4d4d",
        panel_letter="A",
    )
    _plot_panel(
        axes[1],
        bird_top,
        color="#cc8ac7",
        edgecolor="#8a7a89",
        panel_letter="B",
    )

    for ax in axes:
        ax.grid(False)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.94, wspace=0.24)
    return fig, {"mammals": mammal_top, "birds": bird_top}


def station_summary_table(stations_df: pd.DataFrame) -> pd.DataFrame:
    """Build the requested station summary table by sensor type."""
    if stations_df.empty:
        return pd.DataFrame(
            columns=[
                "Sensor Type",
                "Number of Sensors",
                "Number of Sensor Days",
                "Number of Unique Locations",
                "Number of Records",
                "Date Coverage",
            ]
        )

    df = stations_df.copy()
    df["Sensor Type"] = df.get("measurement_type", "Unknown").map(_normalise_sensor_label)

    start_col = "project_system_record_start_timestamp"
    end_col = "project_system_record_end_timestamp"
    if start_col in df.columns and end_col in df.columns:
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
        sensor_days = (df[end_col] - df[start_col]).dt.days.fillna(0).clip(lower=0)
    else:
        sensor_days = pd.Series([0] * len(df), index=df.index, dtype="int64")

    if "record_count" in df.columns:
        records = pd.to_numeric(df["record_count"], errors="coerce").fillna(0)
    else:
        records = pd.Series([0] * len(df), index=df.index, dtype="int64")

    lat = pd.to_numeric(df.get("latitude"), errors="coerce") if "latitude" in df.columns else None
    lon = pd.to_numeric(df.get("longitude"), errors="coerce") if "longitude" in df.columns else None
    if lat is None or lon is None:
        unique_loc = pd.Series([0] * len(df), index=df.index, dtype="int64")
    else:
        unique_loc = pd.Series(list(zip(lat.round(5), lon.round(5))), index=df.index)

    df = df.assign(_sensor_days=sensor_days, _records=records, _loc=unique_loc)

    summary = (
        df.groupby("Sensor Type", as_index=False)
        .agg(
            **{
                "Number of Sensors": ("device_id", "nunique"),
                "Number of Sensor Days": ("_sensor_days", "sum"),
                "Number of Unique Locations": ("_loc", "nunique"),
                "Number of Records": ("_records", "sum"),
                "_start": (start_col, "min") if start_col in df.columns else ("Sensor Type", "first"),
                "_end": (end_col, "max") if end_col in df.columns else ("Sensor Type", "first"),
            }
        )
        .sort_values("Sensor Type")
        .reset_index(drop=True)
    )

    if start_col in df.columns and end_col in df.columns:
        summary["Date Coverage"] = (
            summary["_start"].dt.date.astype(str) + " to " + summary["_end"].dt.date.astype(str)
        )
    else:
        summary["Date Coverage"] = "Not available"

    summary = summary.drop(columns=["_start", "_end"])
    return summary[
        [
            "Sensor Type",
            "Number of Sensors",
            "Number of Sensor Days",
            "Number of Unique Locations",
            "Number of Records",
            "Date Coverage",
        ]
    ]


def redlist_status_table(all_species_df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-species redlist table."""
    if all_species_df.empty:
        return pd.DataFrame(columns=["species", "common_name", "iucn_redlist_status", "sensor_types"])

    df = all_species_df.copy()
    if "species" not in df.columns:
        return pd.DataFrame(columns=["species", "common_name", "iucn_redlist_status", "sensor_types"])

    df["species"] = df["species"].fillna("").astype(str).str.strip()
    df = df[df["species"] != ""]

    if "measurement_type" in df.columns:
        df["sensor_type"] = df["measurement_type"].map(_normalise_sensor_label)
    else:
        df["sensor_type"] = "Unknown"

    grouped = (
        df.groupby("species", as_index=False)
        .agg(
            common_name=("common_name", lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA),
            iucn_redlist_status=(
                "iucn_redlist_status",
                lambda s: s.dropna().iloc[0] if not s.dropna().empty else "Not Evaluated",
            ),
            sensor_types=("sensor_type", lambda s: ", ".join(sorted(set(s.dropna().astype(str))))),
        )
        .sort_values(["iucn_redlist_status", "species"])
        .reset_index(drop=True)
    )
    return grouped


def major_concern_species_table(all_species_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Create a table of species in major-concern IUCN categories."""
    redlist = redlist_status_table(all_species_df)
    if redlist.empty:
        return pd.DataFrame(columns=["species", "common_name", "iucn_redlist_status", "observation_count", "sensor_types"])

    concern = redlist[redlist["iucn_redlist_status"].isin(CONCERN_STATUSES)].copy()
    if concern.empty:
        return pd.DataFrame(columns=["species", "common_name", "iucn_redlist_status", "observation_count", "sensor_types"])

    counts = _species_series(all_species_df).value_counts().rename_axis("species").reset_index(name="observation_count")
    concern = concern.merge(counts, on="species", how="left")
    concern["observation_count"] = concern["observation_count"].fillna(0).astype(int)
    concern = concern.sort_values(["iucn_redlist_status", "observation_count"], ascending=[True, False])
    return concern.head(top_n).reset_index(drop=True)


def _normalise_sensor_filter(sensor_type: str | None) -> str | None:
    if sensor_type is None:
        return None
    value = str(sensor_type).strip().lower()
    aliases = {
        "all": None,
        "camera": "Camera",
        "camera trap": "Camera",
        "camera_trap": "Camera",
        "camera-trap": "Camera",
        "bioacoustic": "Bioacoustic",
        "audio": "Bioacoustic",
        "edna": "eDNA",
        "e dna": "eDNA",
        "e-dna": "eDNA",
        "e_dna": "eDNA",
    }
    if value not in aliases:
        raise ValueError(
            "sensor_type must be one of 'all', 'camera trap', 'bioacoustic', or 'edna'."
        )
    return aliases[value]


def plot_stations_satellite_by_sensor(
    stations_df: pd.DataFrame,
    sensor_type: str | None = "all",
) -> folium.Map:
    """Plot station points on a satellite basemap with optional sensor filtering.

    Parameters
    ----------
    stations_df:
        Station records from ``load_project_data(...).stations``.
    sensor_type:
        One of ``all`` (default), ``camera trap``, ``bioacoustic``, or ``edna``.
        Common aliases like ``camera`` and ``audio`` are also accepted.
    """
    if stations_df.empty:
        return folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    df = stations_df.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "geometry" in df.columns:
            df["latitude"] = df.geometry.y
            df["longitude"] = df.geometry.x
        else:
            return folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    df["Sensor Type"] = df.get("measurement_type", "Unknown").map(_normalise_sensor_label)
    selected = _normalise_sensor_filter(sensor_type)
    if selected is not None:
        df = df[df["Sensor Type"] == selected].copy()

    if df.empty:
        return folium.Map(location=[0, 0], zoom_start=2, tiles="Esri WorldImagery")

    center = [df["latitude"].mean(), df["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=8, tiles="Esri WorldImagery")

    for _, row in df.iterrows():
        sensor = row.get("Sensor Type", "Unknown")
        popup = (
            f"Device: {row.get('device_id', '')}<br>"
            f"Sensor: {sensor}<br>"
            f"Data type: {row.get('data_type', '')}<br>"
            f"Records: {row.get('record_count', '')}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color="#1f2933",
            weight=2,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=0.95,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(fmap)

    return fmap


def plot_sampling_location_maps(stations_df: pd.DataFrame) -> dict[str, folium.Map]:
    """Return combined and per-sensor sampling location maps."""
    return {
        "all_sampling_locations": plot_stations_satellite_by_sensor(stations_df, sensor_type="all"),
        "camera_sampling_locations": plot_stations_satellite_by_sensor(stations_df, sensor_type="camera trap"),
        "bioacoustic_sampling_locations": plot_stations_satellite_by_sensor(stations_df, sensor_type="bioacoustic"),
        "edna_sampling_locations": plot_stations_satellite_by_sensor(stations_df, sensor_type="edna"),
    }


def _stations_to_geodataframe(stations_df: pd.DataFrame) -> gpd.GeoDataFrame:
    if stations_df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    df = stations_df.copy()
    if "geometry" in df.columns and not df["geometry"].isna().all():
        try:
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=getattr(df, "crs", None) or "EPSG:4326")
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            return gdf
        except Exception:
            pass

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    mask = lat.notna() & lon.notna()
    df = df[mask].copy()
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")


def _load_project_boundary(
    project_boundary: str | Path | gpd.GeoDataFrame | None,
    *,
    target_crs: object,
) -> gpd.GeoDataFrame | None:
    if project_boundary is None:
        return None

    try:
        if isinstance(project_boundary, gpd.GeoDataFrame):
            boundary = project_boundary.copy()
        else:
            boundary = gpd.read_file(str(project_boundary))
    except Exception as exc:
        warnings.warn(f"Could not read project boundary; no boundary will be drawn. Original error: {exc}", stacklevel=2)
        return None

    if boundary.empty:
        warnings.warn("Provided project boundary is empty; no boundary will be drawn.", stacklevel=2)
        return None

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326", allow_override=True)

    try:
        return boundary.to_crs(target_crs)
    except Exception as exc:
        warnings.warn(f"Could not reproject project boundary; no boundary will be drawn. Original error: {exc}", stacklevel=2)
        return None


def _add_scalebar(ax, length_m: float | None = None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return

    if length_m is None:
        target = width * 0.2
        choices = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], dtype=float)
        length_m = float(choices[np.argmin(np.abs(choices - target))])

    start_x = x0 + width * 0.06
    start_y = y0 + height * 0.08
    end_x = start_x + length_m

    ax.plot([start_x, end_x], [start_y, start_y], color="#1f2933", linewidth=3, solid_capstyle="butt")
    ax.plot([start_x, start_x], [start_y - height * 0.008, start_y + height * 0.008], color="#1f2933", linewidth=2)
    ax.plot([end_x, end_x], [start_y - height * 0.008, start_y + height * 0.008], color="#1f2933", linewidth=2)

    if length_m >= 1000:
        label = f"{length_m / 1000:.0f} km"
    else:
        label = f"{int(length_m)} m"
    ax.text((start_x + end_x) / 2, start_y + height * 0.02, label, ha="center", va="bottom", fontsize=9, color="#1f2933")


_WEB_MERCATOR_LIMIT = 20037508.342789244
_TILE_SIZE = 256
_INITIAL_RESOLUTION = (2 * math.pi * 6378137) / _TILE_SIZE


def _mercator_to_tile(x: float, y: float, zoom: int) -> tuple[int, int]:
    n_tiles = 2**zoom
    tile_x = int(((x + _WEB_MERCATOR_LIMIT) / (2 * _WEB_MERCATOR_LIMIT)) * n_tiles)
    tile_y = int((( _WEB_MERCATOR_LIMIT - y) / (2 * _WEB_MERCATOR_LIMIT)) * n_tiles)
    tile_x = max(0, min(n_tiles - 1, tile_x))
    tile_y = max(0, min(n_tiles - 1, tile_y))
    return tile_x, tile_y


def _tile_bounds(tile_x: int, tile_y: int, zoom: int) -> tuple[float, float, float, float]:
    n_tiles = 2**zoom
    tile_span = (2 * _WEB_MERCATOR_LIMIT) / n_tiles
    minx = -_WEB_MERCATOR_LIMIT + tile_x * tile_span
    maxx = minx + tile_span
    maxy = _WEB_MERCATOR_LIMIT - tile_y * tile_span
    miny = maxy - tile_span
    return minx, maxx, miny, maxy


def _choose_satellite_zoom(x_span: float, y_span: float) -> int:
    span = max(x_span, y_span)
    if span <= 1500:
        return 16
    if span <= 3000:
        return 15
    if span <= 7000:
        return 14
    if span <= 15000:
        return 13
    if span <= 30000:
        return 12
    if span <= 60000:
        return 11
    return 10


def _fetch_satellite_tile(tile_x: int, tile_y: int, zoom: int) -> np.ndarray | None:
    url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
        f"tile/{zoom}/{tile_y}/{tile_x}"
    )
    try:
        response = httpx.get(url, timeout=20.0)
        response.raise_for_status()
        return plt.imread(BytesIO(response.content), format="jpeg")
    except Exception:
        return None


def _add_satellite_basemap(ax, *, crs: object) -> bool:
    if str(crs).upper() not in {"EPSG:3857", "3857"}:
        return False

    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    x_span = maxx - minx
    y_span = maxy - miny
    if x_span <= 0 or y_span <= 0:
        return False

    zoom = _choose_satellite_zoom(x_span, y_span)
    min_tile_x, max_tile_y = _mercator_to_tile(minx, miny, zoom)
    max_tile_x, min_tile_y = _mercator_to_tile(maxx, maxy, zoom)

    added_any = False
    for tile_x in range(min(min_tile_x, max_tile_x), max(min_tile_x, max_tile_x) + 1):
        for tile_y in range(min(min_tile_y, max_tile_y), max(min_tile_y, max_tile_y) + 1):
            image = _fetch_satellite_tile(tile_x, tile_y, zoom)
            if image is None:
                continue
            tile_minx, tile_maxx, tile_miny, tile_maxy = _tile_bounds(tile_x, tile_y, zoom)
            ax.imshow(
                image,
                extent=(tile_minx, tile_maxx, tile_miny, tile_maxy),
                interpolation="bilinear",
                zorder=0,
            )
            added_any = True

    return added_any


def plot_stations_static(
    stations_df: pd.DataFrame,
    sensor_type: str = "all",
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
):
    """Plot sampling locations as a static PNG-ready figure over satellite imagery.

    The project boundary overlay is optional and only drawn when provided.
    """
    selected = _normalise_sensor_filter(sensor_type)
    gdf = _stations_to_geodataframe(stations_df)
    if gdf.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No sampling locations available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        return fig

    gdf["Sensor Type"] = gdf.get("measurement_type", "Unknown").map(_normalise_sensor_label)
    if selected is not None:
        gdf = gdf[gdf["Sensor Type"] == selected].copy()

    if gdf.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No sampling locations available for selected sensor", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        return fig

    gdf_plot = gdf.to_crs(epsg=3857)
    boundary_gdf = _load_project_boundary(project_boundary, target_crs=gdf_plot.crs)
    if boundary_gdf is not None:
        boundary_gdf = boundary_gdf[boundary_gdf.geometry.notna()].copy()
        if boundary_gdf.empty:
            boundary_gdf = None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set a padded extent so satellite tiles and points are visible even for tight clusters.
    minx, miny, maxx, maxy = gdf_plot.total_bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    pad_x = span_x * 0.20
    pad_y = span_y * 0.20
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if not _add_satellite_basemap(ax, crs=gdf_plot.crs):
        warnings.warn("Could not load satellite basemap; plotting without tiles.", stacklevel=2)

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="#1f2933", linewidth=2, zorder=3)

    gdf_plot.plot(ax=ax, color="#ffffff", edgecolor="#1f2933", markersize=70, linewidth=1.5, zorder=4)

    _add_scalebar(ax)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


def plot_sampling_location_png(
    stations_df: pd.DataFrame,
    sensor_type: str = "all",
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
):
    """Backward-compatible wrapper for static station map plotting."""
    return plot_stations_static(
        stations_df,
        sensor_type=sensor_type,
        project_boundary=project_boundary,
    )


def plot_sensor_activity_timeline(stations_df: pd.DataFrame, sensor_type: str):
    """Plot sensor activity periods by device/location as horizontal timelines."""
    if stations_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        _apply_minimal_axes_style(ax)
        return fig

    selected = _normalise_sensor_filter(sensor_type)
    if selected is None:
        raise ValueError("sensor_type must be 'camera trap', 'bioacoustic', or 'edna'.")

    df = stations_df.copy()
    df["Sensor Type"] = df.get("measurement_type", "Unknown").map(_normalise_sensor_label)
    df = df[df["Sensor Type"] == selected].copy()

    start_col = "project_system_record_start_timestamp"
    end_col = "project_system_record_end_timestamp"
    if start_col not in df.columns or end_col not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        _apply_minimal_axes_style(ax)
        return fig

    df[start_col] = pd.to_datetime(df[start_col], errors="coerce", utc=True).dt.tz_convert(None)
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=[start_col, end_col])
    df = df[df[end_col] >= df[start_col]]

    if "device_id" not in df.columns:
        df["device_id"] = [f"site_{i+1}" for i in range(len(df))]

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        _apply_minimal_axes_style(ax)
        return fig

    devices = sorted(df["device_id"].astype(str).unique())
    y_map = {device: idx for idx, device in enumerate(devices)}

    line_color = PLOT_COLORS[0] if selected == "Camera" else PLOT_COLORS[1]

    fig_height = max(4.5, min(12, 0.3 * len(devices) + 1.5))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for _, row in df.iterrows():
        y = y_map[str(row["device_id"])]
        ax.hlines(y=y, xmin=row[start_col], xmax=row[end_col], color=line_color, linewidth=2)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{selected} locations")
    _apply_minimal_axes_style(ax)
    plt.tight_layout()
    return fig


def _filter_by_class(df: pd.DataFrame, class_name: str) -> pd.DataFrame:
    class_series = _taxonomic_class_series(df).astype(str).str.strip().str.lower()
    target = class_name.strip().lower()
    aliases = {
        "mammalia": ["mammalia", "mammal"],
        "aves": ["aves", "bird", "avian"],
    }
    tokens = aliases.get(target, [target])
    mask = class_series.apply(lambda value: any(token in value for token in tokens))
    return df[mask].copy()


def _smooth_accumulation_curve(curve: pd.DataFrame) -> pd.DataFrame:
    if curve.empty or len(curve) < 5:
        return curve

    out = curve.copy()
    window = max(7, int(len(out) * 0.05))
    if window % 2 == 0:
        window += 1
    x = out["x"].to_numpy(dtype=float)
    x_log = np.log1p(x)

    def _rolling_smooth(values: np.ndarray, smooth_window: int) -> np.ndarray:
        return (
            pd.Series(values)
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    for col in ["mean", "lower", "upper"]:
        y = out[col].to_numpy(dtype=float)
        y_roll = _rolling_smooth(y, window)

        # Blend rolling smooth with low-degree polynomial trend on log effort for a visually smooth curve.
        deg = 3 if len(y) >= 8 else 2
        try:
            coeff = np.polyfit(x_log, y_roll, deg=deg)
            y_trend = np.polyval(coeff, x_log)
            y_smooth = 0.6 * y_roll + 0.4 * y_trend
        except Exception:
            y_smooth = y_roll

        out[col] = y_smooth

    out["mean"] = np.maximum.accumulate(out["mean"].to_numpy(dtype=float))
    out["lower"] = np.maximum.accumulate(out["lower"].to_numpy(dtype=float))
    out["upper"] = np.maximum.accumulate(out["upper"].to_numpy(dtype=float))
    out["lower"] = np.minimum(out["lower"], out["mean"])
    out["upper"] = np.maximum(out["upper"], out["mean"])

    # Densify on the x-axis for smoother rendering of lines and confidence ribbons.
    if len(out) >= 3:
        x_dense = np.linspace(float(out["x"].min()), float(out["x"].max()), num=min(1200, max(240, len(out) * 10)))
        mean_dense = np.interp(x_dense, out["x"].to_numpy(dtype=float), out["mean"].to_numpy(dtype=float))
        lower_dense = np.interp(x_dense, out["x"].to_numpy(dtype=float), out["lower"].to_numpy(dtype=float))
        upper_dense = np.interp(x_dense, out["x"].to_numpy(dtype=float), out["upper"].to_numpy(dtype=float))

        dense_window = max(9, int(len(x_dense) * 0.04))
        if dense_window % 2 == 0:
            dense_window += 1
        mean_dense = _rolling_smooth(mean_dense, dense_window)
        lower_dense = _rolling_smooth(lower_dense, dense_window)
        upper_dense = _rolling_smooth(upper_dense, dense_window)

        dense = pd.DataFrame(
            {
                "x": x_dense,
                "mean": np.maximum.accumulate(mean_dense),
                "lower": np.maximum.accumulate(lower_dense),
                "upper": np.maximum.accumulate(upper_dense),
            }
        )
        dense["lower"] = np.minimum(dense["lower"], dense["mean"])
        dense["upper"] = np.maximum(dense["upper"], dense["mean"])
        return dense

    return out


def _plot_species_accumulation_for_groups(
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    groups: tuple[tuple[str, str], tuple[str, str]] = (("Mammalia", "Mammal"), ("Aves", "Bird")),
    *,
    effort_label: str = "Sampling days",
):
    # Keep one shared effort timeline across classes so panel A/B x-axes are comparable.
    shared_stations = _clip_station_windows_to_observation_range(stations_df, observations_df)

    panels: list[tuple[str, pd.DataFrame, str]] = []
    for idx, (class_name, display_name) in enumerate(groups):
        class_df = _filter_by_class(observations_df, class_name)
        if class_df.empty:
            continue
        curve = _smooth_accumulation_curve(
            _incidence_species_accumulation_curve(
                class_df,
                shared_stations,
                clip_stations_to_observations=False,
            )
        )
        if curve.empty:
            continue
        panels.append((chr(ord("A") + idx), curve, display_name))

    if not panels:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No mammal or bird accumulation data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_minimal_axes_style(ax, grid_axis=None)
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(ncols=len(panels), figsize=(7 * len(panels), 5), sharey=False)
    axes_list = np.atleast_1d(axes)
    show_letters = len(panels) > 1
    shared_x_max = max(float(curve["x"].max()) for _, curve, _ in panels)

    for ax, (label, curve, display_name) in zip(axes_list, panels):
        color = PLOT_COLORS[0] if display_name == "Mammal" else PLOT_COLORS[1]
        x_vals = np.r_[0.0, curve["x"].to_numpy(dtype=float)]
        mean_vals = np.r_[0.0, curve["mean"].to_numpy(dtype=float)]
        lower_vals = np.r_[0.0, curve["lower"].to_numpy(dtype=float)]
        upper_vals = np.r_[0.0, curve["upper"].to_numpy(dtype=float)]

        ax.plot(x_vals, mean_vals, color=color, linewidth=2)
        ax.fill_between(x_vals, lower_vals, upper_vals, color=color, alpha=0.15)
        if show_letters:
            ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
        ax.set_xlabel(effort_label)
        ax.set_ylabel("Species richness")
        ax.set_title(display_name)
        ax.set_xlim(0.0, max(1.0, shared_x_max))

        lower_min = float(curve["lower"].min()) if not curve.empty else 0.0
        upper = float(curve["upper"].max()) if not curve.empty else 0.0
        if upper <= 0:
            ax.set_ylim(0, 1)
        else:
            y_pad = (upper - lower_min) * 0.05 if upper > lower_min else max(0.25, upper * 0.05)
            y_min = max(0.0, lower_min - y_pad)
            ax.set_ylim(y_min, upper * 1.05)
        _apply_minimal_axes_style(ax, grid_axis="both")

    plt.tight_layout()
    return fig


def plot_species_accumulation_mammal_bird_by_sensor(
    camera_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    include_unobserved_stations: bool = False,
) -> dict[str, plt.Figure]:
    """Return A/B mammal-vs-bird accumulation figures for camera and bioacoustic."""
    camera_stations = _filter_stations_for_sensor(
        stations_df,
        camera_df,
        include_unobserved_stations=include_unobserved_stations,
    )
    bio_stations = _filter_stations_for_sensor(
        stations_df,
        bio_df,
        include_unobserved_stations=include_unobserved_stations,
    )
    return {
        "camera": _plot_species_accumulation_for_groups(
            camera_df,
            camera_stations,
            effort_label="Camera trap days",
        ),
        "bioacoustic": _plot_species_accumulation_for_groups(
            bio_df,
            bio_stations,
            effort_label="Bioacoustic days",
        ),
    }


def _top_species_by_class(df: pd.DataFrame, class_name: str, top_n: int = 10) -> pd.DataFrame:
    subset = _filter_by_class(df, class_name)
    names = _display_species_series(subset)
    names = names[names != ""]
    counts = names.value_counts().head(top_n).sort_values(ascending=True)
    return counts.rename_axis("species").reset_index(name="records")


def _plot_top_species_mammal_bird(df: pd.DataFrame, top_n: int = 10):
    mammals = _top_species_by_class(df, "Mammalia", top_n=top_n)
    birds = _top_species_by_class(df, "Aves", top_n=top_n)

    panels: list[tuple[str, pd.DataFrame, str, str]] = []
    if not mammals.empty:
        panels.append(("A", mammals, "Mammal species", PLOT_COLORS[0]))
    if not birds.empty:
        panels.append(("B", birds, "Bird species", PLOT_COLORS[1]))

    if not panels:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No mammal or bird species data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_minimal_axes_style(ax, grid_axis=None)
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(ncols=len(panels), figsize=(7 * len(panels), 7), sharex=False)
    axes_list = np.atleast_1d(axes)
    show_letters = len(panels) > 1

    for ax, (label, panel_df, ylabel, color) in zip(axes_list, panels):
        ax.barh(panel_df["species"], panel_df["records"], color=color)
        if show_letters:
            ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
        ax.set_xlabel("Number of records")
        ax.set_ylabel(ylabel)
        _apply_minimal_axes_style(ax, grid_axis="x")

    plt.tight_layout()
    return fig


def plot_top_species_mammal_bird_by_sensor(
    camera_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> dict[str, plt.Figure]:
    """Return A/B top mammal-vs-bird species figures for camera and bioacoustic."""
    return {
        "camera": _plot_top_species_mammal_bird(camera_df, top_n=top_n),
        "bioacoustic": _plot_top_species_mammal_bird(bio_df, top_n=top_n),
    }


def _edna_class_rank_summary(
    edna_df: pd.DataFrame,
    *,
    class_col: str | None = None,
    rank_col: str | None = None,
    taxon_col: str | None = None,
    count_col: str | None = None,
) -> pd.DataFrame:
    """Build class/rank/count summary from either raw or pre-aggregated eDNA data."""
    if edna_df.empty:
        return pd.DataFrame(columns=["resolved_class", "resolved_rank", "unique_taxa"])

    work = edna_df.copy()

    # Case 1: caller already provides aggregated class/rank/count table.
    if class_col and rank_col and count_col and {class_col, rank_col, count_col}.issubset(work.columns):
        out = work[[class_col, rank_col, count_col]].copy()
        out.columns = ["resolved_class", "resolved_rank", "unique_taxa"]
        out["resolved_class"] = out["resolved_class"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        out["resolved_rank"] = out["resolved_rank"].fillna("Order").astype(str).str.strip().str.title()
        out["resolved_rank"] = out["resolved_rank"].replace({"Higher": "Order", "Order_Or_Higher": "Order"})
        out["unique_taxa"] = pd.to_numeric(out["unique_taxa"], errors="coerce").fillna(0).astype(int)
        return out.groupby(["resolved_class", "resolved_rank"], as_index=False)["unique_taxa"].sum()

    # Case 2: infer summary from raw records.
    class_source = class_col if class_col and class_col in work.columns else None
    if class_source is None:
        for candidate in ["Class", "class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
            if candidate in work.columns:
                class_source = candidate
                break

    if class_source is None:
        resolved_class = pd.Series(["Unknown"] * len(work), index=work.index, dtype="object")
    else:
        resolved_class = work[class_source].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    rank_source = rank_col if rank_col and rank_col in work.columns else None
    if rank_source is None:
        for candidate in ["taxon_rank", "updated_taxon_rank_final", "updated_taxon_rank"]:
            if candidate in work.columns:
                rank_source = candidate
                break

    if rank_source is not None:
        resolved_rank = work[rank_source].fillna("Order").astype(str).str.strip().str.title()
    else:
        resolved_rank = pd.Series(["Order"] * len(work), index=work.index, dtype="object")

    resolved_rank = resolved_rank.replace({"Higher": "Order", "Order_Or_Higher": "Order"})

    if rank_source is None:
        # Derive lowest identification rank when explicit rank column is not present.
        for rank_name, cols in [
            ("Species", ["taxon_name", "Species", "species", "label"]),
            ("Genus", ["Genus", "genus"]),
            ("Family", ["Family", "family"]),
            ("Order", ["Order", "order"]),
        ]:
            existing = [c for c in cols if c in work.columns]
            if not existing:
                continue
            values = pd.Series([""] * len(work), index=work.index, dtype="object")
            for c in existing:
                vals = work[c].fillna("").astype(str).str.strip()
                values = values.mask(values == "", vals)
            if rank_name == "Species":
                mask = values != ""
                resolved_rank.loc[mask] = "Species"
            elif rank_name == "Genus":
                mask = (resolved_rank == "Order") & (values != "")
                resolved_rank.loc[mask] = "Genus"
            elif rank_name == "Family":
                mask = (resolved_rank == "Order") & (values != "")
                resolved_rank.loc[mask] = "Family"

    if taxon_col and taxon_col in work.columns:
        resolved_taxon = work[taxon_col].fillna("").astype(str).str.strip()
    else:
        resolved_taxon = pd.Series([""] * len(work), index=work.index, dtype="object")
        for candidate in ["taxon_name", "Species", "species", "label", "Genus", "genus", "Family", "family", "Order", "order"]:
            if candidate not in work.columns:
                continue
            vals = work[candidate].fillna("").astype(str).str.strip()
            resolved_taxon = resolved_taxon.mask(resolved_taxon == "", vals)
        resolved_taxon = resolved_taxon.mask(resolved_taxon == "", resolved_class)

    out = pd.DataFrame(
        {
            "resolved_class": resolved_class,
            "resolved_rank": resolved_rank,
            "resolved_taxon": resolved_taxon,
        }
    )
    out = out[(out["resolved_class"] != "") & (out["resolved_taxon"] != "")]

    uniq = out.drop_duplicates(["resolved_class", "resolved_rank", "resolved_taxon"])
    summary = (
        uniq.groupby(["resolved_class", "resolved_rank"], as_index=False)
        .size()
        .rename(columns={"size": "unique_taxa"})
    )
    return summary


def plot_edna_records(
    edna_df: pd.DataFrame,
    *,
    class_col: str | None = None,
    rank_col: str | None = None,
    taxon_col: str | None = None,
    count_col: str | None = None,
    rank_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    legend_title: str = "Identified to:",
    font_family: str = "Arial",
    figsize: tuple[float, float] = (5.2, 3.2),
    y_tick_step: int = 20,
    x_label_rotation: float = 45.0,
    return_summary: bool = False,
):
    """Plot compact stacked eDNA class-by-lowest-identification figure.

    The function accepts either:
    1. Raw eDNA records with taxonomic columns; or
    2. Pre-aggregated class/rank/count table (via class_col/rank_col/count_col).

    When ``return_summary=True``, returns a class-indexed summary table that
    includes rank counts and per-class percentages for each rank.
    """
    summary = _edna_class_rank_summary(
        edna_df,
        class_col=class_col,
        rank_col=rank_col,
        taxon_col=taxon_col,
        count_col=count_col,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    if summary.empty:
        ax.text(0.5, 0.5, "No eDNA records available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame()
        return fig

    rank_order_use = rank_order if rank_order is not None else ["Order", "Family", "Genus", "Species"]
    colors_use = color_map if color_map is not None else {
        "Order": "#3E5859",
        "Family": "#F6F6E2",
        "Genus": "#D9ACDE",
        "Species": "#9DEECF",
    }

    summary["resolved_rank"] = summary["resolved_rank"].astype(str).str.title().replace({"Higher": "Order"})
    available_ranks = summary["resolved_rank"].unique().tolist()
    ordered_ranks = [r for r in rank_order_use if r in available_ranks] + [
        r for r in available_ranks if r not in rank_order_use
    ]

    pivot = (
        summary.pivot_table(
            index="resolved_class",
            columns="resolved_rank",
            values="unique_taxa",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=ordered_ranks, fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    bottom = np.zeros(len(pivot), dtype=float)
    for rank in pivot.columns:
        values = pivot[rank].to_numpy(dtype=float)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=colors_use.get(rank, "#bdbdbd"),
            edgecolor="#333333",
            linewidth=0.6,
            label=rank,
            clip_on=False,
        )
        bottom += values

    max_total = float(pivot.sum(axis=1).max()) if not pivot.empty else 0.0
    y_max = int(np.ceil(max_total / float(y_tick_step)) * y_tick_step) if max_total > 0 else y_tick_step
    # Small lower padding preserves the bottom bar edge without showing an x-axis baseline.
    y_lower_pad = max(0.8, float(y_tick_step) * 0.04)
    ax.set_ylim(-y_lower_pad, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, y_tick_step))
    ax.spines["left"].set_bounds(0, y_max)

    ax.set_ylabel("Number of unique taxa", fontfamily=font_family)
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, rotation=x_label_rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_fontfamily(font_family)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_family)

    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        borderaxespad=0.0,
        prop={"family": font_family, "size": 8.5},
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontfamily(font_family)
        legend.get_title().set_fontsize(9)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.96)

    if return_summary:
        summary_out = pivot.copy()
        totals = summary_out.sum(axis=1).astype(float)
        summary_out["Total"] = totals.astype(int)
        for rank in ordered_ranks:
            pct_col = f"{rank} (%)"
            summary_out[pct_col] = np.where(
                totals > 0,
                (summary_out[rank].astype(float) / totals) * 100.0,
                0.0,
            ).round(1)
        return fig, summary_out
    return fig


def plot_edna_unique_taxa_stacked(edna_df: pd.DataFrame):
    """Backward-compatible wrapper for legacy eDNA stacked figure calls."""
    return plot_edna_records(edna_df)


def plot_edna_iucn_by_class_species(
    edna_df: pd.DataFrame,
    *,
    class_col: str | None = None,
    species_col: str | None = None,
    iucn_col: str | None = None,
    rank_col: str | None = None,
    exclude_statuses: set[str] | None = None,
    status_order: list[str] | None = None,
    class_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    legend_title: str = "IUCN status",
    legend_loc: str = "upper right",
    font_family: str = "Arial",
    figsize: tuple[float, float] = (6.2, 3.8),
    y_max: int = 35,
    y_tick_step: int = 5,
    x_label_rotation: float = 45.0,
    return_summary: bool = False,
):
    """Plot species-level unique taxa by class, stacked by IUCN status.

    Styling aligns with ``plot_edna_records``: no x-axis line, compact layout,
    and y-axis left spine bounded to the plotting range.
    """
    if edna_df.empty:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No eDNA records available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame(), pd.DataFrame()
        return fig

    work = edna_df.copy()

    class_source = class_col if class_col and class_col in work.columns else None
    if class_source is None:
        for candidate in ["Class", "class", "class_", "taxon_class", "taxonomic_class", "class_name"]:
            if candidate in work.columns:
                class_source = candidate
                break

    species_source = species_col if species_col and species_col in work.columns else None
    if species_source is None:
        for candidate in ["Species", "species", "taxon_name", "label"]:
            if candidate in work.columns:
                species_source = candidate
                break

    iucn_source = iucn_col if iucn_col and iucn_col in work.columns else None
    if iucn_source is None:
        for candidate in ["IUCN_status_external", "iucn_redlist_status", "IUCN_status", "redlist_status"]:
            if candidate in work.columns:
                iucn_source = candidate
                break

    if class_source is None or species_source is None or iucn_source is None:
        raise ValueError(
            "Missing required columns. Need class, species/taxon, and IUCN status columns."
        )

    selected_cols = [class_source, species_source, iucn_source]
    if rank_col is not None and rank_col in work.columns:
        selected_cols.append(rank_col)

    subset = work[selected_cols].copy()
    subset.columns = ["class_name", "taxon", "iucn_status"] + (["taxon_rank"] if len(selected_cols) == 4 else [])

    subset["class_name"] = subset["class_name"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    subset["taxon"] = subset["taxon"].fillna("").astype(str).str.strip()
    subset["iucn_status"] = subset["iucn_status"].fillna("").astype(str).str.strip()

    # Keep species-level taxa only.
    if "taxon_rank" in subset.columns:
        subset["taxon_rank"] = subset["taxon_rank"].fillna("").astype(str).str.strip().str.lower()
        subset = subset[subset["taxon_rank"] == "species"].copy()
    else:
        subset = subset[
            subset["taxon"].str.contains(r"^[A-Z][a-z]+\s+[a-z][a-z-]+$", regex=True, na=False)
        ].copy()

    default_excluded = {""}
    excluded = {v.lower() for v in (exclude_statuses if exclude_statuses is not None else default_excluded)}
    subset = subset[(subset["taxon"] != "") & (~subset["iucn_status"].str.lower().isin(excluded))].copy()

    if subset.empty:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")
        ax.text(0.5, 0.5, "No species-level taxa with evaluated IUCN status", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame(), pd.DataFrame()
        return fig

    subset["iucn_status"] = subset["iucn_status"].replace(
        {
            "Endangered status": "Endangered",
            "Least concern": "Least Concern",
            "Near threatened": "Near Threatened",
            "Data deficient": "Data Deficient",
            "Critically endangered": "Critically Endangered",
            "Not evaluated": "Not Evaluated",
            "Not_evaluated": "Not Evaluated",
            "NE": "Not Evaluated",
            "ne": "Not Evaluated",
        }
    )

    summary = (
        subset.drop_duplicates(["class_name", "taxon", "iucn_status"])
        .groupby(["class_name", "iucn_status"], as_index=False)
        .size()
        .rename(columns={"size": "unique_taxa"})
    )

    pivot = summary.pivot_table(
        index="class_name",
        columns="iucn_status",
        values="unique_taxa",
        aggfunc="sum",
        fill_value=0,
    )
    if class_order is None:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    else:
        class_order_use = [name for name in class_order if name in pivot.index] + [
            name for name in pivot.index if name not in class_order
        ]
        pivot = pivot.reindex(class_order_use)

    if status_order is None:
        default_order = [
            "Critically Endangered",
            "Endangered",
            "Vulnerable",
            "Near Threatened",
            "Least Concern",
            "Data Deficient",
            "Not Evaluated",
        ]
        ordered_status = [s for s in default_order if s in pivot.columns] + [s for s in pivot.columns if s not in default_order]
    else:
        ordered_status = [s for s in status_order if s in pivot.columns] + [s for s in pivot.columns if s not in status_order]
    pivot = pivot[ordered_status]

    colors_use = color_map if color_map is not None else {
        "Critically Endangered": "#7f0000",
        "Endangered": "#d7301f",
        "Vulnerable": "#fc8d59",
        "Near Threatened": "#fdcc8a",
        "Least Concern": "#c7e9b4",
        "Data Deficient": "#9ecae1",
        "Not Evaluated": "#d9d9d9",
    }

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    bottom = pd.Series(0, index=pivot.index, dtype=float)
    for status in pivot.columns:
        vals = pivot[status]
        ax.bar(
            pivot.index,
            vals,
            bottom=bottom,
            label=status,
            color=colors_use.get(status, "#bdbdbd"),
            edgecolor="#333333",
            linewidth=0.6,
            clip_on=False,
        )
        bottom = bottom + vals

    max_stack_height = float(bottom.max()) if not bottom.empty else 0.0
    tick_step = max(int(y_tick_step), 1)
    y_ceiling = max(float(y_max), max_stack_height)
    y_ceiling = float(max(tick_step, int(math.ceil(y_ceiling / tick_step) * tick_step)))

    ax.set_ylabel("Number of unique species", fontfamily=font_family)
    ax.set_xlabel("")
    ax.set_ylim(0, y_ceiling)
    ax.set_yticks(np.arange(0, y_ceiling + 1, tick_step))
    ax.spines["left"].set_bounds(0, y_ceiling)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, rotation=x_label_rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_fontfamily(font_family)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_family)

    legend = ax.legend(
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        borderaxespad=0.0,
        prop={"family": font_family, "size": 8.5},
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontfamily(font_family)
        legend.get_title().set_fontsize(9)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.96)

    if return_summary:
        return fig, summary, pivot
    return fig


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _clean_text_value(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "na"}:
        return ""
    return text


def _round_to_nice_count(value: float) -> int:
    value = float(value)
    if not np.isfinite(value) or value <= 1:
        return 1
    if value < 10:
        return max(1, int(round(value)))
    magnitude = 10 ** int(math.floor(math.log10(value)))
    return max(1, int(round(value / magnitude) * magnitude))


def _rounded_legend_values(counts: np.ndarray, percentiles: tuple[float, ...]) -> np.ndarray:
    if counts.size == 0:
        return np.array([], dtype=int)

    values = np.percentile(counts.astype(float), tuple(float(v) for v in percentiles))
    rounded = np.array([_round_to_nice_count(v) for v in values], dtype=int)
    rounded = rounded[rounded > 0]

    if rounded.size == 0:
        return np.array([1], dtype=int)

    rounded = np.unique(rounded)
    max_count = int(np.nanmax(counts))
    if max_count > 0 and rounded[-1] < max_count:
        rounded = np.unique(np.append(rounded, _round_to_nice_count(max_count)))

    return rounded.astype(int)


def _bounded_marker_areas(counts: np.ndarray, size_scale: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return np.array([], dtype=float)

    scale = max(float(size_scale), 0.1) / 9.0
    min_area = 42.0 * scale
    max_area = 680.0 * scale

    max_count = float(np.nanmax(counts))
    if max_count <= 0:
        return np.full_like(counts, min_area)

    normalised = np.clip(counts / max_count, 0.0, 1.0)
    return min_area + (normalised * (max_area - min_area))


def plot_edna_data(
    edna_df: pd.DataFrame,
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    location_col: str | None = None,
    sequence_col: str | None = None,
    lon_col: str | None = None,
    lat_col: str | None = None,
    size_scale: float = 9.0,
    figsize: tuple[float, float] = (7.6, 8.6),
    point_color: str = "#e2b318",
    point_edgecolor: str = "white",
    point_linewidth: float = 1.6,
    point_alpha: float = 0.95,
    boundary_facecolor: str = "#d9ded2",
    boundary_edgecolor: str = "white",
    boundary_linewidth: float = 2.8,
    boundary_alpha: float = 0.28,
    legend_title: str = "Sequences per site",
    legend_location: str = "lower left",
    legend_percentiles: tuple[float, float, float] = (10.0, 50.0, 90.0),
    show_legend: bool = True,
    return_summary: bool = False,
):
    """Plot eDNA sampling locations as bubble map over satellite imagery.

    Args:
        edna_df: Raw eDNA records with site identifier, sequence, and coordinates.
        project_boundary: Optional project boundary path or GeoDataFrame.
        location_col: Site identifier column; inferred when omitted.
        sequence_col: Sequence column used to compute per-site sequence counts.
        lon_col: Longitude column; inferred when omitted.
        lat_col: Latitude column; inferred when omitted.
        size_scale: Marker area scale multiplier for bubble sizes.
        figsize: Figure size in inches.
        point_color: Bubble fill color.
        point_edgecolor: Bubble edge color.
        point_linewidth: Bubble edge line width.
        point_alpha: Bubble opacity.
        boundary_facecolor: Project boundary polygon fill color.
        boundary_edgecolor: Project boundary line color.
        boundary_linewidth: Project boundary line width.
        boundary_alpha: Project boundary fill opacity.
        legend_title: Legend title for bubble sizes.
        legend_location: Matplotlib legend location (for example ``lower left``).
        legend_percentiles: Percentiles used to choose legend bubble sizes.
        show_legend: Whether to draw the size legend.
        return_summary: When True, also return per-site summary DataFrame.

    Returns:
        Matplotlib Figure, and optionally the per-site summary table.

    Raises:
        ValueError: If required columns are missing or no valid site records remain.
    """
    if edna_df.empty:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.text(0.5, 0.5, "No eDNA records available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame(columns=["site_id", "longitude", "latitude", "sequence_count"])
        return fig

    location_source = location_col if location_col and location_col in edna_df.columns else None
    if location_source is None:
        location_source = _pick_existing_column(
            edna_df,
            ["eDNA_file_reference_location", "location", "Location", "site", "site_name", "station", "sample_id"],
        )

    sequence_source = sequence_col if sequence_col and sequence_col in edna_df.columns else None
    if sequence_source is None:
        sequence_source = _pick_existing_column(edna_df, ["genetic_sequence", "sequence", "dna_sequence"])

    lon_source = lon_col if lon_col and lon_col in edna_df.columns else None
    if lon_source is None:
        lon_source = _pick_existing_column(edna_df, ["longitude", "lon", "x"])

    lat_source = lat_col if lat_col and lat_col in edna_df.columns else None
    if lat_source is None:
        lat_source = _pick_existing_column(edna_df, ["latitude", "lat", "y"])

    if location_source is None or sequence_source is None or lon_source is None or lat_source is None:
        raise ValueError("Required columns not found for site identifier, sequence, longitude, and latitude.")

    work = edna_df[[location_source, sequence_source, lon_source, lat_source]].copy()
    work = work.rename(
        columns={
            location_source: "site_id",
            sequence_source: "sequence",
            lon_source: "longitude",
            lat_source: "latitude",
        }
    )

    work["site_id"] = work["site_id"].apply(_clean_text_value)
    work["sequence"] = work["sequence"].apply(_clean_text_value)
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")

    work = work[
        (work["site_id"] != "")
        & (work["sequence"] != "")
        & work["longitude"].notna()
        & work["latitude"].notna()
    ].copy()

    if work.empty:
        raise ValueError("No valid eDNA sites found after filtering.")

    site_summary = (
        work.groupby("site_id", as_index=False)
        .agg(
            longitude=("longitude", "median"),
            latitude=("latitude", "median"),
            sequence_count=("sequence", "nunique"),
        )
        .sort_values("sequence_count", ascending=False)
        .reset_index(drop=True)
    )

    sites_gdf = gpd.GeoDataFrame(
        site_summary,
        geometry=gpd.points_from_xy(site_summary["longitude"], site_summary["latitude"]),
        crs="EPSG:4326",
    )
    sites_plot = sites_gdf.to_crs(epsg=3857)
    sites_plot["plot_size"] = sites_plot["sequence_count"] * float(size_scale)

    boundary_gdf = _load_project_boundary(project_boundary, target_crs=sites_plot.crs)

    if boundary_gdf is not None and not boundary_gdf.empty:
        xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
    else:
        xmin, ymin, xmax, ymax = sites_plot.total_bounds

    xpad = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if not _add_satellite_basemap(ax, crs=sites_plot.crs):
        warnings.warn("Could not load satellite basemap; plotting without tiles.", stacklevel=2)
        ax.set_facecolor("#153520")

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.plot(
            ax=ax,
            facecolor=boundary_facecolor,
            edgecolor=boundary_edgecolor,
            linewidth=boundary_linewidth,
            alpha=boundary_alpha,
            zorder=3,
        )

    ax.scatter(
        sites_plot.geometry.x,
        sites_plot.geometry.y,
        s=sites_plot["plot_size"].to_numpy(dtype=float),
        c=point_color,
        edgecolors=point_edgecolor,
        linewidths=point_linewidth,
        alpha=point_alpha,
        zorder=4,
    )

    if show_legend and not sites_plot.empty:
        percentiles = tuple(float(v) for v in legend_percentiles)
        values = np.percentile(sites_plot["sequence_count"].to_numpy(dtype=float), percentiles)
        legend_values = np.unique(
            np.clip(
                np.rint(values).astype(int),
                int(sites_plot["sequence_count"].min()),
                int(sites_plot["sequence_count"].max()),
            )
        )

        legend_handles = []
        for value in legend_values:
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markersize=np.sqrt(float(value) * float(size_scale)),
                    markerfacecolor=point_color,
                    markeredgecolor=point_edgecolor,
                    markeredgewidth=max(1.0, point_linewidth - 0.3),
                    alpha=point_alpha,
                    label=f"{int(value)}",
                )
            )

        legend = ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc=legend_location,
            frameon=True,
            facecolor="white",
            edgecolor="#d7d7d7",
            framealpha=0.95,
            fontsize=9,
            title_fontsize=10,
            borderpad=0.8,
            labelspacing=0.7,
        )
        legend.set_zorder(10)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()

    if return_summary:
        return fig, site_summary
    return fig


def plot_station_data(
    stations_df: pd.DataFrame,
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    location_col: str | None = None,
    sequence_col: str | None = None,
    lon_col: str | None = None,
    lat_col: str | None = None,
    year_col: str | None = None,
    color_by_year: bool = False,
    sensor_type_col: str | None = None,
    color_by_sensor_type: bool = False,
    clean_points: bool = True,
    clean_point_size: float = 52.0,
    size_scale: float = 9.0,
    figsize: tuple[float, float] = (7.6, 8.6),
    point_color: str = "#1a2e2f",
    point_edgecolor: str = "none",
    point_linewidth: float = 0.0,
    point_alpha: float = 1.0,
    boundary_facecolor: str = "#d9ded2",
    boundary_edgecolor: str = "white",
    boundary_linewidth: float = 2.8,
    boundary_alpha: float = 0.28,
    legend_title: str = "Records per site",
    legend_location: str = "lower left",
    legend_percentiles: tuple[float, float, float] = (10.0, 50.0, 90.0),
    show_legend: bool = True,
    return_summary: bool = False,
):
    """Plot camera trap or bioacoustic station locations over satellite imagery.

    Mirrors the style of :func:`plot_edna_data`.  When ``color_by_year`` is
    ``True`` each station is coloured by the year it was active; a categorical
    year legend replaces the default single-colour palette and the bubble-size
    legend is still drawn when ``show_legend`` is ``True``.

    Args:
        stations_df: Station records.  May be station-level (one row per
            deployment) or observation-level (one row per detection).  Rows are
            aggregated by ``location_col`` to produce per-site record counts.
        project_boundary: Optional project boundary path or GeoDataFrame.
        location_col: Site identifier column; inferred when omitted
            (``device_id`` → ``station_id`` → ``station_name`` → …).
        sequence_col: Column whose distinct values are counted per site to
            derive bubble size.  When ``None`` each row counts as one record.
        lon_col: Longitude column; inferred when omitted.
        lat_col: Latitude column; inferred when omitted.
        year_col: Column from which the deployment year is derived.  Inferred
            from ``project_system_record_start_timestamp`` or any column whose
            name contains ``"start"`` or ``"date"`` when omitted.  Only used
            when ``color_by_year=True``.
        color_by_year: When ``True``, colour stations by year collected using a
            qualitative palette; otherwise all stations share ``point_color``.
        sensor_type_col: Column identifying the sensor/survey type (e.g.
            ``measurement_type``, ``sensor_type``).  Inferred from common
            column names when omitted.  Only used when ``color_by_sensor_type=True``.
        color_by_sensor_type: When ``True``, colour stations by sensor type
            (e.g. camera trap, bioacoustic, eDNA) using a qualitative palette;
            a categorical legend replaces the single-colour default.
        clean_points: When ``True``, draw stations as uniform clean points
            (no size scaling by record count).
        clean_point_size: Marker area used when ``clean_points=True``.
        size_scale: Marker area scale multiplier for bubble sizes when
            ``clean_points=False``.
        figsize: Figure size in inches.
        point_color: Bubble fill colour used when ``color_by_year=False``.
        point_edgecolor: Bubble edge colour.
        point_linewidth: Bubble edge line width.
        point_alpha: Bubble opacity.
        boundary_facecolor: Project boundary polygon fill colour.
        boundary_edgecolor: Project boundary line colour.
        boundary_linewidth: Project boundary line width.
        boundary_alpha: Project boundary fill opacity.
        legend_title: Legend title for bubble sizes (ignored when
            ``clean_points=True``).
        legend_location: Matplotlib legend location string (e.g. ``lower left``).
        legend_percentiles: Percentiles used to choose representative bubble
            sizes for the size legend.
        show_legend: Whether to draw legends (year or sensor-type legend when
            the corresponding flag is ``True`` and size legend when
            ``clean_points=False``).
        return_summary: When ``True``, also return the per-site summary DataFrame.

    Returns:
        Matplotlib Figure, and optionally the per-site summary table.

    Raises:
        ValueError: If required columns are missing or no valid site records remain.
    """
    _YEAR_PALETTE = [
        "#1a2e2f", "#7b2d8b", "#b85c00", "#1a5c48",
        "#8b1a1a", "#1a3a8b", "#4a7a1a", "#c23b00",
        "#5b1a8b", "#006a78",
    ]

    if stations_df.empty:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.text(0.5, 0.5, "No station records available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame(columns=["site_id", "longitude", "latitude", "record_count"])
        return fig

    # --- resolve columns -------------------------------------------------
    location_source = location_col if location_col and location_col in stations_df.columns else None
    if location_source is None:
        location_source = _pick_existing_column(
            stations_df,
            ["device_id", "station_id", "station_name", "site_id", "location", "site", "name"],
        )
    # If no site identifier, create one from index
    if location_source is None:
        stations_df = stations_df.copy()
        stations_df["site_id"] = stations_df.index.astype(str)
        location_source = "site_id"

    sequence_source = sequence_col if sequence_col and sequence_col in stations_df.columns else None

    lon_source = lon_col if lon_col and lon_col in stations_df.columns else None
    if lon_source is None:
        lon_source = _pick_existing_column(stations_df, ["longitude", "lon", "x"])

    lat_source = lat_col if lat_col and lat_col in stations_df.columns else None
    if lat_source is None:
        lat_source = _pick_existing_column(stations_df, ["latitude", "lat", "y"])

    # Fall back to extracting coordinates from a geometry column (GeoDataFrame or DataFrame with 'geometry')
    _coords_from_geometry = False
    if (lon_source is None or lat_source is None):
        geom_col = None
        if hasattr(stations_df, "geometry") and hasattr(stations_df.geometry, "name"):
            geom_col = stations_df.geometry.name
        elif "geometry" in stations_df.columns:
            geom_col = "geometry"
        if geom_col and stations_df[geom_col].notna().any():
            _coords_from_geometry = True

    if location_source is None or (lon_source is None and not _coords_from_geometry):
        raise ValueError("Required columns not found for site identifier, longitude, and latitude or geometry.")

    # --- resolve year column ---------------------------------------------
    year_source: str | None = None
    if color_by_year:
        if year_col and year_col in stations_df.columns:
            year_source = year_col
        else:
            year_source = _pick_existing_column(
                stations_df,
                ["project_system_record_start_timestamp", "start_timestamp", "start_date", "date", "year"],
            )
            if year_source is None:
                year_source = next(
                    (c for c in stations_df.columns if "start" in c.lower() or "date" in c.lower()),
                    None,
                )

    # --- resolve sensor type column --------------------------------------
    sensor_source: str | None = None
    if color_by_sensor_type:
        if sensor_type_col and sensor_type_col in stations_df.columns:
            sensor_source = sensor_type_col
        else:
            sensor_source = _pick_existing_column(
                stations_df,
                ["sensor_type", "measurement_type", "device_type", "survey_type", "type"],
            )

    # --- build working frame ---------------------------------------------
    if _coords_from_geometry:
        # Reproject to WGS-84 and pull x/y from geometry
        import geopandas as gpd
        if not isinstance(stations_df, gpd.GeoDataFrame):
            # If geometry column exists but not a GeoDataFrame, convert
            geom_col = "geometry" if "geometry" in stations_df.columns else None
            stations_df = gpd.GeoDataFrame(stations_df, geometry=geom_col)
        if stations_df.crs is None:
            stations_df = stations_df.set_crs("EPSG:4326", allow_override=True)
        src_wgs = stations_df.to_crs("EPSG:4326")
        stations_df = pd.DataFrame(stations_df)  # detach geometry for uniform handling
        stations_df["longitude"] = src_wgs.geometry.x.values
        stations_df["latitude"] = src_wgs.geometry.y.values
        lon_source = "longitude"
        lat_source = "latitude"

    # Capture whether sequence column is numeric BEFORE renaming/coercing
    _sequence_numeric: bool = (
        sequence_source is not None
        and pd.api.types.is_numeric_dtype(stations_df[sequence_source])
    )

    cols = [location_source, lon_source, lat_source]
    if sequence_source:
        cols.append(sequence_source)
    if year_source:
        cols.append(year_source)
    if sensor_source:
        cols.append(sensor_source)
    work = stations_df[list(dict.fromkeys(cols))].copy()

    rename_map: dict[str, str] = {
        location_source: "site_id",
        lon_source: "longitude",
        lat_source: "latitude",
    }
    if sequence_source:
        rename_map[sequence_source] = "sequence"
    if year_source:
        rename_map[year_source] = "_year_raw"
    if sensor_source:
        rename_map[sensor_source] = "_sensor_type"
    work = work.rename(columns=rename_map)

    work["site_id"] = work["site_id"].apply(_clean_text_value)
    work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
    work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
    if "sequence" in work.columns:
        if _sequence_numeric:
            work["sequence"] = pd.to_numeric(work["sequence"], errors="coerce")
        else:
            work["sequence"] = work["sequence"].apply(_clean_text_value)

    # derive integer year from date/timestamp column
    if "_year_raw" in work.columns:
        parsed = pd.to_datetime(work["_year_raw"], errors="coerce", utc=True)
        work["_year"] = parsed.dt.year.where(parsed.notna(), other=None)
    else:
        work["_year"] = None

    # clean sensor type values
    if "_sensor_type" in work.columns:
        work["_sensor_type"] = work["_sensor_type"].apply(_clean_text_value).replace("", None)
    else:
        work["_sensor_type"] = None

    work = work[
        (work["site_id"] != "")
        & work["longitude"].notna()
        & work["latitude"].notna()
    ].copy()

    if work.empty:
        raise ValueError("No valid station sites found after filtering.")

    # --- aggregate per site ----------------------------------------------
    agg_dict: dict[str, object] = {
        "longitude": ("longitude", "median"),
        "latitude": ("latitude", "median"),
    }
    if "sequence" in work.columns:
        if _sequence_numeric:
            agg_dict["record_count"] = ("sequence", "sum")
        else:
            agg_dict["record_count"] = ("sequence", "nunique")
    else:
        agg_dict["record_count"] = ("site_id", "count")

    def _modal_str(s: pd.Series) -> object:
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else None

    base_summary = work.groupby("site_id", as_index=False).agg(**agg_dict)

    if "_year" in work.columns:
        year_mode = work.groupby("site_id")["_year"].agg(_modal_str).rename("year")
        base_summary = base_summary.merge(year_mode.reset_index(), on="site_id", how="left")
    else:
        base_summary["year"] = None

    if "_sensor_type" in work.columns:
        sensor_mode = work.groupby("site_id")["_sensor_type"].agg(_modal_str).rename("sensor_type")
        base_summary = base_summary.merge(sensor_mode.reset_index(), on="site_id", how="left")
    else:
        base_summary["sensor_type"] = None

    site_summary = base_summary

    site_summary = site_summary.sort_values("record_count", ascending=False).reset_index(drop=True)

    # --- project to Web Mercator -----------------------------------------
    sites_gdf = gpd.GeoDataFrame(
        site_summary,
        geometry=gpd.points_from_xy(site_summary["longitude"], site_summary["latitude"]),
        crs="EPSG:4326",
    )
    sites_plot = sites_gdf.to_crs(epsg=3857)
    if clean_points:
        point_area = max(float(clean_point_size), 1.0)
        sites_plot["plot_size"] = np.full(len(sites_plot), point_area, dtype=float)
    else:
        sites_plot["plot_size"] = _bounded_marker_areas(
            sites_plot["record_count"].to_numpy(dtype=float),
            size_scale,
        )

    boundary_gdf = _load_project_boundary(project_boundary, target_crs=sites_plot.crs)

    if boundary_gdf is not None and not boundary_gdf.empty:
        xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
    else:
        xmin, ymin, xmax, ymax = sites_plot.total_bounds

    xpad = (xmax - xmin) * 0.06 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.06 if ymax > ymin else 1.0

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

    if not _add_satellite_basemap(ax, crs=sites_plot.crs):
        warnings.warn("Could not load satellite basemap; plotting without tiles.", stacklevel=2)
        ax.set_facecolor("#153520")

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.plot(
            ax=ax,
            facecolor=boundary_facecolor,
            edgecolor=boundary_edgecolor,
            linewidth=boundary_linewidth,
            alpha=boundary_alpha,
            zorder=3,
        )

    # --- draw points with optional year or sensor-type colouring --------
    if color_by_sensor_type and "sensor_type" in sites_plot.columns and sites_plot["sensor_type"].notna().any():
        valid_sensor_types = sorted(sites_plot["sensor_type"].dropna().unique().tolist())
        sensor_color_map = {st: _YEAR_PALETTE[i % len(_YEAR_PALETTE)] for i, st in enumerate(valid_sensor_types)}

        for st in valid_sensor_types:
            mask = sites_plot["sensor_type"] == st
            subset = sites_plot[mask]
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                s=subset["plot_size"].to_numpy(dtype=float),
                c=sensor_color_map[st],
                edgecolors=point_edgecolor,
                linewidths=point_linewidth,
                alpha=point_alpha,
                zorder=4,
                label=str(st),
            )

        no_sensor = sites_plot[sites_plot["sensor_type"].isna()]
        if not no_sensor.empty:
            ax.scatter(
                no_sensor.geometry.x,
                no_sensor.geometry.y,
                s=no_sensor["plot_size"].to_numpy(dtype=float),
                c="#aaaaaa",
                edgecolors=point_edgecolor,
                linewidths=point_linewidth,
                alpha=point_alpha,
                zorder=4,
                label="Unknown",
            )

        if show_legend:
            sensor_handles = [
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=sensor_color_map[st],
                    markeredgecolor=point_edgecolor,
                    markeredgewidth=max(1.0, point_linewidth - 0.3),
                    alpha=point_alpha,
                    label=str(st),
                )
                for st in valid_sensor_types
            ]
            if not no_sensor.empty:
                sensor_handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="",
                        markersize=8,
                        markerfacecolor="#aaaaaa",
                        markeredgecolor=point_edgecolor,
                        markeredgewidth=max(1.0, point_linewidth - 0.3),
                        alpha=point_alpha,
                        label="Unknown",
                    )
                )
            sensor_legend = ax.legend(
                handles=sensor_handles,
                title="Sensor type",
                loc=legend_location,
                frameon=True,
                facecolor="white",
                edgecolor="#d7d7d7",
                framealpha=0.95,
                fontsize=9,
                title_fontsize=10,
                borderpad=0.8,
                labelspacing=0.7,
            )
            sensor_legend.set_zorder(10)
            ax.add_artist(sensor_legend)

    elif color_by_year and sites_plot["year"].notna().any():
        valid_years = sorted(sites_plot["year"].dropna().unique().astype(int).tolist())
        year_color_map = {yr: _YEAR_PALETTE[i % len(_YEAR_PALETTE)] for i, yr in enumerate(valid_years)}

        for yr in valid_years:
            mask = sites_plot["year"] == yr
            subset = sites_plot[mask]
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                s=subset["plot_size"].to_numpy(dtype=float),
                c=year_color_map[yr],
                edgecolors=point_edgecolor,
                linewidths=point_linewidth,
                alpha=point_alpha,
                zorder=4,
                label=str(yr),
            )

        # stations with no year (draw in neutral colour)
        no_year = sites_plot[sites_plot["year"].isna()]
        if not no_year.empty:
            ax.scatter(
                no_year.geometry.x,
                no_year.geometry.y,
                s=no_year["plot_size"].to_numpy(dtype=float),
                c="#aaaaaa",
                edgecolors=point_edgecolor,
                linewidths=point_linewidth,
                alpha=point_alpha,
                zorder=4,
                label="Unknown",
            )

        if show_legend:
            year_handles = [
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=8,
                    markerfacecolor=year_color_map[yr],
                    markeredgecolor=point_edgecolor,
                    markeredgewidth=max(1.0, point_linewidth - 0.3),
                    alpha=point_alpha,
                    label=str(yr),
                )
                for yr in valid_years
            ]
            if not no_year.empty:
                year_handles.append(
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="",
                        markersize=8,
                        markerfacecolor="#aaaaaa",
                        markeredgecolor=point_edgecolor,
                        markeredgewidth=max(1.0, point_linewidth - 0.3),
                        alpha=point_alpha,
                        label="Unknown",
                    )
                )
            year_legend = ax.legend(
                handles=year_handles,
                title="Year collected",
                loc=legend_location,
                frameon=True,
                facecolor="white",
                edgecolor="#d7d7d7",
                framealpha=0.95,
                fontsize=9,
                title_fontsize=10,
                borderpad=0.8,
                labelspacing=0.7,
            )
            year_legend.set_zorder(10)
            ax.add_artist(year_legend)
    else:
        ax.scatter(
            sites_plot.geometry.x,
            sites_plot.geometry.y,
            s=sites_plot["plot_size"].to_numpy(dtype=float),
            c=point_color,
            edgecolors=point_edgecolor,
            linewidths=point_linewidth,
            alpha=point_alpha,
            zorder=4,
        )

    # --- bubble-size legend ----------------------------------------------
    if show_legend and not clean_points and not sites_plot.empty:
        counts = sites_plot["record_count"].to_numpy(dtype=float)
        percentiles = tuple(float(v) for v in legend_percentiles)
        legend_values = _rounded_legend_values(
            counts,
            percentiles,
        )

        # use neutral grey for the size legend when categorical colouring is active
        size_legend_color = "#777777" if (color_by_year or color_by_sensor_type) else point_color

        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=np.sqrt(float(_bounded_marker_areas(np.array([v], dtype=float), size_scale)[0])),
                markerfacecolor=size_legend_color,
                markeredgecolor=point_edgecolor,
                markeredgewidth=max(1.0, point_linewidth - 0.3),
                alpha=point_alpha,
                label=f"{int(v)}",
            )
            for v in legend_values
        ]

        size_loc = "lower right" if ((color_by_year or color_by_sensor_type) and legend_location in {"lower left", "lower right"}) else legend_location
        size_legend = ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc=size_loc,
            frameon=True,
            facecolor="white",
            edgecolor="#d7d7d7",
            framealpha=0.95,
            fontsize=9,
            title_fontsize=10,
            borderpad=0.8,
            labelspacing=0.7,
        )
        size_legend.set_zorder(10)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    fig.tight_layout()

    if return_summary:
        return fig, site_summary
    return fig


def _build_all_device_days(stations_df: pd.DataFrame) -> pd.DataFrame:
    if stations_df.empty or "device_id" not in stations_df.columns:
        return pd.DataFrame(columns=["device_id", "date"])

    start_candidates = [
        "project_system_record_start_timestamp",
        "installation_timestamp",
        "deployment_timestamp",
        "start_timestamp",
        "start_time",
    ]
    end_candidates = [
        "project_system_record_end_timestamp",
        "removal_timestamp",
        "retrieval_timestamp",
        "end_timestamp",
        "end_time",
    ]

    start_col = next((c for c in start_candidates if c in stations_df.columns), None)
    end_col = next((c for c in end_candidates if c in stations_df.columns), None)
    if start_col is None or end_col is None:
        return pd.DataFrame(columns=["device_id", "date"])

    def _as_day(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

    work = stations_df[["device_id", start_col, end_col]].copy()
    work[start_col] = _as_day(work[start_col])
    work[end_col] = _as_day(work[end_col])

    rows: list[pd.DataFrame] = []
    for _, row in work.iterrows():
        start = row[start_col]
        end = row[end_col]
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        dates = pd.date_range(start, end, freq="D")
        rows.append(pd.DataFrame({"device_id": row["device_id"], "date": dates}))

    if not rows:
        return pd.DataFrame(columns=["device_id", "date"])

    return pd.concat(rows, ignore_index=True).drop_duplicates(["device_id", "date"])


def _clip_station_windows_to_observation_range(
    stations_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clip station deployment windows to observed date ranges per device.

    This prevents inflated effort when a station end date is open-ended
    (for example, far-future placeholders) while observations are only present
    for a shorter period.
    """
    if stations_df.empty:
        return stations_df

    ts_col = _timestamp_column(observations_df)
    if ts_col is None or "device_id" not in observations_df.columns:
        return stations_df

    start_candidates = [
        "project_system_record_start_timestamp",
        "installation_timestamp",
        "deployment_timestamp",
        "start_timestamp",
        "start_time",
    ]
    end_candidates = [
        "project_system_record_end_timestamp",
        "removal_timestamp",
        "retrieval_timestamp",
        "end_timestamp",
        "end_time",
    ]

    start_col = next((c for c in start_candidates if c in stations_df.columns), None)
    end_col = next((c for c in end_candidates if c in stations_df.columns), None)
    if start_col is None or end_col is None:
        return stations_df

    obs = observations_df[["device_id", ts_col]].copy()
    obs["device_id"] = obs["device_id"].astype(str).str.strip()
    obs["_obs_date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs = obs.dropna(subset=["_obs_date"])
    if obs.empty:
        return stations_df

    obs_range = (
        obs.groupby("device_id", as_index=False)
        .agg(_obs_min=("_obs_date", "min"), _obs_max=("_obs_date", "max"))
    )

    out = stations_df.copy()
    out["device_id"] = out["device_id"].astype(str).str.strip()
    out = out.merge(obs_range, on="device_id", how="left")

    start = pd.to_datetime(out[start_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    end = pd.to_datetime(out[end_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

    obs_min = pd.to_datetime(out["_obs_min"], errors="coerce")
    obs_max = pd.to_datetime(out["_obs_max"], errors="coerce")

    clipped_start = start.copy()
    clipped_end = end.copy()

    mask_min = obs_min.notna() & clipped_start.notna()
    clipped_start.loc[mask_min] = clipped_start.loc[mask_min].where(
        clipped_start.loc[mask_min] >= obs_min.loc[mask_min],
        obs_min.loc[mask_min],
    )

    mask_max = obs_max.notna() & clipped_end.notna()
    clipped_end.loc[mask_max] = clipped_end.loc[mask_max].where(
        clipped_end.loc[mask_max] <= obs_max.loc[mask_max],
        obs_max.loc[mask_max],
    )

    out[start_col] = clipped_start
    out[end_col] = clipped_end

    valid = out[start_col].notna() & out[end_col].notna() & (out[end_col] >= out[start_col])
    out = out[valid].copy()
    return out.drop(columns=["_obs_min", "_obs_max"], errors="ignore")


def _incidence_frequency_vector(observations_df: pd.DataFrame, stations_df: pd.DataFrame) -> list[int]:
    """Build iNEXT-style incidence frequency vector: [n_sampling_units, species frequencies...]."""
    all_days = _build_all_device_days(stations_df)
    n_days = len(all_days)
    if n_days == 0:
        return [0]

    ts_col = _timestamp_column(observations_df)
    species_col = "common_name" if "common_name" in observations_df.columns else "species"
    if ts_col is None or "device_id" not in observations_df.columns or species_col not in observations_df.columns:
        return [n_days]

    obs = observations_df[["device_id", ts_col, species_col]].copy()
    obs["date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs["species"] = obs[species_col].fillna("").astype(str).str.strip()
    obs = obs.dropna(subset=["date"])
    obs = obs[obs["species"] != ""]
    if obs.empty:
        return [n_days]

    obs_days = obs[["device_id", "date", "species"]].drop_duplicates()
    merged = all_days.merge(obs_days, on=["device_id", "date"], how="inner")
    if merged.empty:
        return [n_days]

    freq = (
        merged.groupby("species", as_index=False)
        .size()
        .rename(columns={"size": "fq"})
        .sort_values("species")
    )
    values = [int(n_days)] + [int(v) for v in freq["fq"].tolist() if int(v) > 0]
    return values if values else [n_days]


def _inext_curve_via_r(freq_vector: list[int], endpoint: int) -> pd.DataFrame | None:
    """Run R iNEXT to get q=0 size-based curve and confidence bounds."""
    if endpoint <= 0:
        return pd.DataFrame(columns=["x", "mean", "lower", "upper"])
    if shutil.which("Rscript") is None:
        return None

    script = """
    suppressPackageStartupMessages({
      library(iNEXT)
      library(jsonlite)
    })
    freq <- c(__FREQ__)
    out <- iNEXT(freq, endpoint=__ENDPOINT__, datatype='incidence_freq', se=TRUE, conf=0.95)
    est <- out$iNextEst$size_based
    est <- est[est$q == 0, c('t', 'qD', 'qD.LCL', 'qD.UCL')]
    names(est) <- c('x', 'mean', 'lower', 'upper')
    cat(toJSON(est, dataframe='rows', auto_unbox=TRUE))
    """.replace("__FREQ__", ",".join(str(v) for v in freq_vector)).replace("__ENDPOINT__", str(int(endpoint)))

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(script)
        r_path = fh.name

    try:
        proc = subprocess.run(
            ["Rscript", r_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        payload = proc.stdout.strip()
        if not payload:
            return None
        data = json.loads(payload)
        out = pd.DataFrame(data)
        if out.empty:
            return pd.DataFrame(columns=["x", "mean", "lower", "upper"])
        for col in ["x", "mean", "lower", "upper"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["x", "mean", "lower", "upper"]).sort_values("x").reset_index(drop=True)
        return out
    except Exception:
        return None
    finally:
        try:
            Path(r_path).unlink(missing_ok=True)
        except Exception:
            pass


def _filter_stations_for_sensor(
    stations_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    *,
    include_unobserved_stations: bool = False,
) -> pd.DataFrame:
    if stations_df.empty:
        return stations_df

    out = stations_df.copy()

    if "measurement_type" in out.columns and "measurement_type" in obs_df.columns:
        obs_types = set(obs_df["measurement_type"].dropna().astype(str).str.strip().str.lower().unique())
        if obs_types:
            station_types = out["measurement_type"].astype(str).str.strip().str.lower()
            out = out[station_types.isin(obs_types)].copy()

    # Optionally limit station effort to devices that occur in observations.
    # Keeping unobserved stations includes zero-detection effort in accumulation
    # curves, which is appropriate for effort accounting in some workflows.
    if (not include_unobserved_stations) and "device_id" in out.columns and "device_id" in obs_df.columns:
        obs_devices = set(obs_df["device_id"].dropna().astype(str).str.strip())
        if obs_devices:
            out = out[out["device_id"].astype(str).str.strip().isin(obs_devices)].copy()

    return out


def _incidence_species_accumulation_curve(
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    n_permutations: int = 200,
    random_seed: int = 42,
    use_r_inext: bool = False,
    clip_stations_to_observations: bool = True,
) -> pd.DataFrame:
    """Compute incidence-based accumulation by unique device-day effort with CI."""
    if clip_stations_to_observations:
        stations_df = _clip_station_windows_to_observation_range(stations_df, observations_df)

    # Optional R iNEXT path (disabled by default to keep plotting Python-only).
    if use_r_inext:
        freq_vector = _incidence_frequency_vector(observations_df, stations_df)
        endpoint = int(freq_vector[0]) if freq_vector else 0
        inext_curve = _inext_curve_via_r(freq_vector, endpoint=endpoint)
        if inext_curve is not None:
            return inext_curve

    all_days = _build_all_device_days(stations_df)
    if all_days.empty:
        return pd.DataFrame(columns=["x", "mean", "lower", "upper"])

    ts_col = _timestamp_column(observations_df)
    if ts_col is None or "device_id" not in observations_df.columns:
        return pd.DataFrame(
            {
                "x": np.arange(1, len(all_days) + 1),
                "mean": np.zeros(len(all_days)),
                "lower": np.zeros(len(all_days)),
                "upper": np.zeros(len(all_days)),
            }
        )

    species_col = "common_name" if "common_name" in observations_df.columns else "species"
    if species_col not in observations_df.columns:
        return pd.DataFrame(
            {
                "x": np.arange(1, len(all_days) + 1),
                "mean": np.zeros(len(all_days)),
                "lower": np.zeros(len(all_days)),
                "upper": np.zeros(len(all_days)),
            }
        )

    obs = observations_df[["device_id", ts_col, species_col]].copy()
    obs["date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs["species"] = obs[species_col].fillna("").astype(str).str.strip()
    obs = obs.dropna(subset=["date"])
    obs = obs[obs["species"] != ""]

    samples = all_days.drop_duplicates(["device_id", "date"]).reset_index(drop=True)
    if obs.empty:
        n_samples = len(samples)
        return pd.DataFrame(
            {
                "x": np.arange(1, n_samples + 1),
                "mean": np.zeros(n_samples),
                "lower": np.zeros(n_samples),
                "upper": np.zeros(n_samples),
            }
        )

    obs_unique = obs[["device_id", "date", "species"]].drop_duplicates()
    merged = samples.merge(obs_unique, on=["device_id", "date"], how="left")

    species_per_sample = (
        merged.dropna(subset=["species"])  # keep only observed incidences
        .groupby(level=0)["species"]
        .agg(lambda s: set(s.astype(str)))
        .reindex(range(len(samples)), fill_value=set())
        .tolist()
    )

    n_samples = len(species_per_sample)
    if n_samples == 0:
        return pd.DataFrame(columns=["x", "mean", "lower", "upper"])

    rng = np.random.default_rng(random_seed)
    curves = np.zeros((n_permutations, n_samples), dtype=float)

    for i in range(n_permutations):
        order = rng.permutation(n_samples)
        seen: set[str] = set()
        for j, sample_idx in enumerate(order):
            seen.update(species_per_sample[sample_idx])
            curves[i, j] = len(seen)

    mean = curves.mean(axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)

    return pd.DataFrame(
        {
            "x": np.arange(1, n_samples + 1),
            "mean": mean,
            "lower": lower,
            "upper": upper,
        }
    )


def _plot_single_accumulation_curve(
    curve: pd.DataFrame,
    *,
    effort_label: str,
    panel_letter: str,
    line_color: str = "#d88bd8",
    background_color: str = "#f1f1f1",
    grid_color: str = "#d6d6d6",
) -> plt.Figure:
    """Render a single accumulation curve with a soft ribbon style.

    Parameters
    ----------
    curve:
        DataFrame with columns ``x``, ``mean``, ``lower``, ``upper``.
    effort_label:
        X-axis label describing sampling effort.
    panel_letter:
        Label drawn at the top-left of the panel.
    line_color:
        Main line and point color.
    background_color:
        Panel and figure background color.
    grid_color:
        Gridline color.

    Returns
    -------
    matplotlib.figure.Figure
        Styled figure containing the accumulation curve.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    if curve.empty:
        ax.text(0.5, 0.5, "No accumulation data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_minimal_axes_style(ax, grid_axis=None)
        return fig

    ax.fill_between(curve["x"], curve["lower"], curve["upper"], color=line_color, alpha=0.20, linewidth=0)
    ax.plot(curve["x"], curve["mean"], color=line_color, linewidth=3.0, solid_capstyle="round")

    end_x = float(curve["x"].iloc[-1])
    end_y = float(curve["mean"].iloc[-1])
    ax.scatter([end_x], [end_y], s=120, color=line_color, edgecolor="none", zorder=5)

    x_min = max(0.0, float(curve["x"].min()))
    x_max = float(curve["x"].max())
    ax.set_xlim(x_min, x_max * 1.02 if x_max > 0 else 1.0)

    y_min = max(0.0, float(curve["lower"].min()) * 0.98)
    y_max = float(curve["upper"].max())
    if y_max <= 0:
        y_max = 1.0
    ax.set_ylim(y_min, y_max * 1.05)

    ax.set_xlabel(effort_label, fontsize=14)
    ax.set_ylabel("Species richness", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)

    ax.grid(True, which="major", color=grid_color, linewidth=1.2)
    _apply_minimal_axes_style(ax, grid_axis=None)

    ax.text(
        -0.08,
        1.02,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="#111111",
        clip_on=False,
    )
    fig.tight_layout()
    return fig


def plot_species_accumulation_curve(
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame | None = None,
    *,
    class_column: str | None = None,
    effort_label: str = "Sensor days",
    panel_letter: str = "C",
) -> dict[str, plt.Figure]:
    """Plot one incidence-based accumulation curve per taxonomic class.

    The function computes an incidence-based species accumulation curve for each
    class present in ``observations_df`` and returns a separate figure for each
    class. Curves include a smoothed 95% confidence ribbon.

    Parameters
    ----------
    observations_df:
        Observation records with at least ``device_id``, a timestamp column,
        a species/common-name column, and a class column (or equivalent).
    stations_df:
        Station deployment records used to derive sampling effort. If omitted,
        the function falls back to ``observations_df``.
    class_column:
        Optional explicit class column. When omitted, the first available class
        field among ``class``, ``class_``, ``taxonomic_class``, ``class_name``,
        or ``taxon_class`` is used.
    effort_label:
        X-axis label for effort units.
    panel_letter:
        Panel letter drawn on each figure.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Mapping of class name to figure.
    """
    if observations_df is None or observations_df.empty:
        return {}

    if stations_df is None:
        stations_df = observations_df

    if class_column is not None and class_column in observations_df.columns:
        class_series = observations_df[class_column].fillna("Unknown").astype(str).str.strip()
        class_series = class_series.mask(class_series == "", "Unknown")
    else:
        class_series = _taxonomic_class_series(observations_df)

    working = observations_df.copy()
    working["_accum_class"] = class_series

    class_order = (
        working["_accum_class"]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .index.tolist()
    )

    figures: dict[str, plt.Figure] = {}
    for class_name in class_order:
        class_df = working[working["_accum_class"] == class_name].copy()
        if class_df.empty:
            continue

        class_stations = _filter_stations_for_sensor(stations_df, class_df)
        curve = _smooth_accumulation_curve(
            _incidence_species_accumulation_curve(class_df, class_stations)
        )
        if curve.empty:
            continue

        fig = _plot_single_accumulation_curve(
            curve,
            effort_label=effort_label,
            panel_letter=panel_letter,
        )
        figures[str(class_name)] = fig

    return figures


def plot_species_accumulation(
    camera_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    stations_df: pd.DataFrame | None = None,
):
    """Plot side-by-side incidence-based accumulation curves with 95% CI."""
    camera_stations = stations_df if stations_df is not None else camera_df
    bio_stations = stations_df if stations_df is not None else bio_df
    if stations_df is not None:
        camera_stations = _filter_stations_for_sensor(stations_df, camera_df)
        bio_stations = _filter_stations_for_sensor(stations_df, bio_df)

    cam_curve = _smooth_accumulation_curve(_incidence_species_accumulation_curve(camera_df, camera_stations))
    bio_curve = _smooth_accumulation_curve(_incidence_species_accumulation_curve(bio_df, bio_stations))

    fig, axes = plt.subplots(ncols=2, figsize=(14, 5), sharey=True)

    axes[0].plot(cam_curve["x"], cam_curve["mean"], color=PLOT_COLORS[0], linewidth=2)
    axes[0].fill_between(cam_curve["x"], cam_curve["lower"], cam_curve["upper"], color=PLOT_COLORS[0], alpha=0.15)
    axes[0].text(
        -0.10,
        1.04,
        "A",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    axes[0].set_xlabel("Camera trap days")
    axes[0].set_ylabel("Species richness")
    axes[0].grid(alpha=0.25)

    axes[1].plot(bio_curve["x"], bio_curve["mean"], color=PLOT_COLORS[1], linewidth=2)
    axes[1].fill_between(bio_curve["x"], bio_curve["lower"], bio_curve["upper"], color=PLOT_COLORS[1], alpha=0.15)
    axes[1].text(
        -0.10,
        1.04,
        "B",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )
    axes[1].set_xlabel("Camera trap days")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

    plt.tight_layout()
    return fig


def project_label_reference_table(hdr, labeltype: str = "Observation") -> pd.DataFrame:
    """Convenience helper to fetch project labels as a DataFrame."""
    return get_project_labels(hdr, labeltype=labeltype, include_iucn_status=True)


def save_all_figures(
    bundle: ObservationBundle,
    output_dir: str | Path,
    *,
    top_n: int = 10,
    dpi: int = 300,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
) -> dict[str, str]:
    """Generate and save all requested figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    for key, sensor in [
        ("all_sampling_locations", "all"),
        ("camera_sampling_locations", "camera trap"),
        ("bioacoustic_sampling_locations", "bioacoustic"),
        ("edna_sampling_locations", "edna"),
    ]:
        fig = plot_stations_static(bundle.stations, sensor_type=sensor, project_boundary=project_boundary)
        path = out / f"{key}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved[key] = str(path)

    camera_timeline = plot_sensor_activity_timeline(bundle.stations, sensor_type="camera trap")
    camera_timeline_path = out / "timeline_camera_trap_activity.png"
    camera_timeline.savefig(camera_timeline_path, dpi=dpi, bbox_inches="tight")
    plt.close(camera_timeline)
    saved["timeline_camera_trap_activity"] = str(camera_timeline_path)

    bio_timeline = plot_sensor_activity_timeline(bundle.stations, sensor_type="bioacoustic")
    bio_timeline_path = out / "timeline_bioacoustic_activity.png"
    bio_timeline.savefig(bio_timeline_path, dpi=dpi, bbox_inches="tight")
    plt.close(bio_timeline)
    saved["timeline_bioacoustic_activity"] = str(bio_timeline_path)

    species_figs = plot_species_per_class(bundle.camera, bundle.bioacoustic)

    fig1_camera_path = out / "species_per_class_camera.png"
    species_figs["camera"].savefig(fig1_camera_path, dpi=dpi, bbox_inches="tight")
    plt.close(species_figs["camera"])
    saved["species_per_class_camera"] = str(fig1_camera_path)

    fig1_bio_path = out / "species_per_class_bioacoustic.png"
    species_figs["bioacoustic"].savefig(fig1_bio_path, dpi=dpi, bbox_inches="tight")
    plt.close(species_figs["bioacoustic"])
    saved["species_per_class_bioacoustic"] = str(fig1_bio_path)

    accumulation_figs = plot_species_accumulation_mammal_bird_by_sensor(
        bundle.camera,
        bundle.bioacoustic,
        bundle.stations,
    )
    acc_camera_path = out / "species_accumulation_mammal_bird_camera.png"
    accumulation_figs["camera"].savefig(acc_camera_path, dpi=dpi, bbox_inches="tight")
    plt.close(accumulation_figs["camera"])
    saved["species_accumulation_mammal_bird_camera"] = str(acc_camera_path)

    acc_bio_path = out / "species_accumulation_mammal_bird_bioacoustic.png"
    accumulation_figs["bioacoustic"].savefig(acc_bio_path, dpi=dpi, bbox_inches="tight")
    plt.close(accumulation_figs["bioacoustic"])
    saved["species_accumulation_mammal_bird_bioacoustic"] = str(acc_bio_path)

    top_figs = plot_top_species_mammal_bird_by_sensor(bundle.camera, bundle.bioacoustic, top_n=top_n)
    top_camera_path = out / "top_mammal_bird_species_camera.png"
    top_figs["camera"].savefig(top_camera_path, dpi=dpi, bbox_inches="tight")
    plt.close(top_figs["camera"])
    saved["top_mammal_bird_species_camera"] = str(top_camera_path)

    top_bio_path = out / "top_mammal_bird_species_bioacoustic.png"
    top_figs["bioacoustic"].savefig(top_bio_path, dpi=dpi, bbox_inches="tight")
    plt.close(top_figs["bioacoustic"])
    saved["top_mammal_bird_species_bioacoustic"] = str(top_bio_path)

    edna_fig = plot_edna_unique_taxa_stacked(bundle.all_species[bundle.all_species.get("measurement_type", "") == "eDNA"] if "measurement_type" in bundle.all_species.columns else pd.DataFrame())
    edna_fig_path = out / "edna_unique_taxa_stacked.png"
    edna_fig.savefig(edna_fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(edna_fig)
    saved["edna_unique_taxa_stacked"] = str(edna_fig_path)

    return saved


def save_all_tables(
    bundle: ObservationBundle,
    output_dir: str | Path,
    *,
    major_concern_top_n: int = 100,
) -> dict[str, str]:
    """Generate and save all requested tables as CSV files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sensor_summary = station_summary_table(bundle.stations)
    redlist = redlist_status_table(bundle.all_species)
    major_concern = major_concern_species_table(bundle.all_species, top_n=major_concern_top_n)
    species_class = species_per_class_table(bundle.camera, bundle.bioacoustic)

    saved: dict[str, str] = {}

    sensor_path = out / "sensor_summary.csv"
    sensor_summary.to_csv(sensor_path, index=False)
    saved["sensor_summary"] = str(sensor_path)

    redlist_path = out / "redlist_status_table.csv"
    redlist.to_csv(redlist_path, index=False)
    saved["redlist_status"] = str(redlist_path)

    concern_path = out / "major_concern_species_table.csv"
    major_concern.to_csv(concern_path, index=False)
    saved["major_concern_species"] = str(concern_path)

    class_path = out / "species_per_class_table.csv"
    species_class.to_csv(class_path, index=False)
    saved["species_per_class"] = str(class_path)

    return saved
