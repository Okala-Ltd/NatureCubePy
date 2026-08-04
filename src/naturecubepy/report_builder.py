"""High-level report asset building utilities for NatureCubePy.

Produces observation data caches, figures, tables, and short science text
sections. Delivery DOCX assembly belongs in OkalaReporter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from naturecubepy.analysis import ObservationBundle, load_project_data, save_observation_bundle
from naturecubepy.old_viz import (
    major_concern_species_table,
    save_all_figures,
    save_all_tables,
    species_per_class_table,
    station_summary_table,
)


@dataclass
class ReportBuildResult:
    """Container for generated report assets (no delivery DOCX)."""

    output_dir: str
    figures: dict[str, str]
    tables: dict[str, str]
    text_sections: dict[str, str]
    data_files: dict[str, str]
    report_markdown: str | None


def plot_edna_identification_by_class(
    class_rank_table: pd.DataFrame,
    *,
    class_col: str = "resolved_class",
    rank_col: str = "resolved_rank",
    count_col: str = "unique_taxa",
    rank_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    sort_classes_desc: bool = True,
    title: str = "Figure 6.3.2.1 Unique taxa by class and identification level",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Create stacked bar plot of eDNA unique taxa by class and taxonomic rank.

    Args:
        class_rank_table: Long-form table containing class, rank, and count columns.
        class_col: Column name containing taxonomic class labels.
        rank_col: Column name containing taxonomic rank labels.
        count_col: Column name containing unique taxa counts.
        rank_order: Optional explicit rank stacking order.
        color_map: Optional mapping from rank label to bar color.
        sort_classes_desc: Whether to sort classes by total taxa descending.
        title: Plot title.
        ax: Existing matplotlib axis to draw on. If None, a new figure is created.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, pandas.DataFrame]:
            Figure, axis, and the pivoted table used for plotting.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {class_col, rank_col, count_col}
    missing = required.difference(class_rank_table.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

    default_rank_order = ["Order", "Family", "Genus", "Species"]
    rank_order_use = rank_order if rank_order is not None else default_rank_order
    default_color_map = {
        "Order": "#2f4f4f",
        "Family": "#d9d6c4",
        "Genus": "#c8a2c8",
        "Species": "#74d3b8",
    }
    color_map_use = color_map if color_map is not None else default_color_map

    work = class_rank_table.copy()
    work[class_col] = work[class_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    work[rank_col] = work[rank_col].fillna("Unknown").astype(str).str.strip().str.title().replace("", "Unknown")
    work[count_col] = pd.to_numeric(work[count_col], errors="coerce").fillna(0)

    available_ranks = work[rank_col].dropna().unique().tolist()
    ordered_ranks = [r for r in rank_order_use if r in available_ranks] + [
        r for r in available_ranks if r not in rank_order_use
    ]

    pivot = work.pivot_table(
        index=class_col,
        columns=rank_col,
        values=count_col,
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=ordered_ranks, fill_value=0)

    if sort_classes_desc:
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 6))
    else:
        fig = ax.figure

    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")

    bottom = pd.Series(0, index=pivot.index)
    for rank in pivot.columns:
        values = pivot[rank]
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            label=rank,
            color=color_map_use.get(rank, "#bdbdbd"),
            edgecolor="#333333",
            linewidth=0.4,
        )
        bottom = bottom + values

    ax.set_ylabel("Number of unique taxa")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Identified to:", frameon=False, loc="upper left")
    fig.tight_layout()

    return fig, ax, pivot


def _species_series(df: pd.DataFrame) -> pd.Series:
    if "common_name" in df.columns:
        names = df["common_name"].fillna("").astype(str).str.strip()
        if (names != "").any():
            return names
    if "species" in df.columns:
        return df["species"].fillna("").astype(str).str.strip()
    if "label" in df.columns:
        return df["label"].fillna("").astype(str).str.strip()
    return pd.Series(dtype="object")


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


def _class_series(df: pd.DataFrame) -> pd.Series:
    for col in ["class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
        if col in df.columns:
            out = df[col].fillna("Unknown").astype(str).str.strip()
            return out.mask(out == "", "Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


def _sensor_section_text(
    sensor_title: str,
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> str:
    records = len(observations_df)
    species = _species_series(observations_df)
    species = species[species != ""]
    n_species = species.nunique()

    class_counts = _class_series(observations_df).value_counts()
    top_classes = ", ".join(f"{name} ({int(count)})" for name, count in class_counts.head(3).items())
    if top_classes == "":
        top_classes = "No class information available"

    top_species = species.value_counts().head(top_n)
    top_species_text = ", ".join(f"{name} ({int(count)})" for name, count in top_species.items())
    if top_species_text == "":
        top_species_text = "No species names available"

    ts_col = _timestamp_column(observations_df)
    if ts_col is not None:
        ts = pd.to_datetime(observations_df[ts_col], errors="coerce", utc=True).dropna()
        if not ts.empty:
            date_coverage = f"{ts.min().date()} to {ts.max().date()}"
        else:
            date_coverage = "Not available"
    else:
        date_coverage = "Not available"

    station_subset = stations_df.copy()
    if "measurement_type" in station_subset.columns:
        station_subset = station_subset[
            station_subset["measurement_type"].astype(str).str.strip().str.lower()
            == sensor_title.lower().replace(" trap", "")
        ]
    n_stations = station_subset["device_id"].nunique() if "device_id" in station_subset.columns else 0

    return "\n".join(
        [
            f"{sensor_title} data contains {records:,} records across {n_species:,} unique named taxa.",
            f"Date coverage: {date_coverage}.",
            f"Number of stations/devices: {n_stations:,}.",
            f"Most represented classes: {top_classes}.",
            f"Top {top_n} taxa by record count: {top_species_text}.",
        ]
    )


def build_core_report_text(bundle: ObservationBundle, *, top_n: int = 10) -> dict[str, str]:
    """Build core narrative text for camera trap, bioacoustic, and summary tables."""
    camera_text = _sensor_section_text("Camera", bundle.camera, bundle.stations, top_n=top_n)
    bio_text = _sensor_section_text("Bioacoustic", bundle.bioacoustic, bundle.stations, top_n=top_n)

    sensor_summary = station_summary_table(bundle.stations)
    class_summary = species_per_class_table(bundle.camera, bundle.bioacoustic)
    concern_summary = major_concern_species_table(bundle.all_species, top_n=10)

    summary_text = "\n".join(
        [
            "Summary tables prepared:",
            f"- Sensor summary rows: {len(sensor_summary):,}",
            f"- Species-per-class rows: {len(class_summary):,}",
            f"- Major concern species rows: {len(concern_summary):,}",
        ]
    )

    return {
        "camera_trap": camera_text,
        "bioacoustic": bio_text,
        "summary_tables": summary_text,
    }


def _render_report_markdown(
    *,
    figure_paths: dict[str, str],
    table_paths: dict[str, str],
    text_sections: dict[str, str],
    data_files: dict[str, str],
) -> str:
    lines: list[str] = [
        "# Biodiversity Report Assets",
        "",
        "## Camera Trap Section",
        "",
        text_sections.get("camera_trap", ""),
        "",
        "## Bioacoustic Section",
        "",
        text_sections.get("bioacoustic", ""),
        "",
        "## Summary Tables",
        "",
        text_sections.get("summary_tables", ""),
        "",
        "## Data Caches",
        "",
    ]

    for name, path in data_files.items():
        lines.append(f"- {name}: {path}")

    lines.extend(["", "## Figures", ""])
    for name, path in figure_paths.items():
        lines.append(f"- {name}: {path}")

    lines.extend(["", "## Table Files", ""])
    for name, path in table_paths.items():
        lines.append(f"- {name}: {path}")

    lines.append("")
    return "\n".join(lines)


def generate_reports(
    hdr,
    output_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    project_boundary: str | Path | None = None,
    include_iucn_status: bool = True,
    allow_missing_sources: bool = True,
    top_n: int = 10,
    write_markdown: bool = True,
) -> ReportBuildResult:
    """Generate report assets: data caches, figures, tables, and core text.

    Does not assemble a delivery DOCX — that belongs in OkalaReporter.
    """
    out = Path(output_dir)
    fig_dir = out / "figures"
    table_dir = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(data_dir) if data_dir is not None else (out / "data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_project_data(
        hdr,
        include_iucn_status=include_iucn_status,
        allow_missing_sources=allow_missing_sources,
    )
    data_files = save_observation_bundle(bundle, cache_dir)
    figure_paths = save_all_figures(bundle, fig_dir, top_n=top_n, project_boundary=project_boundary)
    table_paths = save_all_tables(bundle, table_dir)
    text_sections = build_core_report_text(bundle, top_n=top_n)

    report_md_path: str | None = None
    if write_markdown:
        report_markdown = _render_report_markdown(
            figure_paths=figure_paths,
            table_paths=table_paths,
            text_sections=text_sections,
            data_files=data_files,
        )
        report_path = out / "report_assets.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        report_md_path = str(report_path)

    return ReportBuildResult(
        output_dir=str(out),
        figures=figure_paths,
        tables=table_paths,
        text_sections=text_sections,
        data_files=data_files,
        report_markdown=report_md_path,
    )
