"""Observation loading, summary tables, and project asset export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd

from naturecubepy.api import (
    build_iucn_map,
    enrich_with_iucn_status,
    get_audio_observation_data,
    get_camera_trap_data,
    get_edna_observation_data,
    get_station_info,
    normalise_species_frame,
)
from naturecubepy.schema import SPECIES_OBS_CORE_COLUMNS


DEFAULT_SENSOR_TYPES = ("camera", "bioacoustic", "edna")

# Canonical sensor key -> (API loader, species-frame datatype label)
_SENSOR_OBS_LOADERS = {
    "camera": (get_camera_trap_data, "image"),
    "bioacoustic": (get_audio_observation_data, "audio"),
    "edna": (get_edna_observation_data, "eDNA"),
}


@dataclass
class ObservationBundle:
    camera: pd.DataFrame
    bioacoustic: pd.DataFrame
    all_species: pd.DataFrame
    stations: pd.DataFrame
    edna: pd.DataFrame | None = None
    sensor_types: tuple[str, ...] = DEFAULT_SENSOR_TYPES


def _ensure_iucn_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "iucn_redlist_status" not in out.columns:
        out["iucn_redlist_status"] = pd.NA
    return out


def _empty_observation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "species",
            "common_name",
            "label",
            "class",
            "measurement_type",
            "timestamp",
            "iucn_redlist_status",
        ]
    )


def _empty_stations_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "device_id",
            "measurement_type",
            "lon",
            "lat",
            "site",
        ]
    )


def normalise_sensor_types(sensor_types: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Map config/aliases to canonical sensor keys: camera, bioacoustic, edna."""
    if not sensor_types:
        return DEFAULT_SENSOR_TYPES

    aliases = {
        "camera": "camera",
        "camera trap": "camera",
        "image": "camera",
        "bioacoustic": "bioacoustic",
        "bio": "bioacoustic",
        "audio": "bioacoustic",
        "edna": "edna",
        "e dna": "edna",
        "eDNA": "edna",
    }
    out: list[str] = []
    for raw in sensor_types:
        key = aliases.get(str(raw).strip().lower().replace("_", " "), None)
        if key is None:
            warnings.warn(f"Unknown sensor type '{raw}'; ignoring.", stacklevel=2)
            continue
        if key not in out:
            out.append(key)
    return tuple(out) if out else DEFAULT_SENSOR_TYPES


def _fetch_source(
    name: str,
    future,
    *,
    allow_missing_sources: bool,
    empty_factory,
) -> pd.DataFrame:
    try:
        return future.result()
    except Exception as exc:
        if not allow_missing_sources:
            raise
        warnings.warn(
            f"{name} source unavailable; continuing with empty data. Original error: {exc}",
            stacklevel=3,
        )
        return empty_factory()


def load_project_data(
    hdr,
    *,
    sensor_types: list[str] | tuple[str, ...] | None = None,
    include_iucn_status: bool = True,
    allow_missing_sources: bool = True,
) -> ObservationBundle:
    """Load observation + station data for the requested sensor types only.

    Uses per-sensor API helpers (``get_camera_trap_data``,
    ``get_audio_observation_data``, ``get_edna_observation_data``,
    ``get_station_info``) and only schedules network work for sensors listed
    in ``sensor_types`` (default: all three).
    """
    selected = normalise_sensor_types(sensor_types)

    with ThreadPoolExecutor(max_workers=max(2, len(selected) + 2)) as executor:
        obs_futures: dict[str, object] = {}
        for sensor in selected:
            loader, _ = _SENSOR_OBS_LOADERS[sensor]
            obs_futures[sensor] = executor.submit(loader, hdr, include_iucn_status=False)

        # Station metadata per selected sensor type (API accepts camera/bioacoustic/edna).
        station_futures = {
            sensor: executor.submit(get_station_info, hdr, sensor) for sensor in selected
        }
        iucn_future = executor.submit(build_iucn_map, hdr) if include_iucn_status else None

        obs_data: dict[str, pd.DataFrame] = {}
        for sensor, future in obs_futures.items():
            obs_data[sensor] = _fetch_source(
                sensor,
                future,
                allow_missing_sources=allow_missing_sources,
                empty_factory=_empty_observation_frame,
            )

        stations_by_sensor: dict[str, pd.DataFrame] = {}
        station_frames: list[pd.DataFrame] = []
        for sensor, future in station_futures.items():
            frame = _fetch_source(
                f"{sensor}_stations",
                future,
                allow_missing_sources=allow_missing_sources,
                empty_factory=_empty_stations_frame,
            )
            stations_by_sensor[sensor] = frame
            if not frame.empty:
                station_frames.append(frame)

        if station_frames:
            stations = pd.concat(station_frames, ignore_index=True, sort=False)
            if "device_id" in stations.columns:
                stations = stations.drop_duplicates(subset=["device_id"], keep="first")
        else:
            stations = _empty_stations_frame()

        iucn_map = None
        if include_iucn_status and iucn_future is not None:
            try:
                iucn_map = iucn_future.result()
            except Exception as exc:
                if not allow_missing_sources:
                    raise
                warnings.warn(
                    f"IUCN status source unavailable; proceeding without enrichment. Original error: {exc}",
                    stacklevel=2,
                )
                iucn_map = None

    # Drop sensors with neither observations nor stations so downstream export
    # skips empty figures / tables / CSV caches.
    kept: list[str] = []
    for sensor in selected:
        obs = obs_data.get(sensor, _empty_observation_frame())
        st = stations_by_sensor.get(sensor, _empty_stations_frame())
        if (obs is not None and not obs.empty) or (st is not None and not st.empty):
            kept.append(sensor)
        else:
            print(
                f"Warning: no {sensor} data found; "
                "skipping figures, tables, and caches for this sensor."
            )
    selected = tuple(kept)

    camera = obs_data.get("camera", _empty_observation_frame())
    bioacoustic = obs_data.get("bioacoustic", _empty_observation_frame())
    edna = obs_data.get("edna") if "edna" in selected else None

    if include_iucn_status and iucn_map is not None:
        camera = enrich_with_iucn_status(camera, iucn_map)
        bioacoustic = enrich_with_iucn_status(bioacoustic, iucn_map)
        if edna is not None:
            edna = enrich_with_iucn_status(edna, iucn_map)
    else:
        camera = _ensure_iucn_column(camera)
        bioacoustic = _ensure_iucn_column(bioacoustic)
        if edna is not None:
            edna = _ensure_iucn_column(edna)

    species_frames: list[pd.DataFrame] = []
    for sensor in selected:
        frame = obs_data.get(sensor, _empty_observation_frame())
        _, dtype_label = _SENSOR_OBS_LOADERS[sensor]
        species_frames.append(normalise_species_frame(frame, dtype_label))
    non_empty = [f for f in species_frames if not f.empty]
    all_species = (
        pd.concat(non_empty, ignore_index=True, sort=False)
        if non_empty
        else pd.DataFrame(columns=SPECIES_OBS_CORE_COLUMNS)
    )

    return ObservationBundle(
        camera=camera,
        bioacoustic=bioacoustic,
        all_species=all_species,
        stations=stations,
        edna=edna,
        sensor_types=selected,
    )


def save_observation_bundle(bundle: ObservationBundle, data_dir: str | Path) -> dict[str, str]:
    """Persist observation frames as CSV caches under ``data_dir``.

    Empty frames are skipped (with a terminal warning) and any stale CSV for
    that name is removed so prior empty artefacts do not linger.
    """
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    frames: dict[str, pd.DataFrame] = {
        "all_species": bundle.all_species,
        "stations": bundle.stations,
    }
    if "camera" in bundle.sensor_types:
        frames["camera"] = bundle.camera
    if "bioacoustic" in bundle.sensor_types:
        frames["bioacoustic"] = bundle.bioacoustic
    if "edna" in bundle.sensor_types and bundle.edna is not None:
        frames["edna"] = bundle.edna

    for name, frame in frames.items():
        path = out / f"{name}.csv"
        if frame is None or frame.empty:
            print(f"Warning: no rows for {name}; not writing empty CSV cache.")
            if path.exists():
                path.unlink()
            continue
        frame.to_csv(path, index=False)
        saved[name] = str(path)

    # Remove stale sensor caches for sensors that were requested but had no data.
    for sensor in ("camera", "bioacoustic", "edna"):
        if sensor in frames:
            continue
        path = out / f"{sensor}.csv"
        if path.exists():
            print(f"Warning: removing stale empty/unused cache {path.name}.")
            path.unlink()
    return saved

# ---------------------------------------------------------------------------
# Summary tables / dataset counts
# ---------------------------------------------------------------------------

CONCERN_STATUSES = {
    "Critically Endangered",
    "Endangered",
    "Vulnerable",
    "Near Threatened",
}

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

def save_all_tables(
    bundle: ObservationBundle,
    output_dir: str | Path,
    *,
    major_concern_top_n: int = 100,
    filename_prefix: str | None = None,
) -> dict[str, str]:
    """Generate and save summary tables as CSV files.

    When ``filename_prefix`` is set, files are named ``{prefix}_{stem}.csv``;
    dict keys remain the logical names (e.g. ``sensor_summary``).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    tables = {
        "sensor_summary": (f"{prefix}sensor_summary.csv", station_summary_table(bundle.stations)),
        "redlist_status": (f"{prefix}redlist_status_table.csv", redlist_status_table(bundle.all_species)),
        "major_concern_species": (
            f"{prefix}major_concern_species_table.csv",
            major_concern_species_table(bundle.all_species, top_n=major_concern_top_n),
        ),
        "species_per_class": (
            f"{prefix}species_per_class_table.csv",
            species_per_class_table(
                bundle.camera if "camera" in bundle.sensor_types else pd.DataFrame(),
                bundle.bioacoustic if "bioacoustic" in bundle.sensor_types else pd.DataFrame(),
            ),
        ),
    }

    saved: dict[str, str] = {}
    for key, (filename, table) in tables.items():
        path = out / filename
        if table is None or table.empty:
            print(f"Warning: no rows for table '{key}'; not writing empty CSV.")
            if path.exists():
                path.unlink()
            continue
        table.to_csv(path, index=False)
        saved[key] = str(path)

    return saved


@dataclass
class ProjectAssetExport:
    """Paths produced by :func:`export_project_assets` (no report narrative)."""

    output_dir: str
    figures: dict[str, str]
    tables: dict[str, str]
    data_files: dict[str, str]
    bundle: ObservationBundle


def export_project_assets(
    hdr,
    output_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    sensor_types: list[str] | tuple[str, ...] | None = None,
    include_iucn_status: bool = True,
    allow_missing_sources: bool = True,
    top_n: int = 10,
    logo_path: str | Path | None = None,
    filename_prefix: str | None = None,
) -> ProjectAssetExport:
    """Load a project and export data caches, figures, and summary tables.

    This is the public entry point for bulk asset export. Report narrative,
    branding, and DOCX/PDF assembly belong in OkalaReporter (or other apps).

    Pass ``sensor_types`` (e.g. ``["camera", "bioacoustic"]``) to skip unused
    sensor API calls and figures. When ``filename_prefix`` is set, figure and
    table files are named ``{prefix}_{logical_name}.{ext}``.
    """
    # Local import avoids a circular dependency at module load time.
    from naturecubepy.viz import save_all_figures

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
        sensor_types=sensor_types,
        include_iucn_status=include_iucn_status,
        allow_missing_sources=allow_missing_sources,
    )
    data_files = save_observation_bundle(bundle, cache_dir)
    figure_paths = save_all_figures(
        bundle,
        fig_dir,
        top_n=top_n,
        project_boundary=project_boundary,
        sensor_types=bundle.sensor_types,
        logo_path=logo_path,
        filename_prefix=filename_prefix,
    )
    table_paths = save_all_tables(
        bundle,
        table_dir,
        filename_prefix=filename_prefix,
    )

    return ProjectAssetExport(
        output_dir=str(out),
        figures=figure_paths,
        tables=table_paths,
        data_files=data_files,
        bundle=bundle,
    )
