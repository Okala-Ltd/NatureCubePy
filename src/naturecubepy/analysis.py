"""Observation loading and data-cache helpers for report pipelines."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import warnings

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


@dataclass
class ObservationBundle:
    camera: pd.DataFrame
    bioacoustic: pd.DataFrame
    all_species: pd.DataFrame
    stations: pd.DataFrame
    edna: pd.DataFrame | None = None


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


def load_project_data(
    hdr,
    *,
    include_iucn_status: bool = True,
    allow_missing_sources: bool = True,
) -> ObservationBundle:
    """Load camera, bioacoustic, eDNA, all-species, and station datasets.

    Each data source is fetched exactly once. The IUCN map is built a single
    time and applied to all frames, avoiding the redundant triple-fetching
    that occurs when calling get_species_observations() alongside the
    individual observation functions.

    Independent network calls are executed in parallel to reduce wall-clock
    time during notebook startup.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        camera_future = executor.submit(get_camera_trap_data, hdr, include_iucn_status=False)
        bio_future = executor.submit(get_audio_observation_data, hdr, include_iucn_status=False)
        edna_future = executor.submit(get_edna_observation_data, hdr, include_iucn_status=False)
        stations_future = executor.submit(get_station_info, hdr, None)
        iucn_future = executor.submit(build_iucn_map, hdr) if include_iucn_status else None

        source_futures = {
            "camera": camera_future,
            "bioacoustic": bio_future,
            "edna": edna_future,
            "stations": stations_future,
        }
        source_data: dict[str, pd.DataFrame] = {}
        for name, future in source_futures.items():
            try:
                source_data[name] = future.result()
            except Exception as exc:
                if not allow_missing_sources:
                    raise
                fallback = _empty_stations_frame() if name == "stations" else _empty_observation_frame()
                source_data[name] = fallback
                warnings.warn(
                    f"{name} source unavailable; continuing with empty data. Original error: {exc}",
                    stacklevel=2,
                )

        camera = source_data["camera"]
        bioacoustic = source_data["bioacoustic"]
        edna = source_data["edna"]
        stations = source_data["stations"]

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

        if include_iucn_status and iucn_map is not None:
            camera = enrich_with_iucn_status(camera, iucn_map)
            bioacoustic = enrich_with_iucn_status(bioacoustic, iucn_map)
            edna = enrich_with_iucn_status(edna, iucn_map)
        else:
            camera = _ensure_iucn_column(camera)
            bioacoustic = _ensure_iucn_column(bioacoustic)
            edna = _ensure_iucn_column(edna)

    frames = [
        normalise_species_frame(camera, "image"),
        normalise_species_frame(bioacoustic, "audio"),
        normalise_species_frame(edna, "eDNA"),
    ]
    non_empty = [f for f in frames if not f.empty]
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
    )


def save_observation_bundle(bundle: ObservationBundle, data_dir: str | Path) -> dict[str, str]:
    """Persist observation frames as CSV caches under ``data_dir``."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    frames = {
        "camera": bundle.camera,
        "bioacoustic": bundle.bioacoustic,
        "all_species": bundle.all_species,
        "stations": bundle.stations,
    }
    if bundle.edna is not None:
        frames["edna"] = bundle.edna

    for name, frame in frames.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        saved[name] = str(path)
    return saved
