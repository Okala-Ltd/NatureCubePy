"""
Core API wrapper functions for the Okala dashboard.

This module provides functions for authenticating with the Okala API and
interacting with project data including stations, media assets, labels,
eDNA records, and timestamps.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Literal

import geopandas as gpd
import httpx
import pandas as pd

from naturecubepy.schema import (
    AuthHeaders,
    DataTypes,
    GetProjectGeometryResponse,
    IUCNSpeciesLabelInput,
    Label,
    LabelType,
    MediaRecordAPIFlat,
    MediaTimestampUpdate,
    SegmentRecordAPIFlat,
    SpeciesTable,
    StationResponseAPI,
    TimestampUpdateResponse,
    eDNAUploadResponse,
    eDNAUploadSchema,
)

_PROD_URL = "https://naturecube.io/api/"
_DEV_URL = "http://127.0.0.1:8000/api/"
_DEFAULT_TIMEOUT = 180.0
_STATIONS_PAGE_SIZE = 1000
_MEDIA_FETCH_CHUNK_SIZE = 50
_RATE_LIMIT_MAX_RETRIES = 6
_RATE_LIMIT_BACKOFF_SECONDS = 2.0
_TIMEOUT_MAX_RETRIES = 3
_TIMEOUT_BACKOFF_SECONDS = 2.0
_INTER_REQUEST_DELAY_SECONDS = 0.3

_MEASUREMENT_TYPE_TO_DATATYPES = {
    "camera": ["image", "video"],
    "bioacoustic": ["audio"],
    "edna": ["eDNA"],
}

_CLASS_COLUMN_NAMES = ["class", "class_", "taxonomic_class", "class_name", "taxon_class"]

# Columns guaranteed to appear in the unified species observation DataFrame.
_SPECIES_OBS_CORE_COLUMNS: list[str] = [
    "project_system_record_id",
    "device_id",
    "data_type",
    "measurement_type",
    "latitude",
    "longitude",
    "label",
    "label_id",
    "common_name",
    "species",
    "class",
    "genus",
    "family",
    "order",
]

_STATION_LOOKUP_COLUMNS: list[str] = [
    "project_system_record_id",
    "device_id",
    "data_type",
    "measurement_type",
    "latitude",
    "longitude",
]

_EMPTY_OBSERVATION_FRAME = pd.DataFrame(columns=_SPECIES_OBS_CORE_COLUMNS)


def _extract_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalise list/dict API payloads to a list of row dictionaries."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("table", "rows", "data", "items", "results"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]

    return []


def _extract_total_count(payload: Any) -> int | None:
    """Extract total row count from paginated payloads when available."""
    if not isinstance(payload, dict):
        return None

    for key in ("total", "total_count", "totalCount", "count", "recordsTotal"):
        if key not in payload:
            continue
        try:
            total = int(payload[key])
        except (TypeError, ValueError):
            continue
        if total >= 0:
            return total
    return None


def _standardize_taxonomic_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose public taxonomic class column is always named ``class``."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "class" not in out.columns:
        for source_name in _CLASS_COLUMN_NAMES[1:]:
            if source_name in out.columns:
                out = out.rename(columns={source_name: "class"})
                break

    if "class_" in out.columns:
        out = out.drop(columns=["class_"])

    return out


def _should_fetch_next_page(payload: Any, fetched_count: int, offset: int, limit: int) -> bool:
    """Determine whether another paginated request is needed."""
    total = _extract_total_count(payload)
    if total is not None:
        return (offset + fetched_count) < total
    return fetched_count >= limit


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_key() -> str:
    """Retrieve the API key from the ``OKALA_API_KEY`` environment variable.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    EnvironmentError
        If ``OKALA_API_KEY`` is not set.

    Examples
    --------
    >>> import os
    >>> os.environ["OKALA_API_KEY"] = "mykey"
    >>> get_key()
    'mykey'
    """
    api_key = os.environ.get("OKALA_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OKALA_API_KEY environment variable not set.")
    return api_key


def auth_headers(api_key: str, okala_url: str = _PROD_URL) -> AuthHeaders:
    """Create an authentication context for the production Okala API.

    Parameters
    ----------
    api_key:
        A valid Okala project API key.
    okala_url:
        Base URL for the Okala API. Defaults to the production endpoint.

    Returns
    -------
    AuthHeaders
        An ``AuthHeaders`` object with ``key`` and ``root`` attributes.

    Examples
    --------
    >>> hdr = auth_headers("mykey")
    >>> hdr.root
    'https://naturecube.io/api/'
    """
    return AuthHeaders(key=api_key, root=okala_url.rstrip("/") + "/")


def auth_headers_dev(api_key: str, okala_url: str = _DEV_URL) -> AuthHeaders:
    """Create an authentication context for the development Okala API.

    Parameters
    ----------
    api_key:
        A valid Okala project API key.
    okala_url:
        Base URL for the Okala dev API. Defaults to the development endpoint.

    Returns
    -------
    AuthHeaders
        An ``AuthHeaders`` object with ``key`` and ``root`` attributes.

    Examples
    --------
    >>> hdr = auth_headers_dev("mykey")
    >>> hdr.root
    'http://127.0.0.1:8000/api/'
    """
    return AuthHeaders(key=api_key, root=okala_url.rstrip("/") + "/")


# ---------------------------------------------------------------------------
# Project & stations
# ---------------------------------------------------------------------------


def get_project(hdr: AuthHeaders, timeout: float = _DEFAULT_TIMEOUT) -> GetProjectGeometryResponse:
    """Retrieve and display the active project associated with the API key.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    GetProjectGeometryResponse
        The project geometry response containing boundary, ROIs, and locations.

    Examples
    --------
    >>> project = get_project(hdr)  # doctest: +SKIP
    Setting your active project as - My Project
    """
    url = f"{hdr.root}getProject/{hdr.key}"
    print("Retrieving project data...")
    response = httpx.get(url, timeout=timeout)
    print(f"Received response with status code {response.status_code}")
    response.raise_for_status()
    data = response.json()
    print("Project data retrieved successfully")
    project_name = data["boundary"]["features"][0]["properties"]["project_name"]
    print(f"Setting your active project as - {project_name}")
    return GetProjectGeometryResponse.model_validate(data)


def _normalise_measurement_type(measurement_type: str | None) -> str | None:
    """Normalise measurement type input to internal lowercase keys."""
    if measurement_type is None:
        return None
    mt = str(measurement_type).strip().lower()
    if mt == "all":
        return None
    if mt == "edna":
        return "edna"
    return mt


def _get_station_features_raw(hdr: AuthHeaders, datatype: str) -> list[dict[str, Any]]:
    """Fetch all raw GeoJSON features for a datatype endpoint, handling pagination."""
    url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
    offset = 0
    all_features: list[dict[str, Any]] = []

    while True:
        params = {"offset": offset, "limit": _STATIONS_PAGE_SIZE}
        response = httpx.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not features:
            break
        all_features.extend(features)
        if len(features) < _STATIONS_PAGE_SIZE:
            break
        offset += _STATIONS_PAGE_SIZE

    return all_features


def _fetch_stations_for_datatype(hdr: AuthHeaders, datatype: str) -> gpd.GeoDataFrame:
    """Fetch all station rows for a datatype as a GeoDataFrame (paginated)."""
    features = _get_station_features_raw(hdr, datatype)
    if not features:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame.from_features(features)


def _filter_stations_by_measurement_type(gdf: gpd.GeoDataFrame, measurement_type: str) -> gpd.GeoDataFrame:
    """Filter station rows by measurement type when the column exists."""
    if "measurement_type" not in gdf.columns:
        return gdf
    measurement_series = gdf["measurement_type"].astype(str).str.strip().str.lower()
    return gdf[measurement_series == measurement_type]


def _filter_stations_by_datatype(stations: gpd.GeoDataFrame, datatype: str) -> gpd.GeoDataFrame:
    """Filter station rows by datatype when available.

    Some API responses include mixed datatypes in a single payload, so this
    guard ensures each downstream fetch only uses station IDs for the target
    datatype.
    """
    if "data_type" not in stations.columns:
        return stations
    dtype_series = stations["data_type"].astype(str).str.strip().str.lower()
    return stations[dtype_series == datatype.lower()]


def _filter_stations_with_records(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter stations to rows that report at least one record when available."""
    if "record_count" not in stations.columns:
        return stations
    counts = pd.to_numeric(stations["record_count"], errors="coerce").fillna(0)
    return stations[counts > 0]


def get_station_info(hdr: AuthHeaders, measurement_type: str | None) -> gpd.GeoDataFrame:
    """Retrieve all station metadata for a project, optionally filtered by measurement type.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    measurement_type:
        One of ``"camera"``, ``"bioacoustic"``, ``"edna"``, or None/"all" for all stations.
        ``"camera"`` fetches and combines both ``"image"`` and ``"video"`` stations.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing station metadata and geometry.

    Raises
    ------
    ValueError
        If ``measurement_type`` is not a supported value.

    Examples
    --------
    >>> stations = get_station_info(hdr, measurement_type="camera")  # doctest: +SKIP
    >>> all_stations = get_station_info(hdr, measurement_type=None)  # doctest: +SKIP
    """
    normalised_mt = _normalise_measurement_type(measurement_type)
    if normalised_mt is None:
        # Fetch all supported endpoint datatypes and concatenate.
        dfs = []
        for mt, datatypes in _MEASUREMENT_TYPE_TO_DATATYPES.items():
            for datatype in datatypes:
                gdf = _fetch_stations_for_datatype(hdr, datatype)
                dfs.append(_filter_stations_by_measurement_type(gdf, mt))
        if dfs:
            return gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
        return gpd.GeoDataFrame()

    if normalised_mt not in _MEASUREMENT_TYPE_TO_DATATYPES:
        raise ValueError(
            f"Invalid measurement_type: {measurement_type!r}. "
            f"Must be one of {list(_MEASUREMENT_TYPE_TO_DATATYPES)} or None/'all'."
        )

    dfs = []
    for datatype in _MEASUREMENT_TYPE_TO_DATATYPES[normalised_mt]:
        gdf = _fetch_stations_for_datatype(hdr, datatype)
        dfs.append(_filter_stations_by_measurement_type(gdf, normalised_mt))
    if dfs:
        return gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
    return gpd.GeoDataFrame()


def get_stations_typed(hdr: AuthHeaders, datatype: DataTypes) -> StationResponseAPI:
    """Retrieve all station metadata for a project as a typed response.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.

    Returns
    -------
    StationResponseAPI
        A typed response containing station features.

    Examples
    --------
    >>> stations = get_stations_typed(hdr, "video")  # doctest: +SKIP
    """
    features = _get_station_features_raw(hdr, datatype)
    return StationResponseAPI.model_validate({"type": "FeatureCollection", "features": features})


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


# ---------------------------------------------------------------------------
# Media assets & segments
# ---------------------------------------------------------------------------


def get_media_assets(

    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
    limit: int = 1000,
    offset: int = 0,
) -> list[MediaRecordAPIFlat]:
    """Retrieve media assets for given project system record IDs, with pagination support.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    psr_ids:
        List of project system record IDs.
    limit:
        Maximum number of records per page (default 1000).
    offset:
        Starting offset (default 0).

    Returns
    -------
    list[MediaRecordAPIFlat]
        A list of media records with full details.

    Examples
    --------
    >>> assets = get_media_assets(hdr, "video", psr_ids=[123])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}"
    all_results = []
    current_offset = offset
    while True:
        params = {"offset": current_offset, "limit": limit}
        response = httpx.post(url, json=psr_ids, params=params, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        items = _extract_rows_from_payload(payload)
        if not items:
            break
        all_results.extend([MediaRecordAPIFlat.model_validate(item) for item in items])
        if not _should_fetch_next_page(payload, fetched_count=len(items), offset=current_offset, limit=limit):
            break
        current_offset += len(items)
    return all_results


def _fetch_paginated_rows_for_psr_ids(
    url: str,
    project_system_record_ids: int | list[int],
    *,
    error_label: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch paginated rows for PSR IDs using sequential chunked requests with 429 backoff."""
    psr_ids = _normalise_psr_ids(project_system_record_ids)
    chunk_size = _MEDIA_FETCH_CHUNK_SIZE
    limit = 1000
    all_results: list[dict[str, Any]] = []

    for i in range(0, len(psr_ids), chunk_size):
        chunk = psr_ids[i:i + chunk_size]
        offset = 0

        while True:
            params = {"offset": offset, "limit": limit}
            response: httpx.Response | None = None

            for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    response = httpx.post(url, json=chunk, params=params, timeout=_DEFAULT_TIMEOUT)
                    if response.status_code == 429:
                        if attempt >= _RATE_LIMIT_MAX_RETRIES:
                            response.raise_for_status()
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after is not None else 0.0
                        except (TypeError, ValueError):
                            delay = 0.0
                        if delay <= 0:
                            delay = _RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt)
                        if error_label is not None:
                            print(f"Rate limited (attempt {attempt + 1}), waiting {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    response.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt >= _TIMEOUT_MAX_RETRIES:
                        if error_label is not None:
                            print(f"Timeout fetching {error_label} after {_TIMEOUT_MAX_RETRIES + 1} attempts.")
                        raise
                    delay = _TIMEOUT_BACKOFF_SECONDS * (2 ** attempt)
                    time.sleep(delay)
                except httpx.HTTPStatusError:
                    raise
                except Exception as exc:
                    if error_label is not None:
                        print(
                            f"Error fetching {error_label}: {exc}\n"
                            f"Payload: {chunk}\n"
                            f"Response: {getattr(response, 'text', '')}"
                        )
                    raise

            if response is None:
                break

            payload = response.json()
            items = _extract_rows_from_payload(payload)
            if not items:
                break

            all_results.extend(items)
            if not _should_fetch_next_page(payload, fetched_count=len(items), offset=offset, limit=limit):
                break
            offset += len(items)

        # Polite delay between chunks to stay under the rate limit.
        if i + chunk_size < len(psr_ids):
            time.sleep(_INTER_REQUEST_DELAY_SECONDS)

    return all_results


def get_media_assets_df(
    hdr: AuthHeaders,
    datatype: DataTypes,
    project_system_record_ids: int | list[int],
) -> pd.DataFrame:
    """Retrieve media assets as a DataFrame for given project system record IDs.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    project_system_record_ids:
        A project system record ID or a list of IDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of media assets.

    Examples
    --------
    >>> assets = get_media_assets_df(hdr, "video", project_system_record_ids=[123, 456])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}"
    rows = _fetch_paginated_rows_for_psr_ids(
        url,
        project_system_record_ids,
        error_label="media assets",
    )
    return pd.DataFrame(rows)


def get_media_segments(
    hdr: AuthHeaders,
    datatype: DataTypes,
    project_system_record_ids: int | list[int],
) -> pd.DataFrame:
    """Retrieve media segments for one or more project system record IDs.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    project_system_record_ids:
        A project system record ID or a list of IDs.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of media segments.

    Examples
    --------
    >>> segments = get_media_segments(hdr, "audio", project_system_record_ids=[123, 456])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaSegments/{datatype}/{hdr.key}"
    rows = _fetch_paginated_rows_for_psr_ids(url, project_system_record_ids)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observation data helpers
# ---------------------------------------------------------------------------


def _normalise_psr_ids(project_system_record_ids: int | list[int]) -> list[int]:
    """Validate and normalise project system record IDs to a list of positive ints."""
    ids = [project_system_record_ids] if isinstance(project_system_record_ids, int) else list(project_system_record_ids)
    if not ids:
        raise ValueError("project_system_record_ids must contain at least one ID.")
    try:
        normalised = [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("project_system_record_ids must contain integers.") from exc
    if any(i <= 0 for i in normalised):
        raise ValueError("project_system_record_ids must contain positive integers.")
    return normalised


def _build_station_lookup(stations: gpd.GeoDataFrame, data_type: str) -> pd.DataFrame:
    """Build a flat station lookup table with location metadata."""
    if "project_system_record_id" not in stations.columns:
        raise ValueError(
            "Station data does not contain 'project_system_record_id'. "
            "Cannot link media assets to stations."
        )
    lookup = stations.dropna(subset=["project_system_record_id"]).copy()
    if lookup.empty:
        return pd.DataFrame(columns=_STATION_LOOKUP_COLUMNS)

    lookup["project_system_record_id"] = lookup["project_system_record_id"].astype(int)
    lookup["latitude"] = lookup.geometry.y
    lookup["longitude"] = lookup.geometry.x
    if "data_type" not in lookup.columns:
        lookup["data_type"] = data_type

    available = [c for c in _STATION_LOOKUP_COLUMNS if c in lookup.columns]
    return lookup[available].drop_duplicates(subset=["project_system_record_id"])


def _resolve_psr_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``project_system_record_id`` exists, falling back to the FK alias."""
    if "project_system_record_id" not in df.columns:
        if "project_system_record_id_fk" in df.columns:
            df = df.copy()
            df["project_system_record_id"] = df["project_system_record_id_fk"]
        else:
            raise ValueError(
                "Media asset data does not contain a project system record identifier. "
                "Expected 'project_system_record_id' or 'project_system_record_id_fk'."
            )
    df = df.copy()
    df["project_system_record_id"] = pd.to_numeric(df["project_system_record_id"], errors="raise").astype(int)
    return df


def _merge_segments(media_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join segment columns onto media rows, preferring segment values.

    Some backends return overlapping fields (for example ``label`` and
    ``species``) in both media-assets and segment payloads. Segment rows can
    contain newer/verified values, so segment values should override media
    values when present.
    """
    if segments_df.empty:
        return media_df

    segment_unique = segments_df.drop_duplicates(subset=["segment_record_id"])
    merged = media_df.merge(
        segment_unique,
        on="segment_record_id",
        how="left",
        suffixes=("", "__segment"),
    )

    for col in segment_unique.columns:
        if col == "segment_record_id":
            continue

        segment_col = f"{col}__segment"
        if segment_col not in merged.columns:
            continue

        if col in merged.columns:
            merged[col] = merged[segment_col].combine_first(merged[col])
        else:
            merged[col] = merged[segment_col]
        merged = merged.drop(columns=[segment_col])

    return merged


def _merge_station_lookup(media_df: pd.DataFrame, station_lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-join station location metadata onto a media DataFrame."""
    station_cols = ["project_system_record_id"] + [
        c for c in station_lookup.columns if c not in media_df.columns
    ]
    return media_df.merge(station_lookup[station_cols], on="project_system_record_id", how="left")


def _fetch_and_merge_media(
    hdr: AuthHeaders,
    datatype: DataTypes,
    station_lookup: pd.DataFrame,
    data_type_fallback: str,
) -> pd.DataFrame:
    """Fetch media assets, merge with station lookup, and return a flat DataFrame.

    Media rows are fetched in one batched path, then optional segment fields
    are joined in one batched pass.
    """
    psr_ids = station_lookup["project_system_record_id"].tolist()
    media_df = get_media_assets_df(hdr, datatype, project_system_record_ids=psr_ids)
    if media_df.empty:
        return pd.DataFrame()

    media_df = _resolve_psr_id_column(media_df)
    segments_df = get_media_segments(hdr, datatype, project_system_record_ids=psr_ids)
    media_df = _merge_segments(media_df, segments_df)
    merged = _merge_station_lookup(media_df, station_lookup)

    if "data_type" not in merged.columns:
        merged["data_type"] = data_type_fallback

    return merged


def _fetch_and_merge_edna_assets(hdr: AuthHeaders, station_lookup: pd.DataFrame) -> pd.DataFrame:
    """Fetch eDNA assets for a station lookup and merge location metadata."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for psr_id in station_lookup["project_system_record_id"].tolist():
        try:
            assets = get_edna_assets(hdr, project_system_record_id=psr_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {404, 500}:
                # The backend may return 500 for stations without eDNA rows.
                # Treat these as no-data stations and continue.
                continue
            msg = (
                f"eDNA assets API returned HTTP {exc.response.status_code} for PSR {psr_id}. "
                "Species labels will be unavailable for this station."
            )
            errors.append(msg)
            continue
        except Exception as exc:
            errors.append(f"Failed to fetch eDNA assets for PSR {psr_id}: {exc}")
            continue
        if not assets.empty:
            assets = assets.copy()
            assets["project_system_record_id"] = psr_id
            frames.append(assets)

    if not frames:
        if errors:
            print(
                f"Warning: could not retrieve eDNA species labels ({len(errors)} station(s) failed). "
                f"First error: {errors[0]}"
            )
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _merge_station_lookup(merged, station_lookup)
    if "data_type" not in merged.columns:
        merged["data_type"] = "eDNA"
    return merged


def _fetch_observations_for_datatype(
    hdr: AuthHeaders,
    datatype: str,
    stations: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch observation rows for a single datatype and attach station metadata."""
    scoped_stations = _filter_stations_by_datatype(stations, datatype)
    if str(datatype).lower() == "edna":
        scoped_stations = _filter_stations_with_records(scoped_stations)
    station_lookup = _build_station_lookup(scoped_stations, datatype)
    if station_lookup.empty:
        return pd.DataFrame(), station_lookup

    if str(datatype).lower() == "edna":
        merged = _fetch_and_merge_edna_assets(hdr, station_lookup)
    else:
        merged = _fetch_and_merge_media(hdr, datatype, station_lookup, str(datatype))
    return merged, station_lookup


# ---------------------------------------------------------------------------
# High-level observation retrieval
# ---------------------------------------------------------------------------


def get_camera_trap_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve merged camera trap media rows for all image and video stations.

    Both image and video camera trap data are always returned together.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    include_iucn_status:
        If ``True``, add an ``iucn_redlist_status`` column joined on ``species``.

    Returns
    -------
    pandas.DataFrame
        A validated DataFrame containing merged camera trap media rows, with an
        optional ``iucn_redlist_status`` column when ``include_iucn_status=True``.

    Examples
    --------
    >>> df = get_camera_trap_data(hdr)  # doctest: +SKIP
    >>> df = get_camera_trap_data(hdr, include_iucn_status=True)  # doctest: +SKIP
    >>> sorted(df["data_type"].unique())
    ['image', 'video']
    """
    stations = get_station_info(hdr, "camera")
    frames: list[pd.DataFrame] = []

    for datatype in _MEASUREMENT_TYPE_TO_DATATYPES["camera"]:
        merged, _ = _fetch_observations_for_datatype(hdr, datatype, stations)
        if merged.empty:
            continue
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=_STATION_LOOKUP_COLUMNS)

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = _standardize_taxonomic_class_column(result)
    if include_iucn_status:
        result = _enrich_with_iucn_status(result, _build_iucn_map(hdr))
    return result


def get_audio_observation_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve merged bioacoustic (audio) species observation rows.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    include_iucn_status:
        If ``True``, add an ``iucn_redlist_status`` column joined on ``species``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per labelled audio segment, joined to station
        location metadata, with an optional ``iucn_redlist_status`` column
        when ``include_iucn_status=True``.

    Examples
    --------
    >>> df = get_audio_observation_data(hdr)  # doctest: +SKIP
    >>> df = get_audio_observation_data(hdr, include_iucn_status=True)  # doctest: +SKIP
    """
    # Pull directly from the audio datatype endpoint to avoid depending on
    # backend measurement_type naming consistency (e.g. "audio" vs "bioacoustic").
    stations = _fetch_stations_for_datatype(hdr, "audio")
    merged, station_lookup = _fetch_observations_for_datatype(hdr, "audio", stations)
    if station_lookup.empty:
        return pd.DataFrame(columns=_STATION_LOOKUP_COLUMNS)
    if merged.empty:
        return station_lookup.drop(columns=["latitude", "longitude"], errors="ignore")
    merged = _standardize_taxonomic_class_column(merged)
    if include_iucn_status:
        merged = _enrich_with_iucn_status(merged, _build_iucn_map(hdr))
    return merged.reset_index(drop=True)


def get_audio_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve merged bioacoustic (audio) rows.

    This is a convenience alias for get_audio_observation_data.
    """
    return get_audio_observation_data(
        hdr=hdr,
        include_iucn_status=include_iucn_status,
    )


def get_edna_assets(hdr: AuthHeaders, project_system_record_id: int) -> pd.DataFrame:
    """Retrieve eDNA assets for a single project system record ID.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    project_system_record_id:
        The project system record ID.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of eDNA assets.

    Raises
    ------
    ValueError
        If ``project_system_record_id`` is not a positive integer.

    Examples
    --------
    >>> assets = get_edna_assets(hdr, project_system_record_id=123)  # doctest: +SKIP
    """
    psr_id = int(project_system_record_id)
    if psr_id <= 0:
        raise ValueError("project_system_record_id must be a positive integer.")
    url = f"{hdr.root}geteDNAAssets/{psr_id}/{hdr.key}"
    response: httpx.Response | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
        try:
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429 or attempt >= _RATE_LIMIT_MAX_RETRIES:
                raise
            retry_after = exc.response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except (TypeError, ValueError):
                delay = 0.0
            if delay <= 0:
                delay = _RATE_LIMIT_BACKOFF_SECONDS * (2**attempt)
            time.sleep(delay)
    if response is None:
        return pd.DataFrame()
    payload = response.json()

    # Backends may return either a flat list of rows or a wrapped payload
    # containing the table in a top-level key.
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("table"), list):
            rows = payload["table"]
        elif isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            rows = [payload]
    else:
        rows = []

    return pd.DataFrame(rows)


def get_edna_observation_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve merged eDNA observation rows for all project stations.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    include_iucn_status:
        If ``True``, add an ``iucn_redlist_status`` column joined
        from the project labels.  The eDNA backend already includes IUCN status
        in its response; this flag adds a second enrichment pass from the
        project labels endpoint for any rows where the status is absent.

    Returns
    -------
    pandas.DataFrame
        A flat DataFrame with one row per eDNA observation, merged with station
        location columns (``latitude``, ``longitude``, ``device_id``) and
        species identification fields (``label``, ``species``, ``genus``,
        ``family``, ``order``, ``common_name``, ``iucn_redlist_status``).

    Examples
    --------
    >>> df = get_edna_observation_data(hdr)  # doctest: +SKIP
    """
    stations = get_station_info(hdr, "edna")
    merged, station_lookup = _fetch_observations_for_datatype(hdr, "eDNA", stations)
    if station_lookup.empty:
        return pd.DataFrame(columns=["project_system_record_id", "device_id", "measurement_type", "latitude", "longitude"])

    if merged.empty:
        return station_lookup.reset_index(drop=True)

    merged = _standardize_taxonomic_class_column(merged)

    if include_iucn_status and "iucn_redlist_status" not in merged.columns:
        merged = _enrich_with_iucn_status(merged, _build_iucn_map(hdr))

    return merged.reset_index(drop=True)


def _normalise_species_frame(df: pd.DataFrame, data_type_fallback: str) -> pd.DataFrame:
    """Ensure core species observation columns are present, filling missing ones with ``pd.NA``."""
    if df.empty:
        return _EMPTY_OBSERVATION_FRAME.copy()

    out = _standardize_taxonomic_class_column(df)
    if "label" not in out.columns:
        out["label"] = out["species"] if "species" in out.columns else pd.NA
    if "data_type" not in out.columns:
        out["data_type"] = data_type_fallback
    for col in _SPECIES_OBS_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _normalise_measurement_type_inputs(
    measurement_types: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normalise user measurement-type input to internal keys."""
    if measurement_types is None:
        return list(_MEASUREMENT_TYPE_TO_DATATYPES.keys())

    raw = [measurement_types] if isinstance(measurement_types, str) else list(measurement_types)
    if not raw:
        return list(_MEASUREMENT_TYPE_TO_DATATYPES.keys())
    normalised: list[str] = []
    for value in raw:
        mt = _normalise_measurement_type(value)
        if mt is None:
            return list(_MEASUREMENT_TYPE_TO_DATATYPES.keys())
        if mt not in _MEASUREMENT_TYPE_TO_DATATYPES:
            raise ValueError(
                f"Invalid measurement_type: {value!r}. "
                f"Must be one of {list(_MEASUREMENT_TYPE_TO_DATATYPES)} or None/'all'."
            )
        if mt not in normalised:
            normalised.append(mt)
    return normalised


def get_species_observations(
    hdr: AuthHeaders,
    measurement_types: str | list[str] | tuple[str, ...] | None = None,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve species observations for one or more measurement-type domains.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    measurement_types:
        A measurement-type key, a sequence of keys, or None/"all".
        Supported keys are ``"camera"``, ``"bioacoustic"``, and ``"edna"``.
    include_iucn_status:
        If ``True``, attach IUCN Red List status to the merged output.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing merged observations across requested domains.

    Examples
    --------
    >>> obs = get_species_observations(hdr, "all")  # doctest: +SKIP
    >>> obs = get_species_observations(hdr, ["camera", "bioacoustic"])  # doctest: +SKIP
    >>> obs = get_species_observations(hdr, include_iucn_status=True)  # doctest: +SKIP
    """
    requested = _normalise_measurement_type_inputs(measurement_types)
    frames: list[pd.DataFrame] = []

    for mt in requested:
        if mt == "camera":
            frame = get_camera_trap_data(hdr, include_iucn_status=False)
            fallback = "image"
        elif mt == "bioacoustic":
            frame = get_audio_observation_data(hdr, include_iucn_status=False)
            fallback = "audio"
        else:
            frame = get_edna_observation_data(hdr)
            fallback = "eDNA"
        frames.append(_normalise_species_frame(frame, fallback))

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return _EMPTY_OBSERVATION_FRAME.copy()

    result = pd.concat(non_empty, ignore_index=True, sort=False)
    if include_iucn_status:
        result = _enrich_with_iucn_status(result, _build_iucn_map(hdr))
    return result


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _normalise_species_name(name: str) -> str:
    """Normalise a species name for reliable dictionary lookup.

    Strips leading/trailing whitespace (including non-breaking spaces) and
    lowercases the string so that case and spacing differences between the
    IUCN table and observation data do not silently break the join.
    """
    return name.replace("\xa0", " ").strip().lower()


def _build_iucn_map(hdr: AuthHeaders) -> dict[str, str | None]:
    """Fetch the full IUCN label table and return a normalised species-name → status mapping.

    Keys are lowercased and stripped (via :func:`_normalise_species_name`) so
    that case/whitespace differences between the IUCN table and observation
    data do not silently break the join.

    A warning is printed if the map is empty, which usually means the
    ``species`` attribute name on :class:`SpeciesLight` does not match the
    actual field returned by the API — check the schema and update the
    ``getattr`` call below accordingly.
    """
    iucn_table = get_iucn_labels(hdr, offset=0, limit=20_000)
    iucn_map: dict[str, str | None] = {}
    for item in iucn_table.table:
        status = getattr(item, "iucn_redlist_status", None)
        species_name = getattr(item, "species", None)
        label_name = getattr(item, "label", None)

        if species_name:
            iucn_map[_normalise_species_name(species_name)] = status
        if label_name:
            iucn_map[_normalise_species_name(label_name)] = status
    if not iucn_map:
        print(
            "Warning: IUCN map is empty. The 'species' attribute may not exist on "
            "SpeciesLight — check the schema field name (e.g. 'scientific_name', "
            "'latin_name') and update _build_iucn_map accordingly."
        )
    return iucn_map


def _enrich_with_iucn_status(df: pd.DataFrame, iucn_map: dict[str, str | None]) -> pd.DataFrame:
    """Add an ``iucn_redlist_status`` column to a DataFrame by joining on ``species``.

    Both sides of the join are normalised via :func:`_normalise_species_name`
    before lookup so that case and whitespace differences do not produce
    spurious ``NaN`` values. If a ``species`` column is absent the DataFrame
    is returned unchanged. Existing ``iucn_redlist_status`` values are never
    overwritten.
    """
    if "species" not in df.columns and "label" not in df.columns:
        return df
    df = df.copy()
    if "iucn_redlist_status" not in df.columns:
        df["iucn_redlist_status"] = pd.NA
    mask = df["iucn_redlist_status"].isna()

    if "species" in df.columns:
        normalised_species = df.loc[mask, "species"].dropna().map(_normalise_species_name)
        df.loc[normalised_species.index, "iucn_redlist_status"] = normalised_species.map(iucn_map)

    if "label" in df.columns:
        mask = df["iucn_redlist_status"].isna()
        normalised_label = df.loc[mask, "label"].dropna().map(_normalise_species_name)
        df.loc[normalised_label.index, "iucn_redlist_status"] = normalised_label.map(iucn_map)

    return df


def get_project_labels(
    hdr: AuthHeaders,
    labeltype: LabelType,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve project-specific labels as a DataFrame.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        One of ``"Bioacoustic"``, ``"Camera"``, or ``"Observation"``.
    include_iucn_status:
        If ``True``, attach IUCN Red List status to each label.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing project labels, with an optional
        ``iucn_redlist_status`` column when ``include_iucn_status=True``.

    Examples
    --------
    >>> labels = get_project_labels(hdr, "Camera")  # doctest: +SKIP
    >>> labels = get_project_labels(hdr, "Camera", include_iucn_status=True)  # doctest: +SKIP
    """
    df = pd.DataFrame(_fetch_project_labels_rows(hdr, labeltype))
    if include_iucn_status and not df.empty:
        df = _enrich_with_iucn_status(df, _build_iucn_map(hdr))
    return df


def _fetch_project_labels_rows(hdr: AuthHeaders, labeltype: LabelType) -> list[dict[str, Any]]:
    """Fetch and normalise project label payload rows."""
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return _extract_rows_from_payload(response.json())


def add_project_labels(
    hdr: AuthHeaders,
    labeltype: LabelType,
    labels: list[Label],
) -> dict[str, str]:
    """Add labels to the project.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        One of ``"Bioacoustic"``, ``"Camera"``, or ``"Observation"``.
    labels:
        A list of :class:`Label` objects to add.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> add_project_labels(hdr, "Camera", labels=[Label(...)])  # doctest: +SKIP
    """
    url = f"{hdr.root}addProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.post(url, json=[label.model_dump(mode="json") for label in labels], timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_iucn_labels(
    hdr: AuthHeaders,
    offset: int,
    limit: int,
    search_term: str | None = None,
) -> SpeciesTable:
    """Retrieve labels from the wider IUCN database.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    offset:
        Pagination offset.
    limit:
        Maximum number of records to return (max 20 000).
    search_term:
        Optional search term to filter results.

    Returns
    -------
    SpeciesTable
        A ``SpeciesTable`` with ``table`` (list of :class:`SpeciesLight`) and
        ``pagination_state``.

    Raises
    ------
    ValueError
        If ``limit`` exceeds 20 000.

    Examples
    --------
    >>> result = get_iucn_labels(hdr, offset=0, limit=100, search_term="horse")  # doctest: +SKIP
    """
    if limit > 20_000:
        raise ValueError("limit cannot exceed 20000.")
    params: dict[str, Any] = {"offset": offset, "limit": limit, "search_term": search_term or ""}
    url = f"{hdr.root}getIUCNLabels/{hdr.key}"
    response = httpx.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return SpeciesTable.model_validate(response.json())


def add_iucn_labels(
    hdr: AuthHeaders,
    labels: list[IUCNSpeciesLabelInput],
    chunksize: int = 200,
) -> Any:
    """Add labels from the IUCN database in chunks.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labels:
        A list of :class:`IUCNSpeciesLabelInput` objects to add.
    chunksize:
        Number of records per submission chunk.

    Returns
    -------
    Any
        The API response from the last chunk submission.

    Examples
    --------
    >>> add_iucn_labels(hdr, labels=my_labels, chunksize=200)  # doctest: +SKIP
    """
    n = len(labels)
    if n < 100:
        print("Dataset is too small to chunk — submitting all records.")
        chunks = [labels]
    else:
        if chunksize > n:
            chunksize = n // 2
            print(f"chunksize exceeds dataset size; using chunksize={chunksize}.")
        num_chunks = math.ceil(n / chunksize)
        chunks = [labels[i * chunksize:(i + 1) * chunksize] for i in range(num_chunks)]

    url = f"{hdr.root}addIUCNLabels/{hdr.key}"
    resp = None
    for i, chunk in enumerate(chunks, start=1):
        payload = [label.model_dump(mode="json", by_alias=True) for label in chunk]
        response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        resp = response.json()
        print(f"Submitted {min(i * chunksize, n)} of {n} labels.")
    return resp


def update_segment_labels(hdr: AuthHeaders, labels: list[Label]) -> dict[str, str]:
    """Update segment labels.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labels:
        A list of :class:`Label` objects with updated information.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> update_segment_labels(hdr, labels)  # doctest: +SKIP
    """
    url = f"{hdr.root}updateSegmentLabels/{hdr.key}"
    response = httpx.put(url, json=[label.model_dump(mode="json") for label in labels], timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def push_new_labels(
    hdr: AuthHeaders,
    submission_records: list[Label],
    chunksize: int,
) -> None:
    """Push new segment labels to the platform in chunks.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    submission_records:
        A list of :class:`Label` objects to submit.
    chunksize:
        Number of records per chunk.

    Examples
    --------
    >>> push_new_labels(hdr, submission_records, chunksize=30)  # doctest: +SKIP
    """
    n = len(submission_records)
    if chunksize > n:
        print(f"chunksize exceeds dataset size; using chunksize={n}.")
        chunksize = n
    num_chunks = math.ceil(n / chunksize)
    for i in range(num_chunks):
        chunk = submission_records[i * chunksize:(i + 1) * chunksize]
        update_segment_labels(hdr, chunk)
        print(f"Submitted {min((i + 1) * chunksize, n)} of {n} labels.")


def set_segment_blank_status(
    hdr: AuthHeaders,
    blank_status: bool,
    segment_record_ids: list[int],
) -> dict[str, str]:
    """Mark or unmark segment labels as blank.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    blank_status:
        ``True`` to mark as blank, ``False`` to unmark.
    segment_record_ids:
        A list of segment record IDs to update.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> set_segment_blank_status(hdr, blank_status=True, segment_record_ids=[101, 102])  # doctest: +SKIP
    """
    url = f"{hdr.root}segmentLabelsBlankStatus/{hdr.key}/{str(blank_status).lower()}"
    response = httpx.put(url, json=segment_record_ids, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    resp = response.json()
    print(resp.get("message", ""))
    return resp


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def update_media_timestamps(
    hdr: AuthHeaders,
    media_records: list[MediaTimestampUpdate],
) -> list[TimestampUpdateResponse]:
    """Update timestamps for one or more media file records.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    media_records:
        A list of :class:`MediaTimestampUpdate` objects.

    Returns
    -------
    list[TimestampUpdateResponse]
        A list of responses indicating success for each update.

    Examples
    --------
    >>> from datetime import datetime
    >>> updates = [MediaTimestampUpdate(media_file_record_id=123, new_timestamp=datetime.now())]
    >>> result = update_media_timestamps(hdr, updates)  # doctest: +SKIP
    """
    url = f"{hdr.root}updateTimestamps/{hdr.key}"
    payload = [record.model_dump(mode="json") for record in media_records]
    response = httpx.put(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = [TimestampUpdateResponse.model_validate(item) for item in response.json()]
    print(f"Successfully updated {len(result)} media timestamp(s).")
    return result


def push_new_timestamps(
    hdr: AuthHeaders,
    media_metadata: list[MediaTimestampUpdate],
    chunksize: int,
) -> None:
    """Update timestamps for multiple media records, split into chunks.

    Recommended for large datasets (>1 000 records) to prevent timeouts.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    media_metadata:
        A list of :class:`MediaTimestampUpdate` objects.
    chunksize:
        Number of records per chunk. Recommended: 50–200.

    Examples
    --------
    >>> push_new_timestamps(hdr, updates, chunksize=100)  # doctest: +SKIP
    """
    n = len(media_metadata)
    if chunksize > n:
        print(f"chunksize exceeds dataset size; altering chunksize to {n}.")
        chunksize = n
    num_chunks = math.ceil(n / chunksize)
    for i in range(num_chunks):
        chunk = media_metadata[i * chunksize:(i + 1) * chunksize]
        update_media_timestamps(hdr, chunk)
        print(f"Submitted {min((i + 1) * chunksize, n)} of {n} timestamps.")


# ---------------------------------------------------------------------------
# eDNA
# ---------------------------------------------------------------------------


def check_edna_labels(
    hdr: AuthHeaders,
    edna_data: list[eDNAUploadSchema],
) -> list[eDNAUploadResponse]:
    """Validate eDNA records against the Okala database.

    Uses a hierarchical taxonomy approach:
    species → genus → family → order → class → phylum → kingdom.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    edna_data:
        A list of :class:`eDNAUploadSchema` objects.

    Returns
    -------
    list[eDNAUploadResponse]
        Validated records with ``label``, ``label_id``, ``status``, and ``message``.

    Examples
    --------
    >>> validated = check_edna_labels(hdr, edna_records)  # doctest: +SKIP
    """
    url = f"{hdr.root}checkeDNALabels/{hdr.key}"
    payload = [record.model_dump(mode="json", by_alias=True) for record in edna_data]
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = [eDNAUploadResponse.model_validate(item) for item in response.json()]
    print(f"Validated {len(result)} eDNA records.")
    return result


def check_edna_labels_df(hdr: AuthHeaders, edna_data: pd.DataFrame) -> pd.DataFrame:
    """Validate eDNA records from a DataFrame against the Okala database.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    edna_data:
        A DataFrame with eDNA records. Required columns: ``marker_name``,
        ``sequence``, ``primer``, ``timestamp``. Optional taxonomy columns:
        ``kingdom``, ``phylum``, ``class``, ``order``, ``family``, ``genus``,
        ``species``, ``confidence``.

    Returns
    -------
    pandas.DataFrame
        The original data with additional columns: ``label``, ``label_id``,
        ``status``, and ``message``.

    Raises
    ------
    ValueError
        If required columns are missing.

    Examples
    --------
    >>> validated = check_edna_labels_df(hdr, edna_records)  # doctest: +SKIP
    """
    required_cols = {"marker_name", "sequence", "primer", "timestamp"}
    missing = required_cols - set(edna_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")

    data = edna_data.copy()
    if "confidence" not in data.columns:
        data["confidence"] = 100
    if "class_" in data.columns and "class" not in data.columns:
        data["class"] = data["class_"]

    records = [{k: v for k, v in row.items() if pd.notna(v)} for row in data.to_dict(orient="records")]

    url = f"{hdr.root}checkeDNALabels/{hdr.key}"
    response = httpx.post(url, json=records, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = _standardize_taxonomic_class_column(pd.DataFrame(response.json()))
    print(f"Validated {len(result)} eDNA records.")
    return result


def upload_edna_records(
    hdr: AuthHeaders,
    validated_data: list[eDNAUploadResponse],
    project_system_record_id: int,
) -> list[eDNAUploadResponse]:
    """Upload validated eDNA records to a project system record.

    Only records with ``status == "success"`` are uploaded.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    validated_data:
        A list of validated :class:`eDNAUploadResponse` objects from
        :func:`check_edna_labels`.
    project_system_record_id:
        The project system record ID to upload records to.

    Returns
    -------
    list[eDNAUploadResponse]
        A list of responses for each uploaded record.

    Raises
    ------
    ValueError
        If no successful records exist.

    Examples
    --------
    >>> validated = check_edna_labels(hdr, edna_records)  # doctest: +SKIP
    >>> result = upload_edna_records(hdr, validated, project_system_record_id=123)  # doctest: +SKIP
    """
    successful = [r for r in validated_data if r.status == "success"]
    if not successful:
        raise ValueError("No successful records to upload — all records failed validation.")

    print(f"Uploading {len(successful)} validated eDNA records.")
    url = f"{hdr.root}uploadeDNA/{hdr.key}/{project_system_record_id}"
    payload = [record.model_dump(mode="json", by_alias=True) for record in successful]
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = [eDNAUploadResponse.model_validate(item) for item in response.json()]
    print(f"Upload complete: {len(result)} records processed.")
    return result