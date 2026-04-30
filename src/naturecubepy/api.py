"""
Core API wrapper functions for the Okala dashboard.

This module provides functions for authenticating with the Okala API and
interacting with project data including stations, media assets, labels,
eDNA records, and timestamps.
"""

from __future__ import annotations

import math
import os
from typing import Any, Literal

import geopandas as gpd
import httpx
import pandas as pd

from naturecubepy.schema import (
    AuthHeaders,
    CameraTrapDataRecord,
    DataTypes,
    GetProjectGeometryResponse,
    IUCNSpeciesLabelInput,
    Label,
    LabelType,
    MediaRecordAPIFlat,
    MediaTimestampUpdate,
    SegmentRecordAPIFlat,
    SpeciesLight,
    SpeciesTable,
    StationResponseAPI,
    TimestampUpdateResponse,
    eDNAUploadResponse,
    eDNAUploadSchema,
)

_PROD_URL = "https://naturecube.io/api/"
_DEV_URL = "http://127.0.0.1:8000/api/"
_DEFAULT_TIMEOUT = 60.0

_MEASUREMENT_TYPE_TO_DATATYPE = {
    "camera": "image",
    "bioacoustic": "audio",
    "edna": "eDNA",
}

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


def get_station_info(hdr: AuthHeaders, measurement_type: str) -> gpd.GeoDataFrame:
    """Retrieve all station metadata for a project, optionally filtered by measurement type.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    measurement_type:
        One of ``"camera"``, ``"bioacoustic"``, ``"edna"``, or None/"all" for all stations.

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
    if measurement_type is None or str(measurement_type).lower() == "all":
        # Fetch all types and concatenate
        dfs = []
        for mt, datatype in _MEASUREMENT_TYPE_TO_DATATYPE.items():
            url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
            response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
            response.raise_for_status()
            gdf = gpd.read_file(response.text)
            # Only keep stations matching the measurement_type (in case API returns more)
            gdf = gdf[gdf.get("measurement_type", None) == mt.capitalize()]
            dfs.append(gdf)
        if dfs:
            return gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
        else:
            return gpd.GeoDataFrame()
    else:
        mt_lower = str(measurement_type).lower()
        if mt_lower not in _MEASUREMENT_TYPE_TO_DATATYPE:
            raise ValueError(
                f"Invalid measurement_type: {measurement_type!r}. "
                f"Must be one of {list(_MEASUREMENT_TYPE_TO_DATATYPE)} or None/'all'."
            )
        datatype = _MEASUREMENT_TYPE_TO_DATATYPE[mt_lower]
        url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
        response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        gdf = gpd.read_file(response.text)
        return gdf[gdf.get("measurement_type", None) == mt_lower.capitalize()]


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
    url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return StationResponseAPI.model_validate(response.json())


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
        items = response.json()
        if not items:
            break
        all_results.extend([MediaRecordAPIFlat.model_validate(item) for item in items])
        if len(items) < limit:
            break
        current_offset += limit
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
    all_results = []
    psr_ids = _normalise_psr_ids(project_system_record_ids)
    chunk_size = 25  # Tune for backend performance
    for i in range(0, len(psr_ids), chunk_size):
        chunk = psr_ids[i:i+chunk_size]
        offset = 0
        limit = 1000
        while True:
            params = {"offset": offset, "limit": limit}
            try:
                response = httpx.post(url, json=chunk, params=params, timeout=_DEFAULT_TIMEOUT)
                response.raise_for_status()
            except Exception as exc:
                print(f"Error fetching media assets: {exc}\nPayload: {chunk}\nResponse: {getattr(response, 'text', '')}")
                raise
            items = response.json()
            if not items:
                break
            all_results.extend(items)
            if len(items) < limit:
                break
            offset += limit
    return pd.DataFrame(all_results)


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
    all_results = []
    psr_ids = _normalise_psr_ids(project_system_record_ids)
    chunk_size = 25  # Tune for backend performance
    for i in range(0, len(psr_ids), chunk_size):
        chunk = psr_ids[i:i+chunk_size]
        offset = 0
        limit = 1000
        while True:
            params = {"offset": offset, "limit": limit}
            response = httpx.post(url, json=chunk, params=params, timeout=_DEFAULT_TIMEOUT)
            response.raise_for_status()
            items = response.json()
            if not items:
                break
            all_results.extend(items)
            if len(items) < limit:
                break
            offset += limit
    return pd.DataFrame(all_results)


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
    """Left-join segment columns onto a media DataFrame, avoiding duplicate columns."""
    if segments_df.empty:
        return media_df
    segment_cols = [c for c in segments_df.columns if c == "segment_record_id" or c not in media_df.columns]
    return media_df.merge(
        segments_df[segment_cols].drop_duplicates(subset=["segment_record_id"]),
        on="segment_record_id",
        how="left",
    )


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
    """Fetch media assets and segments, merge with station lookup, and return a flat DataFrame."""
    psr_ids = station_lookup["project_system_record_id"].tolist()
    chunk_size = 100  
    media_frames = []
    segment_frames = []
    for i in range(0, len(psr_ids), chunk_size):
        chunk = psr_ids[i:i+chunk_size]
        media_df = get_media_assets_df(hdr, datatype, project_system_record_ids=chunk)
        if not media_df.empty:
            media_frames.append(media_df)
        segments_df = get_media_segments(hdr, datatype, project_system_record_ids=chunk)
        if not segments_df.empty:
            segment_frames.append(segments_df)

    if not media_frames:
        return pd.DataFrame()

    media_df = pd.concat(media_frames, ignore_index=True)
    media_df = _resolve_psr_id_column(media_df)
    segments_df = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()

    merged = _merge_segments(media_df, segments_df)
    merged = _merge_station_lookup(merged, station_lookup)

    if "data_type" not in merged.columns:
        merged["data_type"] = data_type_fallback

    return merged


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

    for datatype in ("image", "video"):
        station_lookup = _build_station_lookup(stations, datatype)
        if station_lookup.empty:
            continue

        merged = _fetch_and_merge_media(hdr, datatype, station_lookup, datatype)  # type: ignore[arg-type]
        if merged.empty:
            continue

        validated = [
            CameraTrapDataRecord.model_validate(row).model_dump(mode="json", by_alias=True)
            for row in merged.to_dict(orient="records")
        ]
        frames.append(pd.DataFrame(validated))

    if not frames:
        return pd.DataFrame(columns=_STATION_LOOKUP_COLUMNS)

    result = pd.concat(frames, ignore_index=True, sort=False)
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
    stations = get_station_info(hdr, "bioacoustic")
    station_lookup = _build_station_lookup(stations, "audio")
    if station_lookup.empty:
        return pd.DataFrame(columns=_STATION_LOOKUP_COLUMNS)

    merged = _fetch_and_merge_media(hdr, "audio", station_lookup, "audio")  # type: ignore[arg-type]
    if merged.empty:
        return station_lookup.drop(columns=["latitude", "longitude"], errors="ignore")
    if include_iucn_status:
        merged = _enrich_with_iucn_status(merged, _build_iucn_map(hdr))
    return merged.reset_index(drop=True)


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
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_edna_observation_data(hdr: AuthHeaders) -> pd.DataFrame:
    """Retrieve merged eDNA observation rows for all project stations.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    pandas.DataFrame
        A flat DataFrame with one row per eDNA asset, merged with station
        metadata columns.

    Examples
    --------
    >>> df = get_edna_observation_data(hdr)  # doctest: +SKIP
    """
    stations = get_station_info(hdr, "eDNA")
    station_lookup = _build_station_lookup(stations, "eDNA")
    if station_lookup.empty:
        return pd.DataFrame(columns=["project_system_record_id", "device_id", "measurement_type", "latitude", "longitude"])

    frames: list[pd.DataFrame] = []
    for psr_id in station_lookup["project_system_record_id"].tolist():
        try:
            assets = get_edna_assets(hdr, project_system_record_id=psr_id)
        except Exception:
            continue
        if not assets.empty:
            assets = assets.copy()
            assets["project_system_record_id"] = psr_id
            frames.append(assets)

    if not frames:
        return station_lookup.reset_index(drop=True)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _merge_station_lookup(merged, station_lookup)
    if "data_type" not in merged.columns:
        merged["data_type"] = "eDNA"
    return merged.reset_index(drop=True)


def _normalise_species_frame(df: pd.DataFrame, data_type_fallback: str) -> pd.DataFrame:
    """Ensure core species observation columns are present, filling missing ones with ``pd.NA``."""
    if df.empty:
        return _EMPTY_OBSERVATION_FRAME.copy()

    out = df.copy()
    if "label" not in out.columns:
        out["label"] = out["species"] if "species" in out.columns else pd.NA
    if "data_type" not in out.columns:
        out["data_type"] = data_type_fallback
    for col in _SPECIES_OBS_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def get_all_species_observations(hdr: AuthHeaders) -> pd.DataFrame:
    """Retrieve all species identifications across every measurement type.

    Calls :func:`get_camera_trap_data`, :func:`get_audio_observation_data`,
    and :func:`get_edna_observation_data`, normalises their columns, and
    concatenates the results into a single flat DataFrame.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with at least the columns listed in ``_SPECIES_OBS_CORE_COLUMNS``.
        Additional columns from each source are preserved.

    Examples
    --------
    >>> obs = get_all_species_observations(hdr)  # doctest: +SKIP
    """
    frames = [
        _normalise_species_frame(get_camera_trap_data(hdr), "image"),
        _normalise_species_frame(get_audio_observation_data(hdr), "audio"),
        _normalise_species_frame(get_edna_observation_data(hdr), "eDNA"),
    ]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return _EMPTY_OBSERVATION_FRAME.copy()
    return pd.concat(non_empty, ignore_index=True, sort=False)


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
    iucn_map = {
        _normalise_species_name(item.species): getattr(item, "iucn_redlist_status", None)
        for item in iucn_table.table
        if getattr(item, "species", None)
    }
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
    if "species" not in df.columns:
        return df
    df = df.copy()
    if "iucn_redlist_status" not in df.columns:
        df["iucn_redlist_status"] = pd.NA
    mask = df["iucn_redlist_status"].isna()
    normalised_species = df.loc[mask, "species"].dropna().map(_normalise_species_name)
    df.loc[normalised_species.index, "iucn_redlist_status"] = normalised_species.map(iucn_map)
    return df


def get_project_labels(
    hdr: AuthHeaders,
    labeltype: LabelType,
    include_iucn_status: bool = False,
) -> list[SpeciesLight]:
    """Retrieve project-specific labels, optionally enriched with IUCN status.

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
    list[SpeciesLight]
        A list of project labels with species information.

    Examples
    --------
    >>> labels = get_project_labels(hdr, "Camera", include_iucn_status=True)  # doctest: +SKIP
    """
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    labels = [SpeciesLight.model_validate(item) for item in response.json()]

    if include_iucn_status and labels:
        iucn_map = _build_iucn_map(hdr)
        for label in labels:
            species = getattr(label, "species", None)
            if species and hasattr(label, "iucn_redlist_status"):
                label.iucn_redlist_status = iucn_map.get(species)

    return labels


def get_project_labels_df(
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
        If ``True``, add an ``iucn_redlist_status`` column joined on ``species``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing project labels, with an optional
        ``iucn_redlist_status`` column when ``include_iucn_status=True``.

    Examples
    --------
    >>> labels = get_project_labels_df(hdr, "Camera")  # doctest: +SKIP
    >>> labels = get_project_labels_df(hdr, "Camera", include_iucn_status=True)  # doctest: +SKIP
    """
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    if include_iucn_status and not df.empty:
        df = _enrich_with_iucn_status(df, _build_iucn_map(hdr))
    return df


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
        raise ValueError("limit cannot exceed 20 000.")
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
        print(f"chunksize exceeds dataset size; using chunksize={n}.")
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
    result = pd.DataFrame(response.json())
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