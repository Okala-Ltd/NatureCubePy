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
_DEFAULT_TIMEOUT = 60.0  # seconds


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


def auth_headers(
    api_key: str,
    okala_url: str = _PROD_URL,
) -> AuthHeaders:
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
        An AuthHeaders object with ``key`` and ``root`` attributes.

    Examples
    --------
    >>> hdr = auth_headers("mykey")
    >>> hdr.root
    'https://naturecube.io/api/'
    """
    return AuthHeaders(key=api_key, root=okala_url.rstrip("/") + "/")


def auth_headers_dev(
    api_key: str,
    okala_url: str = _DEV_URL,
) -> AuthHeaders:
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
        An AuthHeaders object with ``key`` and ``root`` attributes.

    Examples
    --------
    >>> hdr = auth_headers_dev("mykey")
    >>> hdr.root
    'http://127.0.0.1:8000/api/'
    """
    return AuthHeaders(key=api_key, root=okala_url.rstrip("/") + "/")


def get_project(hdr: AuthHeaders, timeout: float = _DEFAULT_TIMEOUT) -> GetProjectGeometryResponse:
    """Retrieve and display the active project associated with the API key.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    timeout:
        Request timeout in seconds. Defaults to 60.

    Returns
    -------
    GetProjectGeometryResponse
        The project geometry response containing boundary, ROIs, and locations.

    Examples
    --------
    >>> hdr = auth_headers("mykey")
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


def get_station_info(
    hdr: AuthHeaders,
    datatype: DataTypes,
) -> gpd.GeoDataFrame:
    """Retrieve all station metadata for a project.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing station metadata and geometry.

    Examples
    --------
    >>> hdr = auth_headers("mykey")
    >>> stations = get_station_info(hdr, "video")  # doctest: +SKIP
    """
    url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return gpd.read_file(response.text)


def get_stations_typed(
    hdr: AuthHeaders,
    datatype: DataTypes,
) -> StationResponseAPI:
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
    >>> hdr = auth_headers("mykey")
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
    >>> stations = get_station_info(hdr, "video")  # doctest: +SKIP
    >>> m = plot_stations(stations)  # doctest: +SKIP
    """
    import folium

    print("Plotting stations")

    # Compute centroids in a projected CRS when source data are geographic.
    if geojson_response.crs is not None and geojson_response.crs.is_geographic:
        centroids = geojson_response.to_crs(epsg=3857).geometry.centroid.to_crs(geojson_response.crs)
    else:
        centroids = geojson_response.geometry.centroid
    centre_lat = centroids.y.mean()
    centre_lon = centroids.x.mean()

    # Satellite tile layer via xyzservices (handles URL + attribution automatically).
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10, tiles="Esri WorldImagery")

    record_counts = geojson_response.get("record_count", pd.Series([1] * len(geojson_response)))
    min_count = record_counts.min() if not record_counts.empty else 1
    max_count = record_counts.max() if not record_counts.empty else 1

    def _rescale(value: float, new_min: float = 5, new_max: float = 15) -> float:
        if max_count == min_count:
            return (new_min + new_max) / 2
        return new_min + (value - min_count) / (max_count - min_count) * (new_max - new_min)

    for idx, row in geojson_response.iterrows():
        lat = centroids.loc[idx].y
        lon = centroids.loc[idx].x
        device_id = row.get("device_id", "")
        start_ts = row.get("project_system_record_start_timestamp", "")
        end_ts = row.get("project_system_record_end_timestamp", "")
        count = row.get("record_count", 1)
        measurement_type = row.get("measurement_type", "")
        data_type = row.get("data_type", "")

        popup_html = (
            f"Device ID: {device_id}<br>"
            f"Measurement type: {measurement_type}<br>"
            f"Data type: {data_type}<br>"
            f"Start time: {start_ts}<br>"
            f"End time: {end_ts}<br>"
            f"No. media files: {count}<br>"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=_rescale(count),
            tooltip=str(device_id),
            popup=folium.Popup(popup_html, max_width=300),
            color="red",
            fill=True,
            fill_opacity=0.6,
            opacity=0.2,
        ).add_to(m)

    return m


def get_media_segments(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
) -> list[SegmentRecordAPIFlat]:
    """Retrieve media segments for given project system record IDs.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    psr_ids:
        List of project system record IDs.

    Returns
    -------
    list[SegmentRecordAPIFlat]
        A list of segment records.

    Examples
    --------
    >>> segments = get_media_segments(hdr, "video", psr_ids=[123, 456])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaSegments/{datatype}/{hdr.key}"
    response = httpx.post(url, json=psr_ids, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return [SegmentRecordAPIFlat.model_validate(item) for item in response.json()]


def get_media_assets(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
) -> list[MediaRecordAPIFlat]:
    """Retrieve media assets for given project system record IDs.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    psr_ids:
        List of project system record IDs.

    Returns
    -------
    list[MediaRecordAPIFlat]
        A list of media records with full details.

    Examples
    --------
    >>> assets = get_media_assets(hdr, "video", psr_ids=[123])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}"
    response = httpx.post(url, json=psr_ids, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return [MediaRecordAPIFlat.model_validate(item) for item in response.json()]


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
    >>> assets = get_media_assets_df(hdr, "video", project_system_record_ids=123)  # doctest: +SKIP
    >>> assets = get_media_assets_df(hdr, "video", project_system_record_ids=[123, 456])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}"
    payload = _normalise_project_system_record_ids(project_system_record_ids)
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    try:
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching media assets: {e}\nPayload: {payload}\nResponse: {response.text}")
        raise
    return pd.DataFrame(response.json())


def _normalise_project_system_record_ids(
    project_system_record_ids: int | list[int],
) -> list[int]:
    """Normalise project system record ID inputs to the API's list format."""
    if isinstance(project_system_record_ids, int):
        record_ids = [project_system_record_ids]
    else:
        record_ids = list(project_system_record_ids)

    if not record_ids:
        raise ValueError("project_system_record_ids must contain at least one ID")

    try:
        normalised = [int(record_id) for record_id in record_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("project_system_record_ids must contain integers") from exc

    if any(record_id <= 0 for record_id in normalised):
        raise ValueError("project_system_record_ids must contain positive integers")

    return normalised


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
        A DataFrame of media segments for the specified project system records.

    Examples
    --------
    >>> segments = get_media_segments(hdr, "audio", project_system_record_ids=123)  # doctest: +SKIP
    >>> segments = get_media_segments(hdr, "audio", project_system_record_ids=[123, 456])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaSegments/{datatype}/{hdr.key}"
    payload = _normalise_project_system_record_ids(project_system_record_ids)
    response = httpx.post(
        url,
        json=payload,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def _camera_trap_datatypes(datatype: Literal["image", "video", "both"] = "both") -> list[Literal["image", "video"]]:
    """Normalise camera trap datatype input to the supported image/video list."""
    if datatype == "both":
        return ["image", "video"]
    return [datatype]


def _build_camera_trap_station_lookup(
    stations: gpd.GeoDataFrame,
    datatype: Literal["image", "video"],
) -> pd.DataFrame:
    """Build a station lookup table with stable camera trap metadata."""
    if "project_system_record_id" not in stations.columns:
        raise ValueError(
            "Station data does not contain 'project_system_record_id' column. "
            "Cannot link media assets to stations."
        )

    lookup = stations.dropna(subset=["project_system_record_id"]).copy()
    if lookup.empty:
        return pd.DataFrame(
            columns=[
                "project_system_record_id",
                "device_id",
                "data_type",
                "measurement_type",
                "latitude",
                "longitude",
            ]
        )

    lookup["project_system_record_id"] = lookup["project_system_record_id"].astype(int)
    lookup["latitude"] = lookup.geometry.y
    lookup["longitude"] = lookup.geometry.x
    if "data_type" not in lookup.columns:
        lookup["data_type"] = datatype

    selected_columns = [
        "project_system_record_id",
        "device_id",
        "data_type",
        "measurement_type",
        "latitude",
        "longitude",
    ]
    available_columns = [column for column in selected_columns if column in lookup.columns]
    return lookup[available_columns].drop_duplicates(subset=["project_system_record_id"])


def get_camera_trap_data(
    hdr: AuthHeaders,
    datatype: Literal["image", "video", "both"] = "both",
) -> pd.DataFrame:
    """Retrieve merged camera trap media rows for image and video stations.

    The returned DataFrame combines station metadata, media assets, and media
    segments while avoiding duplicate merge columns. By default, both image and
    video camera trap data are returned together.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        ``"image"``, ``"video"``, or ``"both"``. Defaults to ``"both"``.

    Returns
    -------
    pandas.DataFrame
        A validated DataFrame containing merged camera trap media rows.

    Examples
    --------
    >>> df = get_camera_trap_data(hdr)  # doctest: +SKIP
    >>> sorted(df["data_type"].unique())  # doctest: +SKIP
    ['image', 'video']
    """
    frames: list[pd.DataFrame] = []

    for current_datatype in _camera_trap_datatypes(datatype):
        stations = get_station_info(hdr, current_datatype)
        station_lookup = _build_camera_trap_station_lookup(stations, current_datatype)
        if station_lookup.empty:
            continue

        psr_ids = station_lookup["project_system_record_id"].tolist()
        media_df = get_media_assets_df(hdr, current_datatype, project_system_record_ids=psr_ids)
        if media_df.empty:
            continue

        if "project_system_record_id" not in media_df.columns and "project_system_record_id_fk" in media_df.columns:
            media_df["project_system_record_id"] = media_df["project_system_record_id_fk"]

        if "project_system_record_id" not in media_df.columns:
            raise ValueError(
                "Media asset data does not contain a project system record identifier. "
                "Expected 'project_system_record_id' or 'project_system_record_id_fk'."
            )

        media_df["project_system_record_id"] = pd.to_numeric(
            media_df["project_system_record_id"],
            errors="raise",
        ).astype(int)

        segments_df = get_media_segments(hdr, current_datatype, project_system_record_ids=psr_ids)
        merged_df = media_df.copy()

        if not segments_df.empty:
            segment_merge_columns = [
                column
                for column in segments_df.columns
                if column == "segment_record_id" or column not in merged_df.columns
            ]
            merged_df = merged_df.merge(
                segments_df[segment_merge_columns].drop_duplicates(subset=["segment_record_id"]),
                on="segment_record_id",
                how="left",
            )

        station_merge_columns = ["project_system_record_id"] + [
            column for column in station_lookup.columns if column not in merged_df.columns
        ]
        merged_df = merged_df.merge(
            station_lookup[station_merge_columns],
            on="project_system_record_id",
            how="left",
        )

        if "data_type" not in merged_df.columns:
            merged_df["data_type"] = current_datatype

        validated_rows = [
            CameraTrapDataRecord.model_validate(row).model_dump(mode="json", by_alias=True)
            for row in merged_df.to_dict(orient="records")
        ]
        frames.append(pd.DataFrame(validated_rows))

    if not frames:
        return pd.DataFrame(
            columns=[
                "project_system_record_id",
                "device_id",
                "data_type",
                "measurement_type",
                "latitude",
                "longitude",
            ]
        )

    return pd.concat(frames, ignore_index=True, sort=False)


def get_audio_observation_data(
    hdr: AuthHeaders,
) -> pd.DataFrame:
    """Retrieve merged bioacoustic (audio) species observation rows.

    Mirrors :func:`get_camera_trap_data` but fetches audio stations and their
    associated media assets and segments.  The result is a flat DataFrame with
    one row per labelled audio segment, joined to station location metadata.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing merged audio observation rows with columns
        including ``project_system_record_id``, ``device_id``, ``data_type``,
        ``measurement_type``, ``latitude``, ``longitude``, and any additional
        fields returned by the media asset and segment endpoints.

    Examples
    --------
    >>> df = get_audio_observation_data(hdr)  # doctest: +SKIP
    >>> df.columns.tolist()  # doctest: +SKIP
    ['project_system_record_id', 'device_id', 'data_type', ...]
    """
    stations = get_station_info(hdr, "audio")
    station_lookup = _build_camera_trap_station_lookup(stations, "audio")  # type: ignore[arg-type]
    if station_lookup.empty:
        return pd.DataFrame(
            columns=[
                "project_system_record_id",
                "device_id",
                "data_type",
                "measurement_type",
                "latitude",
                "longitude",
            ]
        )

    psr_ids = station_lookup["project_system_record_id"].tolist()
    media_df = get_media_assets_df(hdr, "audio", project_system_record_ids=psr_ids)
    if media_df.empty:
        return station_lookup.drop(columns=["latitude", "longitude"], errors="ignore")

    if "project_system_record_id" not in media_df.columns and "project_system_record_id_fk" in media_df.columns:
        media_df["project_system_record_id"] = media_df["project_system_record_id_fk"]

    if "project_system_record_id" not in media_df.columns:
        raise ValueError(
            "Media asset data does not contain a project system record identifier. "
            "Expected 'project_system_record_id' or 'project_system_record_id_fk'."
        )

    media_df["project_system_record_id"] = pd.to_numeric(
        media_df["project_system_record_id"],
        errors="raise",
    ).astype(int)

    segments_df = get_media_segments(hdr, "audio", project_system_record_ids=psr_ids)
    merged_df = media_df.copy()

    if not segments_df.empty:
        segment_merge_columns = [
            column
            for column in segments_df.columns
            if column == "segment_record_id" or column not in merged_df.columns
        ]
        merged_df = merged_df.merge(
            segments_df[segment_merge_columns].drop_duplicates(subset=["segment_record_id"]),
            on="segment_record_id",
            how="left",
        )

    station_merge_columns = ["project_system_record_id"] + [
        column for column in station_lookup.columns if column not in merged_df.columns
    ]
    merged_df = merged_df.merge(
        station_lookup[station_merge_columns],
        on="project_system_record_id",
        how="left",
    )

    if "data_type" not in merged_df.columns:
        merged_df["data_type"] = "audio"

    return merged_df.reset_index(drop=True)


def get_edna_assets(
    hdr: AuthHeaders,
    project_system_record_id: int,
) -> pd.DataFrame:
    """Retrieve eDNA assets for a project system record ID.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    project_system_record_id:
        The project system record ID.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of eDNA assets for the specified project system record.

    Examples
    --------
    >>> assets = get_edna_assets(hdr, project_system_record_id=123)  # doctest: +SKIP
    """
    project_system_record_id = int(project_system_record_id)
    if project_system_record_id <= 0:
        raise ValueError("project_system_record_id must be a positive integer")

    url = f"{hdr.root}geteDNAAssets/{project_system_record_id}/{hdr.key}"
    response = httpx.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_edna_observation_data(
    hdr: AuthHeaders,
) -> pd.DataFrame:
    """Retrieve merged eDNA observation rows for all project stations.

    Fetches every eDNA station in the project, pulls assets for each one via
    :func:`get_edna_assets`, and joins station location metadata (latitude,
    longitude, device ID, measurement type) onto the result.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    pandas.DataFrame
        A flat DataFrame with one row per eDNA asset, merged with station
        metadata columns ``project_system_record_id``, ``device_id``,
        ``measurement_type``, ``latitude``, and ``longitude``.

    Examples
    --------
    >>> df = get_edna_observation_data(hdr)  # doctest: +SKIP
    >>> df.head()  # doctest: +SKIP
    """
    stations = get_station_info(hdr, "eDNA")
    station_lookup = _build_camera_trap_station_lookup(stations, "audio")  # type: ignore[arg-type]
    if station_lookup.empty:
        return pd.DataFrame(
            columns=[
                "project_system_record_id",
                "device_id",
                "measurement_type",
                "latitude",
                "longitude",
            ]
        )

    frames: list[pd.DataFrame] = []
    for psr_id in station_lookup["project_system_record_id"].tolist():
        try:
            assets = get_edna_assets(hdr, project_system_record_id=psr_id)
        except Exception:
            continue
        if not assets.empty:
            assets["project_system_record_id"] = psr_id
            frames.append(assets)

    if not frames:
        return station_lookup.reset_index(drop=True)

    merged_df = pd.concat(frames, ignore_index=True, sort=False)

    station_merge_columns = ["project_system_record_id"] + [
        col for col in station_lookup.columns if col not in merged_df.columns
    ]
    merged_df = merged_df.merge(
        station_lookup[station_merge_columns],
        on="project_system_record_id",
        how="left",
    )

    if "data_type" not in merged_df.columns:
        merged_df["data_type"] = "eDNA"

    return merged_df.reset_index(drop=True)


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


def _normalise_species_frame(df: pd.DataFrame, data_type_fallback: str) -> pd.DataFrame:
    """Ensure core species observation columns are present, filling missing ones with NaN."""
    if df.empty:
        return pd.DataFrame(columns=_SPECIES_OBS_CORE_COLUMNS)

    out = df.copy()

    # eDNA records expose 'species' but may not have 'label'; unify them.
    if "label" not in out.columns:
        if "species" in out.columns:
            out["label"] = out["species"]
        else:
            out["label"] = pd.NA

    if "data_type" not in out.columns:
        out["data_type"] = data_type_fallback

    for col in _SPECIES_OBS_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out


def get_all_species_observations(
    hdr: AuthHeaders,
) -> pd.DataFrame:
    """Retrieve all species identifications across every measurement type.

    Calls :func:`get_camera_trap_data`, :func:`get_audio_observation_data`,
    and :func:`get_edna_observation_data`, normalises their columns, and
    concatenates the results into a single flat DataFrame.

    Each row represents one species identification with the station location
    and measurement context it came from.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with at least the columns:
        ``project_system_record_id``, ``device_id``, ``data_type``,
        ``measurement_type``, ``latitude``, ``longitude``,
        ``label``, ``label_id``, ``common_name``,
        ``species``, ``genus``, ``family``, ``order``.
        Additional columns from each source are preserved.

    Examples
    --------
    >>> obs = get_all_species_observations(hdr)  # doctest: +SKIP
    >>> obs[["label", "measurement_type", "latitude", "longitude"]].head()  # doctest: +SKIP
    """
    frames: list[pd.DataFrame] = []

    camera_df = get_camera_trap_data(hdr)
    frames.append(_normalise_species_frame(camera_df, "image"))

    audio_df = get_audio_observation_data(hdr)
    frames.append(_normalise_species_frame(audio_df, "audio"))

    edna_df = get_edna_observation_data(hdr)
    frames.append(_normalise_species_frame(edna_df, "eDNA"))

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=_SPECIES_OBS_CORE_COLUMNS)

    return pd.concat(non_empty, ignore_index=True, sort=False)


def get_project_labels(
    hdr: AuthHeaders,
    labeltype: LabelType,
) -> list[SpeciesLight]:
    """Retrieve project-specific labels.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        Either ``"Bioacoustic"``, ``"Camera"``, or ``"Observation"``.

    Returns
    -------
    list[SpeciesLight]
        A list of project labels with species information.

    Examples
    --------
    >>> labels = get_project_labels(hdr, "Camera")  # doctest: +SKIP
    """
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return [SpeciesLight.model_validate(item) for item in response.json()]


def get_project_labels_df(
    hdr: AuthHeaders,
    labeltype: LabelType,
) -> pd.DataFrame:
    """Retrieve project-specific labels as a DataFrame.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        Either ``"Bioacoustic"``, ``"Camera"``, or ``"Observation"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing project labels.

    Examples
    --------
    >>> labels = get_project_labels_df(hdr, "Camera")  # doctest: +SKIP
    """
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return pd.DataFrame(response.json())


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
        Either ``"Bioacoustic"``, ``"Camera"``, or ``"Observation"``.
    labels:
        A list of Label objects to add.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> add_project_labels(hdr, "Camera", labels=[Label(...)])  # doctest: +SKIP
    """
    url = f"{hdr.root}addProjectLabels/{labeltype}/{hdr.key}"
    payload = [label.model_dump(mode="json") for label in labels]
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
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
        Maximum number of records to return (max 20000).
    search_term:
        Optional search term to filter results.

    Returns
    -------
    SpeciesTable
        A SpeciesTable with ``table`` (list of SpeciesLight) and ``pagination_state``.

    Raises
    ------
    ValueError
        If ``limit`` exceeds 20000.

    Examples
    --------
    >>> result = get_iucn_labels(hdr, offset=0, limit=100, search_term="horse")  # doctest: +SKIP
    """
    if limit > 20000:
        raise ValueError("limit cannot be greater than 20000")

    params: dict[str, Any] = {
        "offset": offset,
        "limit": limit,
        "search_term": search_term or "",
    }
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
        A list of IUCNSpeciesLabelInput objects to add.
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
        print("Data is too small to chunk, submitting all data")
        chunks = [labels]
    else:
        if chunksize > n:
            print(f"chunksize is bigger than length of data, altering chunksize to {n // 2}")
            chunksize = n // 2
        num_chunks = math.ceil(n / chunksize)
        chunks = [labels[i * chunksize:(i + 1) * chunksize] for i in range(num_chunks)]

    resp = None
    url = f"{hdr.root}addIUCNLabels/{hdr.key}"
    for i, chunk in enumerate(chunks, start=1):
        payload = [label.model_dump(mode="json", by_alias=True) for label in chunk]
        response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
        response.raise_for_status()
        resp = response.json()
        print(f"submitted {min(i * chunksize, n)} labels of {n}")

    return resp


def update_segment_labels(
    hdr: AuthHeaders,
    labels: list[Label],
) -> dict[str, str]:
    """Update segment labels.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labels:
        A list of Label objects with updated information.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> update_segment_labels(hdr, labels)  # doctest: +SKIP
    """
    url = f"{hdr.root}updateSegmentLabels/{hdr.key}"
    payload = [label.model_dump(mode="json") for label in labels]
    response = httpx.put(url, json=payload, timeout=_DEFAULT_TIMEOUT)
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
        A list of Label objects to submit.
    chunksize:
        Number of records per chunk.

    Examples
    --------
    >>> push_new_labels(hdr, submission_records, chunksize=30)  # doctest: +SKIP
    """
    n = len(submission_records)
    if chunksize > n:
        print(f"chunksize is bigger than length of data, altering chunksize to {n}")
        chunksize = n

    num_chunks = math.ceil(n / chunksize)
    for i in range(num_chunks):
        chunk = submission_records[i * chunksize:(i + 1) * chunksize]
        update_segment_labels(hdr, chunk)
        print(f"submitted {min((i + 1) * chunksize, n)} labels of {n}")


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
        A list of MediaTimestampUpdate objects with ``media_file_record_id``
        and ``new_timestamp``.

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
    print(f"Successfully updated {len(result)} media timestamp(s)")
    return result


def push_new_timestamps(
    hdr: AuthHeaders,
    media_metadata: list[MediaTimestampUpdate],
    chunksize: int,
) -> None:
    """Update timestamps for multiple media records by splitting into chunks.

    Recommended for large datasets (>1000 records) to prevent timeouts.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    media_metadata:
        A list of MediaTimestampUpdate objects.
    chunksize:
        Number of records per chunk. Recommended: 50–200.

    Examples
    --------
    >>> push_new_timestamps(hdr, updates, chunksize=100)  # doctest: +SKIP
    """
    n = len(media_metadata)
    if chunksize > n:
        print(f"chunksize is bigger than length of data, altering chunksize to {n}")
        chunksize = n

    num_chunks = math.ceil(n / chunksize)
    for i in range(num_chunks):
        chunk = media_metadata[i * chunksize:(i + 1) * chunksize]
        update_media_timestamps(hdr, chunk)
        print(f"submitted {min((i + 1) * chunksize, n)} timestamps of {n}")


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
    status_str = str(blank_status).lower()
    url = f"{hdr.root}segmentLabelsBlankStatus/{hdr.key}/{status_str}"
    response = httpx.put(url, json=segment_record_ids, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    resp = response.json()
    print(resp.get("message", ""))
    return resp


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
        A list of eDNAUploadSchema objects with eDNA records.

    Returns
    -------
    list[eDNAUploadResponse]
        The validated records with ``label``, ``label_id``, ``status``, and ``message``.

    Examples
    --------
    >>> validated = check_edna_labels(hdr, edna_records)  # doctest: +SKIP
    """
    url = f"{hdr.root}checkeDNALabels/{hdr.key}"
    payload = [record.model_dump(mode="json", by_alias=True) for record in edna_data]
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = [eDNAUploadResponse.model_validate(item) for item in response.json()]
    print(f"Validated {len(result)} eDNA records")
    return result


def check_edna_labels_df(
    hdr: AuthHeaders,
    edna_data: pd.DataFrame,
) -> pd.DataFrame:
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
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = edna_data.copy()

    if "confidence" not in data.columns:
        data["confidence"] = 100

    # Handle class_ vs class column naming
    if "class_" in data.columns and "class" not in data.columns:
        data["class"] = data["class_"]

    # Convert to list of dicts, dropping NaN values
    records = [
        {k: v for k, v in row.items() if pd.notna(v)}
        for row in data.to_dict(orient="records")
    ]

    url = f"{hdr.root}checkeDNALabels/{hdr.key}"
    response = httpx.post(url, json=records, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = pd.DataFrame(response.json())
    print(f"Validated {len(result)} eDNA records")
    return result


def upload_edna_records(
    hdr: AuthHeaders,
    validated_data: list[eDNAUploadResponse],
    project_system_record_id: int,
) -> list[eDNAUploadResponse]:
    """Upload validated eDNA records to a project system record.

    Only records with ``status == "success"`` will be uploaded.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    validated_data:
        A list of validated eDNAUploadResponse objects from :func:`check_edna_labels`.
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

    if len(successful) == 0:
        raise ValueError("No successful records to upload. All records failed validation.")

    print(f"Uploading {len(successful)} validated eDNA records")

    url = f"{hdr.root}uploadeDNA/{hdr.key}/{project_system_record_id}"
    payload = [record.model_dump(mode="json", by_alias=True) for record in successful]
    response = httpx.post(url, json=payload, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    result = [eDNAUploadResponse.model_validate(item) for item in response.json()]
    print(f"Upload complete: {len(result)} records processed")
    return result
