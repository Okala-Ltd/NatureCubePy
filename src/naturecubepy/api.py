"""
Core API wrapper functions for the Okala dashboard.

This module provides functions for authenticating with the Okala API and
interacting with project data including stations, media assets, labels,
eDNA records, and timestamps.
"""

from __future__ import annotations

import math
import os
from typing import Any

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

    # Compute map centre from station geometries
    centroids = geojson_response.geometry.centroid
    centre_lat = centroids.y.mean()
    centre_lon = centroids.x.mean()

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10)

    record_counts = geojson_response.get("record_count", pd.Series([1] * len(geojson_response)))
    min_count = record_counts.min() if not record_counts.empty else 1
    max_count = record_counts.max() if not record_counts.empty else 1

    def _rescale(value: float, new_min: float = 5, new_max: float = 15) -> float:
        if max_count == min_count:
            return (new_min + new_max) / 2
        return new_min + (value - min_count) / (max_count - min_count) * (new_max - new_min)

    for _, row in geojson_response.iterrows():
        lat = row.geometry.centroid.y
        lon = row.geometry.centroid.x
        device_id = row.get("device_id", "")
        start_ts = row.get("project_system_record_start_timestamp", "")
        end_ts = row.get("project_system_record_end_timestamp", "")
        count = row.get("record_count", 1)

        popup_html = (
            f"QR code: {device_id}<br>"
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
    psr_ids: list[int],
) -> pd.DataFrame:
    """Retrieve media assets as a DataFrame for given project system record IDs.

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
    pandas.DataFrame
        A DataFrame of media assets.

    Examples
    --------
    >>> assets = get_media_assets_df(hdr, "video", psr_ids=[123])  # doctest: +SKIP
    """
    url = f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}"
    response = httpx.post(url, json=psr_ids, timeout=_DEFAULT_TIMEOUT)
    response.raise_for_status()
    return pd.DataFrame(response.json())


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
