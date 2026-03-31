"""
Core API wrapper functions for the Okala dashboard.

This module provides functions for authenticating with the Okala API and
interacting with project data including stations, media assets, labels,
eDNA records, and timestamps.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

import geopandas as gpd
import httpx
import pandas as pd

_PROD_URL = "https://api.dashboard.okala.io/api/"
_DEV_URL = "https://dev.api.dashboard.okala.io/api/"


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
) -> dict[str, str]:
    """Create an authentication context for the production Okala API.

    Parameters
    ----------
    api_key:
        A valid Okala project API key.
    okala_url:
        Base URL for the Okala API. Defaults to the production endpoint.

    Returns
    -------
    dict
        A dict with keys ``"key"`` (the API key) and ``"root"`` (the base URL).

    Examples
    --------
    >>> hdr = auth_headers("mykey")
    >>> hdr["root"]
    'https://api.dashboard.okala.io/api/'
    """
    return {"key": api_key, "root": okala_url.rstrip("/") + "/"}


def auth_headers_dev(
    api_key: str,
    okala_url: str = _DEV_URL,
) -> dict[str, str]:
    """Create an authentication context for the development Okala API.

    Parameters
    ----------
    api_key:
        A valid Okala project API key.
    okala_url:
        Base URL for the Okala dev API. Defaults to the development endpoint.

    Returns
    -------
    dict
        A dict with keys ``"key"`` (the API key) and ``"root"`` (the base URL).

    Examples
    --------
    >>> hdr = auth_headers_dev("mykey")
    >>> hdr["root"]
    'https://dev.api.dashboard.okala.io/api/'
    """
    return {"key": api_key, "root": okala_url.rstrip("/") + "/"}


def get_project(hdr: dict[str, str]) -> None:
    """Retrieve and display the active project name associated with the API key.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.

    Returns
    -------
    None
        Prints a message with the active project name.

    Examples
    --------
    >>> hdr = auth_headers("mykey")
    >>> get_project(hdr)  # doctest: +SKIP
    Setting your active project as - My Project
    """
    url = f"{hdr['root']}getProject/{hdr['key']}"
    response = httpx.get(url)
    response.raise_for_status()
    data = response.json()
    project_name = data["boundary"]["features"][0]["properties"]["project_name"]
    print(f"Setting your active project as - {project_name}")


def get_station_info(
    hdr: dict[str, str],
    datatype: str,
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
    url = f"{hdr['root']}getStations/{datatype}/{hdr['key']}"
    response = httpx.get(url)
    response.raise_for_status()
    return gpd.read_file(response.text)


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


def get_media_assets(
    hdr: dict[str, str],
    datatype: str,
    psr_id: int,
) -> pd.DataFrame:
    """Retrieve media assets for a given project system record ID.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    datatype:
        One of ``"video"``, ``"audio"``, ``"image"``, or ``"eDNA"``.
    psr_id:
        The unique project system record ID.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of media assets for the specified project system record.

    Examples
    --------
    >>> assets = get_media_assets(hdr, "video", psr_id=123)  # doctest: +SKIP
    """
    url = f"{hdr['root']}getMediaAssets/{datatype}/{hdr['key']}"
    response = httpx.post(url, json=psr_id)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def get_project_labels(
    hdr: dict[str, str],
    labeltype: str,
) -> pd.DataFrame:
    """Retrieve project-specific labels.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        Either ``"Bioacoustic"`` or ``"Camera"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing project labels.

    Examples
    --------
    >>> labels = get_project_labels(hdr, "Camera")  # doctest: +SKIP
    """
    url = f"{hdr['root']}getProjectLabels/{labeltype}/{hdr['key']}"
    response = httpx.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json())


def add_project_labels(
    hdr: dict[str, str],
    labeltype: str,
    labels: list[dict[str, Any]],
) -> Any:
    """Add labels to the project.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labeltype:
        Either ``"Bioacoustic"`` or ``"Camera"``.
    labels:
        A list of label objects to add.

    Returns
    -------
    Any
        The API response as a parsed JSON object.

    Examples
    --------
    >>> add_project_labels(hdr, "Camera", labels=[{...}])  # doctest: +SKIP
    """
    url = f"{hdr['root']}addProjectLabels/{labeltype}/{hdr['key']}"
    response = httpx.post(url, json=labels)
    response.raise_for_status()
    return response.json()


def get_iucn_labels(
    hdr: dict[str, str],
    offset: int,
    limit: int,
    search_term: str | None = None,
) -> dict[str, Any]:
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
    dict
        A dict with keys ``"data"`` (DataFrame), ``"total"``, ``"offset"``,
        and ``"limit"``.

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
    url = f"{hdr['root']}getIUCNLabels/{hdr['key']}"
    response = httpx.get(url, params=params)
    response.raise_for_status()
    resp = response.json()

    rows = resp.get("table", [])
    cleaned = [{k: v for k, v in row.items() if v is not None} for row in rows]
    data = pd.DataFrame(cleaned)

    pagination = resp.get("pagination_state", {})
    return {
        "data": data,
        "total": pagination.get("total"),
        "offset": pagination.get("offset"),
        "limit": pagination.get("limit"),
    }


def add_iucn_labels(
    hdr: dict[str, str],
    labels: pd.DataFrame,
    chunksize: int,
) -> Any:
    """Add labels from the IUCN database in chunks.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    labels:
        A DataFrame of labels to add.
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
        chunks = [labels.iloc[i * chunksize:(i + 1) * chunksize] for i in range(num_chunks)]

    resp = None
    url = f"{hdr['root']}addIUCNLabels/{hdr['key']}"
    for i, chunk in enumerate(chunks, start=1):
        response = httpx.post(url, json=chunk.to_dict(orient="records"))
        response.raise_for_status()
        resp = response.json()
        print(f"submitted {i * chunksize} labels of {n}")

    return resp


def _send_updated_labels(hdr: dict[str, str], data_chunk: pd.DataFrame) -> Any:
    """Internal helper: send a single chunk of updated segment labels."""
    url = f"{hdr['root']}updateSegmentLabels/{hdr['key']}"
    response = httpx.put(url, json=data_chunk.to_dict(orient="records"))
    response.raise_for_status()
    return response.json()


def push_new_labels(
    hdr: dict[str, str],
    submission_records: pd.DataFrame,
    chunksize: int,
) -> None:
    """Push new segment labels to the platform in chunks.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    submission_records:
        A DataFrame containing the records to submit.
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
        chunk = submission_records.iloc[i * chunksize:(i + 1) * chunksize]
        _send_updated_labels(hdr, chunk)
        print(f"submitted {min((i + 1) * chunksize, n)} labels of {n}")


def _send_media_chunk(hdr: dict[str, str], data_chunk: pd.DataFrame) -> Any:
    """Internal helper: send a single chunk of timestamp updates."""
    url = f"{hdr['root']}updateTimestamps/{hdr['key']}"
    response = httpx.put(url, json=data_chunk.to_dict(orient="records"))
    response.raise_for_status()
    return response.json()


_ISO8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:?\d{2})?$"
)


def update_media_timestamps(
    hdr: dict[str, str],
    media_records: pd.DataFrame,
) -> Any:
    """Update timestamps for one or more media file records in a single API call.

    For large datasets (>1000 records) consider using :func:`push_new_timestamps`
    which handles chunking automatically.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    media_records:
        A DataFrame with columns ``media_file_record_id`` (int) and
        ``new_timestamp`` (ISO 8601 string, e.g. ``"2024-01-15T10:30:00Z"``).

    Returns
    -------
    Any
        The API response as a parsed JSON object.

    Raises
    ------
    ValueError
        If required columns are missing or values are invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> updates = pd.DataFrame({
    ...     "media_file_record_id": [123, 456],
    ...     "new_timestamp": ["2024-01-15T10:30:00Z", "2024-01-15T14:20:00Z"],
    ... })
    >>> result = update_media_timestamps(hdr, updates)  # doctest: +SKIP
    """
    required_cols = {"media_file_record_id", "new_timestamp"}
    missing = required_cols - set(media_records.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if not pd.api.types.is_numeric_dtype(media_records["media_file_record_id"]):
        raise ValueError("media_file_record_id must be numeric")
    if media_records["media_file_record_id"].isna().any():
        raise ValueError("media_file_record_id cannot contain NA values")
    if (media_records["media_file_record_id"] <= 0).any():
        raise ValueError("media_file_record_id must be positive")
    if (media_records["media_file_record_id"] % 1 != 0).any():
        raise ValueError("media_file_record_id must be an integer")

    if not pd.api.types.is_string_dtype(media_records["new_timestamp"]):
        raise ValueError("new_timestamp must be a string in ISO 8601 format")

    invalid_mask = media_records["new_timestamp"].notna() & ~media_records["new_timestamp"].str.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:?\d{2})?$"
    )
    if invalid_mask.any():
        invalid_rows = list(media_records.index[invalid_mask] + 1)
        raise ValueError(
            "new_timestamp must be in ISO 8601 format, e.g. '2024-01-31T23:59:59Z'. "
            f"Invalid values at row(s): {invalid_rows}"
        )

    subset = media_records[list(required_cols)].copy()
    url = f"{hdr['root']}updateTimestamps/{hdr['key']}"
    response = httpx.put(url, json=subset.to_dict(orient="records"))
    response.raise_for_status()
    result = response.json()
    print(f"Successfully updated {len(subset)} media timestamp(s)")
    return result


def push_new_timestamps(
    hdr: dict[str, str],
    media_metadata: pd.DataFrame,
    chunksize: int,
) -> None:
    """Update timestamps for multiple media records by splitting into chunks.

    Recommended for large datasets (>1000 records) to prevent timeouts.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    media_metadata:
        A DataFrame with columns ``media_file_record_id`` and ``new_timestamp``.
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
        chunk = media_metadata.iloc[i * chunksize:(i + 1) * chunksize]
        _send_media_chunk(hdr, chunk)
        print(f"submitted {min((i + 1) * chunksize, n)} timestamps of {n}")


def set_segment_blank_status(
    hdr: dict[str, str],
    blank_status: bool,
    segment_record_ids: list[int],
) -> Any:
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
    Any
        The API response as a parsed JSON object.

    Examples
    --------
    >>> set_segment_blank_status(hdr, blank_status=True, segment_record_ids=[101, 102])  # doctest: +SKIP
    """
    status_str = str(blank_status).lower()
    url = f"{hdr['root']}segmentLabelsBlankStatus/{hdr['key']}/{status_str}"
    response = httpx.put(url, json=segment_record_ids)
    response.raise_for_status()
    resp = response.json()
    print(resp.get("message", ""))
    return resp


def check_edna_labels(
    hdr: dict[str, str],
    edna_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate eDNA records against the Okala database.

    Uses a hierarchical taxonomy approach:
    species → genus → family → order → class → phylum → kingdom.

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
    >>> validated = check_edna_labels(hdr, edna_records)  # doctest: +SKIP
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

    url = f"{hdr['root']}checkeDNALabels/{hdr['key']}"
    response = httpx.post(url, json=records)
    response.raise_for_status()
    result = pd.DataFrame(response.json())
    print(f"Validated {len(result)} eDNA records")
    return result


def upload_edna_records(
    hdr: dict[str, str],
    validated_data: pd.DataFrame,
    project_system_record_id: int,
) -> pd.DataFrame:
    """Upload validated eDNA records to a project system record.

    Only records with ``status == "success"`` (from :func:`check_edna_labels`)
    are uploaded.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    validated_data:
        A DataFrame of validated eDNA records from :func:`check_edna_labels`.
    project_system_record_id:
        The project system record ID to upload records to.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the upload response for each record.

    Raises
    ------
    ValueError
        If the data has not been validated or no successful records exist.

    Examples
    --------
    >>> validated = check_edna_labels(hdr, edna_records)  # doctest: +SKIP
    >>> result = upload_edna_records(hdr, validated, project_system_record_id=123)  # doctest: +SKIP
    """
    if "status" not in validated_data.columns:
        raise ValueError("Data must be validated first using check_edna_labels()")

    successful = validated_data[validated_data["status"] == "success"]

    if len(successful) == 0:
        raise ValueError("No successful records to upload. All records failed validation.")

    print(f"Uploading {len(successful)} validated eDNA records")

    records = [
        {k: v for k, v in row.items() if pd.notna(v)}
        for row in successful.to_dict(orient="records")
    ]

    url = f"{hdr['root']}uploadeDNA/{hdr['key']}/{project_system_record_id}"
    response = httpx.post(url, json=records)
    response.raise_for_status()
    result = pd.DataFrame(response.json())
    print(f"Upload complete: {len(result)} records processed")
    return result
