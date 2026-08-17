"""
Core API wrapper functions for the Okala dashboard.

This module provides functions for authenticating with the Okala API and
interacting with project data including stations, media assets, labels,
eDNA records, and timestamps.
"""


import collections
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Iterator
import httpx
import pandas as pd
from typing import Any, Callable, Iterator, TypeVar

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
    SegmentRecordAPIFlat,
    MediaTimestampUpdate,
    SPECIES_OBS_CORE_COLUMNS,
    STATION_LOOKUP_COLUMNS,
    SpeciesTable,
    StationResponseAPI,
    TimestampUpdateResponse,
    eDNAUploadResponse,
    eDNAUploadSchema,
)


def extract_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
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


def extract_total_count(payload: Any) -> int | None:
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

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def get_key(key_name='OKALA_API_KEY') -> str:
    """Retrieve the API key from the specified environment variable.

    Parameters
    ----------
    key_name : str
        The name of the environment variable containing the API key.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    EnvironmentError
        If the specified environment variable is not set.

    Examples
    --------
    >>> import os
    >>> os.environ["OKALA_API_KEY"] = "mykey"
    >>> get_key()
    'mykey'
    """
    api_key = os.environ.get(key_name, "")
    if not api_key:
        raise EnvironmentError(f"{key_name} environment variable not set.")
    return api_key


def normalise_api_root(okala_url: str) -> str:
    """Normalise wrapper base URLs so local hosts can omit the /api prefix."""
    parts = urlsplit(okala_url.strip())
    path = parts.path.rstrip("/")

    if not path:
        path = "/api"

    return urlunsplit((parts.scheme, parts.netloc, f"{path}/", parts.query, parts.fragment))


def auth_headers(api_key: str, okala_url: str = "https://naturecube.io/api/") -> AuthHeaders:
    """Create an authentication context for the Okala API.

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
    return AuthHeaders(key=api_key, root=normalise_api_root(okala_url))


# ---------------------------------------------------------------------------
# Project & stations
# ---------------------------------------------------------------------------


def get_project(hdr: AuthHeaders, timeout: float = 180.0) -> GetProjectGeometryResponse:
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


def _strip_null_altitude(coords: Any) -> Any:
    """Drop null Z values that pydantic-geojson injects into coordinates."""
    if (
        isinstance(coords, list)
        and coords
        and all(isinstance(value, (int, float, type(None))) for value in coords)
    ):
        return [value for value in coords if value is not None]
    if isinstance(coords, list):
        return [_strip_null_altitude(value) for value in coords]
    return coords


def _feature_for_geopandas(feature: dict[str, Any]) -> dict[str, Any]:
    """Normalise a GeoJSON feature dict for ``GeoDataFrame.from_features``."""
    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and "coordinates" in geometry:
        geometry = {
            **geometry,
            "coordinates": _strip_null_altitude(geometry["coordinates"]),
        }
        geometry.pop("bbox", None)
    return {**feature, "geometry": geometry}


def project_boundary_gdf(project: GetProjectGeometryResponse) -> gpd.GeoDataFrame:
    """Convert a :func:`get_project` response into a boundary GeoDataFrame (EPSG:4326).

    Parameters
    ----------
    project:
        Response from :func:`get_project`.

    Returns
    -------
    geopandas.GeoDataFrame
        Project boundary features with CRS set to EPSG:4326. Empty when the
        response has no boundary features.
    """
    features = [
        _feature_for_geopandas(feature.model_dump(mode="json"))
        for feature in project.boundary.features
    ]
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def get_project_boundary(
    hdr: AuthHeaders,
    timeout: float = 180.0,
    *,
    project: GetProjectGeometryResponse | None = None,
) -> gpd.GeoDataFrame:
    """Fetch the active project boundary as a GeoDataFrame.

    Uses the ``boundary`` FeatureCollection from ``GET /api/getProject/{api_key}``.
    Pass an existing ``project`` from :func:`get_project` to avoid a second request.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    timeout:
        Request timeout in seconds (ignored when ``project`` is provided).
    project:
        Optional cached :class:`GetProjectGeometryResponse` from :func:`get_project`.

    Returns
    -------
    geopandas.GeoDataFrame
        Project boundary polygon(s) in EPSG:4326, suitable for
        ``plot_stations_static(..., project_boundary=...)``.

    Examples
    --------
    >>> boundary = get_project_boundary(hdr)  # doctest: +SKIP
    >>> project = get_project(hdr)  # doctest: +SKIP
    >>> boundary = get_project_boundary(hdr, project=project)  # doctest: +SKIP
    """
    if project is None:
        project = get_project(hdr, timeout=timeout)
    return project_boundary_gdf(project)


def fetch_station_features(hdr: AuthHeaders, datatype: str) -> gpd.GeoDataFrame:
    """Fetch all station rows for a datatype as a GeoDataFrame, with pagination."""
    url = f"{hdr.root}getStations/{datatype}/{hdr.key}"
    offset = 0
    all_features: list[dict[str, Any]] = []
    while True:
        response = httpx.get(url, params={"offset": offset, "limit": 1000}, timeout=180.0)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", []) if isinstance(payload, dict) else []
        if not features:
            break
        all_features.extend(features)
        if len(features) < 1000:
            break
        offset += 1000
    return gpd.GeoDataFrame.from_features(all_features) if all_features else gpd.GeoDataFrame()


def get_station_info(hdr: AuthHeaders, measurement_type: str | None) -> pd.DataFrame:
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
    pandas.DataFrame
        A flat table containing station metadata with ``latitude`` and
        ``longitude`` columns instead of a geometry column.

    Raises
    ------
    ValueError
        If ``measurement_type`` is not a supported value.

    Examples
    --------
    >>> stations = get_station_info(hdr, measurement_type="camera")  # doctest: +SKIP
    >>> all_stations = get_station_info(hdr, measurement_type=None)  # doctest: +SKIP
    """
    if measurement_type is not None:
        mt = str(measurement_type).strip().lower()
        if mt not in ("all", "", "none") and mt not in {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}:
            _valid_keys = list({"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]})
            raise ValueError(
                f"Invalid measurement_type: {measurement_type!r}. "
                f"Must be one of {_valid_keys} or None/'all'."
            )
        requested = {mt: {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}[mt]} if mt in {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]} else {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}
    else:
        requested = {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}

    dfs = []
    for mt_key, datatypes in requested.items():
        for datatype in datatypes:
            gdf = fetch_station_features(hdr, datatype)
            if gdf.empty:
                continue
            if "measurement_type" in gdf.columns:
                mask = gdf["measurement_type"].astype(str).str.strip().str.lower() == mt_key
                dfs.append(gdf[mask])
            else:
                dfs.append(gdf)
    if not dfs:
        return pd.DataFrame()

    stations = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True))
    if "geometry" in stations.columns:
        stations["latitude"] = stations.geometry.y
        stations["longitude"] = stations.geometry.x
        stations = pd.DataFrame(stations.drop(columns=["geometry"]))
    return stations


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
    gdf = fetch_station_features(hdr, datatype)
    features = list(gdf.__geo_interface__["features"]) if not gdf.empty else []
    return StationResponseAPI.model_validate({"type": "FeatureCollection", "features": features})



# ---------------------------------------------------------------------------
# Media assets & segments
# ---------------------------------------------------------------------------

MEDIA_PAGE_TIMEOUT = 180.0
MEDIA_MAX_RETRIES = 6
MEDIA_RETRY_BASE_SECONDS = 2.0
# The API rate-limits aggressive concurrency (8 parallel page requests return
# HTTP 429), and per-request latency climbs as workers are added, so throughput
# peaks at a handful of workers.
MEDIA_MAX_WORKERS = 3


def _format_duration(seconds: float) -> str:
    """Format a rough ETA as ``45s``, ``12m``, or ``1h04m``."""
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class _ProgressBar:
    """Single-line progress bar for long downloads, safe to advance from threads."""

    def __init__(self, label: str, total: int, *, unit: str = "", width: int = 30) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.unit = unit
        self.width = width
        self.done = 0
        self._lock = threading.Lock()
        self._closed = False
        self._start = time.monotonic()
        self._draw()

    def advance(self, step: int = 1) -> None:
        with self._lock:
            self.done = min(self.done + step, self.total)
            self._draw()

    def notice(self, message: str) -> None:
        """Report a stall (retry, rate limit) without corrupting the bar line."""
        with self._lock:
            print(f"\r\x1b[2K{message}", flush=True)
            self._draw()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                print(flush=True)

    def _draw(self) -> None:
        pct = 100 * self.done / self.total
        unit = f" {self.unit}" if self.unit else ""
        prefix = f"{self.label} "
        suffix = f" {self.done}/{self.total}{unit} ({pct:3.0f}%)"

        elapsed = time.monotonic() - self._start
        if self.done and self.done < self.total and elapsed > 1:
            remaining = (self.total - self.done) * elapsed / self.done
            suffix += f" ETA {_format_duration(remaining)}"

        # Keep the whole line inside the terminal width: a line that wraps makes
        # the carriage return redraw on the wrapped fragment instead of the bar.
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        budget = columns - 1 - len(prefix) - len(suffix) - 2  # 2 for the |bars|
        bar_width = max(min(self.width, budget), 4)
        filled = round(bar_width * self.done / self.total)
        line = f"{prefix}|{'#' * filled}{'.' * (bar_width - filled)}|{suffix}"

        # \x1b[2K clears the row so a shorter line cannot leave stale characters.
        print(f"\r\x1b[2K{line[: columns - 1]}", end="", flush=True)


def _page_kind(fetch_page: Callable[..., Any]) -> str:
    """Name the record type a page fetcher returns, for progress messages."""
    name = getattr(fetch_page, "__name__", "")
    if "segments" in name:
        return "segments"
    if "assets" in name:
        return "media assets"
    return "records"


def _retry_after_seconds(response: httpx.Response, attempt: int) -> float:
    """Compute wait time for a rate-limited or transient failure."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(float(header), MEDIA_RETRY_BASE_SECONDS)
        except ValueError:
            pass
    return MEDIA_RETRY_BASE_SECONDS * (2 ** attempt)


def _post_media_page(
    url: str,
    *,
    psr_ids: list[int],
    limit: int,
    offset: int,
    timeout: float = MEDIA_PAGE_TIMEOUT,
    on_retry: Callable[[str], None] | None = None,
) -> httpx.Response:
    """POST a media page with retries for HTTP 429 rate limits."""
    params = {"limit": min(limit, 1000), "offset": offset}

    for attempt in range(MEDIA_MAX_RETRIES + 1):
        response = httpx.post(url, json=psr_ids, params=params, timeout=timeout)
        if response.status_code == 429 and attempt < MEDIA_MAX_RETRIES:
            wait = _retry_after_seconds(response, attempt)
            if on_retry is not None:
                on_retry(
                    f"Rate limited by the API; waiting {wait:.0f}s "
                    f"(retry {attempt + 1}/{MEDIA_MAX_RETRIES})"
                )
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response

    # Unreachable: final attempt either returns or raise_for_status()'s.
    raise RuntimeError("media page request failed after retries")


# ---------------------------------------------------------------------------
# Page fetchers
# ---------------------------------------------------------------------------

def fetch_media_assets_page(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
    *,
    limit: int = 1000,
    offset: int = 0,
    on_retry: Callable[[str], None] | None = None,
) -> tuple[list[MediaRecordAPIFlat], int | None]:
    response = _post_media_page(
        f"{hdr.root}getMediaAssets/{datatype}/{hdr.key}",
        psr_ids=psr_ids,
        limit=limit,
        offset=offset,
        on_retry=on_retry,
    )
    payload = response.json()
    return [MediaRecordAPIFlat.model_validate(row) for row in extract_rows_from_payload(payload)], extract_total_count(payload)


def fetch_media_segments_page(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
    *,
    limit: int = 1000,
    offset: int = 0,
    on_retry: Callable[[str], None] | None = None,
) -> tuple[list[SegmentRecordAPIFlat], int | None]:
    response = _post_media_page(
        f"{hdr.root}getMediaSegments/{datatype}/{hdr.key}",
        psr_ids=psr_ids,
        limit=limit,
        offset=offset,
        on_retry=on_retry,
    )
    payload = response.json()
    return [SegmentRecordAPIFlat.model_validate(row) for row in extract_rows_from_payload(payload)], extract_total_count(payload)

# ---------------------------------------------------------------------------
# Generic pagination core
# ---------------------------------------------------------------------------

T = TypeVar("T")

def _iter_pages(
    fetch_page: Callable[..., tuple[list[T], int | None]],
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
    *,
    page_size: int,
    chunk_size: int,
    bar: _ProgressBar | None = None,
    on_rows: Callable[[int], None] | None = None,
) -> Iterator[T]:
    notify = bar.notice if bar is not None else None

    for chunk_start in range(0, len(psr_ids), chunk_size):
        chunk = psr_ids[chunk_start : chunk_start + chunk_size]
        offset = 0
        total: int | None = None

        while True:
            try:
                rows, page_total = fetch_page(
                    hdr, datatype, chunk, limit=page_size, offset=offset, on_retry=notify
                )
            except httpx.TimeoutException:
                if len(chunk) <= 1:
                    # Single-station timeouts: brief retry before failing hard.
                    retried = False
                    for attempt in range(2):
                        wait = MEDIA_RETRY_BASE_SECONDS * (2 ** attempt)
                        if bar is not None:
                            bar.notice(
                                f"PSR {chunk[0]} timed out; retrying in {wait:.0f}s "
                                f"({attempt + 1}/2)"
                            )
                        time.sleep(wait)
                        try:
                            rows, page_total = fetch_page(
                                hdr, datatype, chunk, limit=page_size, offset=offset,
                                on_retry=notify,
                            )
                            retried = True
                            break
                        except httpx.TimeoutException:
                            continue
                    if not retried:
                        raise
                else:
                    mid = max(1, len(chunk) // 2)
                    yield from _iter_pages(fetch_page, hdr, datatype, chunk[:mid],
                                           page_size=page_size, chunk_size=chunk_size,
                                           bar=bar, on_rows=on_rows)
                    yield from _iter_pages(fetch_page, hdr, datatype, chunk[mid:],
                                           page_size=page_size, chunk_size=chunk_size,
                                           bar=bar, on_rows=on_rows)
                    break

            if page_total is not None:
                total = page_total

            if not rows:
                break

            yield from rows
            offset += len(rows)
            if on_rows is not None:
                on_rows(len(rows))

            if total is not None:
                if offset >= total:
                    break
            elif len(rows) < page_size:
                break


def _iter_pages_parallel(
    fetch_page: Callable[..., tuple[list[T], int | None]],
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: list[int],
    *,
    page_size: int,
    chunk_size: int,
    max_workers: int,
    total_rows: int | None = None,
) -> Iterator[T]:
    """Fetch chunks concurrently across stations, preserving input order.

    Each chunk is paginated sequentially (offsets depend on prior pages) by
    reusing :func:`_iter_pages`, but independent chunks run in parallel threads.
    Since the page fetches are network-bound HTTP calls, this gives a large
    speed-up over fetching one station at a time.

    When ``total_rows`` is known (from station record counts) the progress bar
    counts rows, which moves every page instead of only when a whole station
    finishes.
    """
    chunks = [
        psr_ids[start : start + chunk_size]
        for start in range(0, len(psr_ids), chunk_size)
    ]
    workers = 1 if max_workers <= 1 or len(chunks) <= 1 else min(max_workers, len(chunks))
    label = f"{datatype} {_page_kind(fetch_page)}"

    row_mode = bool(total_rows and total_rows > 0)
    if row_mode:
        bar = _ProgressBar(label, int(total_rows), unit="rows")
    elif len(chunks) > 1:
        # Single-station calls (e.g. camera trap's per-station loop) skip the bar
        # so an outer progress bar can own the display.
        bar = _ProgressBar(label, len(chunks), unit="stations")
    else:
        bar = None
    on_rows = bar.advance if (bar is not None and row_mode) else None

    def chunk_done() -> None:
        if bar is not None and not row_mode:
            bar.advance()

    try:
        if workers == 1:
            for chunk in chunks:
                yield from _iter_pages(
                    fetch_page, hdr, datatype, chunk,
                    page_size=page_size, chunk_size=chunk_size,
                    bar=bar, on_rows=on_rows,
                )
                chunk_done()
            return

        def collect(chunk: list[int]) -> list[T]:
            return list(
                _iter_pages(
                    fetch_page, hdr, datatype, chunk,
                    page_size=page_size, chunk_size=chunk_size,
                    bar=bar, on_rows=on_rows,
                )
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(collect, chunk) for chunk in chunks]
            # Advance on completion so the bar tracks real progress, but yield in
            # submission order so rows stay ordered by station.
            for future in futures:
                future.add_done_callback(lambda _f: chunk_done())
            for future in futures:
                yield from future.result()
    finally:
        if bar is not None:
            bar.close()

# ---------------------------------------------------------------------------
# Public iterators
# ---------------------------------------------------------------------------

def iter_media_assets(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
    total_rows: int | None = None,
) -> Iterator[MediaRecordAPIFlat]:
    yield from _iter_pages_parallel(
        fetch_media_assets_page, hdr, datatype, normalise_psr_ids(psr_ids),
        page_size=page_size, chunk_size=chunk_size, max_workers=max_workers,
        total_rows=total_rows,
    )


def iter_media_segments(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
    total_rows: int | None = None,
) -> Iterator[SegmentRecordAPIFlat]:
    yield from _iter_pages_parallel(
        fetch_media_segments_page, hdr, datatype, normalise_psr_ids(psr_ids),
        page_size=page_size, chunk_size=chunk_size, max_workers=max_workers,
        total_rows=total_rows,
    )

# ---------------------------------------------------------------------------
# Convenience collectors
# ---------------------------------------------------------------------------

def get_media_assets(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
) -> list[MediaRecordAPIFlat]:
    return list(iter_media_assets(hdr, datatype, psr_ids, page_size=page_size, chunk_size=chunk_size, max_workers=max_workers))

def get_media_segments(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
) -> list[SegmentRecordAPIFlat]:
    return list(iter_media_segments(hdr, datatype, psr_ids, page_size=page_size, chunk_size=chunk_size, max_workers=max_workers))

def get_media_assets_df(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
    total_rows: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        row.model_dump()
        for row in iter_media_assets(hdr, datatype, psr_ids, page_size=page_size, chunk_size=chunk_size, max_workers=max_workers, total_rows=total_rows)
    )

def get_media_segments_df(
    hdr: AuthHeaders,
    datatype: DataTypes,
    psr_ids: int | list[int],
    *,
    page_size: int = 1000,
    chunk_size: int = 1,
    max_workers: int = MEDIA_MAX_WORKERS,
) -> pd.DataFrame:
    return pd.DataFrame(
        row.model_dump()
        for row in iter_media_segments(hdr, datatype, psr_ids, page_size=page_size, chunk_size=chunk_size, max_workers=max_workers)
    )

# ---------------------------------------------------------------------------
# Observation data helpers
# ---------------------------------------------------------------------------


def normalise_psr_ids(project_system_record_ids: int | list[int]) -> list[int]:
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


def build_station_lookup(stations: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Build a flat station lookup table with location metadata."""
    if stations.empty:
        return pd.DataFrame(columns=STATION_LOOKUP_COLUMNS)
    if "project_system_record_id" not in stations.columns:
        raise ValueError(
            "Station data does not contain 'project_system_record_id'. "
            "Cannot link media assets to stations."
        )
    lookup = stations.dropna(subset=["project_system_record_id"]).copy()
    if lookup.empty:
        return pd.DataFrame(columns=STATION_LOOKUP_COLUMNS)

    lookup["project_system_record_id"] = lookup["project_system_record_id"].astype(int)
    if "geometry" in lookup.columns:
        lookup["latitude"] = lookup.geometry.y
        lookup["longitude"] = lookup.geometry.x
    elif not {"latitude", "longitude"}.issubset(lookup.columns):
        raise ValueError(
            "Station data must contain either a geometry column or latitude "
            "and longitude columns."
        )
    if "data_type" not in lookup.columns:
        lookup["data_type"] = data_type

    available = [c for c in STATION_LOOKUP_COLUMNS if c in lookup.columns]
    return lookup[available].drop_duplicates(subset=["project_system_record_id"])


def resolve_psr_id_column(df: pd.DataFrame) -> pd.DataFrame:
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


def merge_segments(media_df: pd.DataFrame, segments_df: pd.DataFrame | list[object]) -> pd.DataFrame:
    """Left-join segment columns onto media rows, preferring segment values.

    Some backends return overlapping fields (for example ``label`` and
    ``species``) in both media-assets and segment payloads. Segment rows can
    contain newer/verified values, so segment values should override media
    values when present.
    """
    if isinstance(segments_df, list):
        if not segments_df:
            return media_df

        rows: list[dict[str, object]] = []
        for row in segments_df:
            if hasattr(row, "model_dump"):
                rows.append(row.model_dump(mode="json"))
            elif isinstance(row, dict):
                rows.append(row)
            else:
                rows.append(vars(row))
        segments_df = pd.DataFrame(rows)

    if not isinstance(segments_df, pd.DataFrame):
        raise TypeError("segments_df must be a pandas DataFrame or list of segment records.")

    if segments_df.empty:
        return media_df

    if "segment_record_id" not in segments_df.columns or "segment_record_id" not in media_df.columns:
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


def merge_station_lookup(media_df: pd.DataFrame, station_lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-join station location metadata onto a media DataFrame."""
    station_cols = ["project_system_record_id"] + [
        c for c in station_lookup.columns if c not in media_df.columns
    ]
    return media_df.merge(station_lookup[station_cols], on="project_system_record_id", how="left")


def fetch_and_merge_edna_assets(hdr: AuthHeaders, station_lookup: pd.DataFrame) -> pd.DataFrame:
    """Fetch eDNA assets for a station lookup and merge location metadata."""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    psr_ids = station_lookup["project_system_record_id"].tolist()
    bar = _ProgressBar("eDNA assets", len(psr_ids)) if len(psr_ids) > 1 else None
    try:
        for psr_id in psr_ids:
            assets = pd.DataFrame()
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
            finally:
                if bar is not None:
                    bar.advance()
            if not assets.empty:
                assets = assets.copy()
                assets["project_system_record_id"] = psr_id
                frames.append(assets)
    finally:
        if bar is not None:
            bar.close()

    if not frames:
        if errors:
            print(
                f"Warning: could not retrieve eDNA species labels ({len(errors)} station(s) failed). "
                f"First error: {errors[0]}"
            )
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merge_station_lookup(merged, station_lookup)
    if "data_type" not in merged.columns:
        merged["data_type"] = "eDNA"
    return merged


def fetch_observations_for_datatype(
    hdr: AuthHeaders,
    datatype: str,
    stations: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch observation rows for a single datatype and attach station metadata."""
    if "data_type" in stations.columns and str(datatype).strip().lower() != "all":
        scoped_stations = stations[stations["data_type"].astype(str).str.strip().str.lower() == str(datatype).lower()]
    else:
        scoped_stations = stations
    if str(datatype).lower() == "edna":
        if "record_count" in scoped_stations.columns:
            counts = pd.to_numeric(scoped_stations["record_count"], errors="coerce").fillna(0)
            scoped_stations = scoped_stations[counts > 0]
    station_lookup = build_station_lookup(scoped_stations, datatype)
    if station_lookup.empty:
        return pd.DataFrame(), station_lookup

    if str(datatype).lower() == "edna":
        merged = fetch_and_merge_edna_assets(hdr, station_lookup)
    else:
        # getMediaAssets already outer-joins labels and verification fields onto
        # each segment row. A second getMediaSegments pass is unnecessary for
        # observation tables (NatureCubeR uses assets only).
        psr_ids = station_lookup["project_system_record_id"].tolist()
        # record_count is the exact number of segment rows per station, so it
        # gives the progress bar a real total and ETA.
        total_rows = None
        if "record_count" in scoped_stations.columns:
            counts = pd.to_numeric(scoped_stations["record_count"], errors="coerce").fillna(0)
            total_rows = int(counts.sum()) or None
        media_df = get_media_assets_df(hdr, datatype, psr_ids, total_rows=total_rows)
        if media_df.empty:
            return pd.DataFrame(), station_lookup
        media_df = resolve_psr_id_column(media_df)
        merged = merge_station_lookup(media_df, station_lookup)
        if "data_type" not in merged.columns:
            merged["data_type"] = str(datatype)
    return merged, station_lookup


# ---------------------------------------------------------------------------
# High-level observation retrieval
# ---------------------------------------------------------------------------


def get_camera_trap_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve camera trap species observations with station locations.

    Both image and video camera trap data are always returned together.
    Stations are fetched in parallel (one station ID per request, a few workers)
    using the same media downloader as audio observations.

    Fetches media assets (which already include AI/human labels and
    verification flags), joins station coordinates, and keeps only labelled,
    non-blank detections. Image and video are fetched as separate passes so a
    failure in one datatype does not discard the other.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    include_iucn_status:
        If ``True``, add an ``iucn_redlist_status`` column joined on ``species``.

    Returns
    -------
    pandas.DataFrame
        One row per labelled camera detection, with station location columns and
        verification fields such as ``segment_verification_status``.

    Examples
    --------
    >>> df = get_camera_trap_data(hdr)  # doctest: +SKIP
    >>> df = get_camera_trap_data(hdr, include_iucn_status=True)  # doctest: +SKIP
    >>> sorted(df["data_type"].unique())
    ['image', 'video']
    """
    stations = get_station_info(hdr, "camera")
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for datatype in ["image", "video"]:
        scoped_stations = stations
        if "data_type" in stations.columns:
            scoped_stations = stations[
                stations["data_type"].astype(str).str.strip().str.lower() == datatype
            ]
        station_lookup = build_station_lookup(scoped_stations, datatype)
        if station_lookup.empty:
            continue

        psr_ids = station_lookup["project_system_record_id"].tolist()
        total_rows = None
        if "record_count" in scoped_stations.columns:
            counts = pd.to_numeric(scoped_stations["record_count"], errors="coerce").fillna(0)
            total_rows = int(counts.sum()) or None

        try:
            # Parallel single-station requests via get_media_assets_df; no inter-
            # station sleep. getMediaAssets already outer-joins labels.
            media_df = get_media_assets_df(hdr, datatype, psr_ids, total_rows=total_rows)
            if media_df.empty:
                continue
            media_df = resolve_psr_id_column(media_df)
            merged = merge_station_lookup(media_df, station_lookup)
            if "data_type" not in merged.columns:
                merged["data_type"] = datatype
            if "label" in merged.columns:
                merged = merged[merged["label"].notna()]
            if "blank" in merged.columns:
                merged = merged[merged["blank"] != True]  # noqa: E712
            if not merged.empty:
                frames.append(merged)
        except Exception as exc:
            errors.append(f"{datatype}: {exc}")

    if errors:
        print(
            f"Warning: camera trap download had {len(errors)} datatype failure(s). "
            f"First error: {errors[0]}"
        )

    if not frames:
        if errors:
            raise RuntimeError(
                "Camera trap download failed for all datatypes. "
                f"First error: {errors[0]}"
            )
        return pd.DataFrame(columns=STATION_LOOKUP_COLUMNS)

    result = pd.concat(frames, ignore_index=True, sort=False)
    if include_iucn_status:
        result = enrich_with_iucn_status(result, build_iucn_map(hdr))
    return result



def get_audio_observation_data(
    hdr: AuthHeaders,
    include_iucn_status: bool = False,
) -> pd.DataFrame:
    """Retrieve bioacoustic species observations with station locations.

    Fetches audio media assets (which already include AI/human labels and
    verification flags), joins station coordinates, and keeps only labelled,
    non-blank detections.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    include_iucn_status:
        If ``True``, add an ``iucn_redlist_status`` column joined on ``species``.

    Returns
    -------
    pandas.DataFrame
        One row per labelled audio detection, with station location columns and
        verification fields such as ``segment_verification_status``
        (``ai_derived``, ``labeller_verified``, or ``manager_verified``).

    Examples
    --------
    >>> df = get_audio_observation_data(hdr)  # doctest: +SKIP
    >>> df = get_audio_observation_data(hdr, include_iucn_status=True)  # doctest: +SKIP
    """
    # Pull directly from the audio datatype endpoint to avoid depending on
    # backend measurement_type naming consistency (e.g. "audio" vs "bioacoustic").
    stations = fetch_station_features(hdr, "audio")
    merged, station_lookup = fetch_observations_for_datatype(hdr, "audio", stations)
    if station_lookup.empty:
        return pd.DataFrame(columns=STATION_LOOKUP_COLUMNS)
    if merged.empty:
        return station_lookup.drop(columns=["latitude", "longitude"], errors="ignore")

    # getMediaAssets outer-joins labels, so unlabelled segments arrive with null
    # label columns. Blank segments are labelled but reviewer-marked as empty.
    if "label" in merged.columns:
        merged = merged[merged["label"].notna()]
    if "blank" in merged.columns:
        merged = merged[merged["blank"] != True]  # noqa: E712
    if merged.empty:
        return pd.DataFrame(columns=list(merged.columns))

    if include_iucn_status:
        merged = enrich_with_iucn_status(merged, build_iucn_map(hdr))
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
    response = httpx.get(url, timeout=180.0)
    response.raise_for_status()
    return pd.DataFrame(extract_rows_from_payload(response.json()))


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
    merged, station_lookup = fetch_observations_for_datatype(hdr, "eDNA", stations)
    if station_lookup.empty:
        return pd.DataFrame(columns=["project_system_record_id", "device_id", "measurement_type", "latitude", "longitude"])

    if merged.empty:
        return station_lookup.reset_index(drop=True)

    if include_iucn_status and "iucn_redlist_status" not in merged.columns:
        merged = enrich_with_iucn_status(merged, build_iucn_map(hdr))

    return merged.reset_index(drop=True)


def normalise_species_frame(df: pd.DataFrame, data_type_fallback: str) -> pd.DataFrame:
    """Ensure core species observation columns are present, filling missing ones with ``pd.NA``."""
    if df.empty:
        return pd.DataFrame(columns=SPECIES_OBS_CORE_COLUMNS).copy()

    out = df.copy()
    if "label" not in out.columns:
        out["label"] = out["species"] if "species" in out.columns else pd.NA
    if "data_type" not in out.columns:
        out["data_type"] = data_type_fallback
    for col in SPECIES_OBS_CORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def resolve_measurement_types(
    measurement_types: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Normalise measurement-type input to a list of internal keys."""
    all_keys = list({"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}.keys())
    if measurement_types is None:
        return all_keys
    raw = [measurement_types] if isinstance(measurement_types, str) else list(measurement_types)
    if not raw:
        return all_keys
    result: list[str] = []
    for value in raw:
        mt = str(value).strip().lower()
        if mt in ("all", ""):
            return all_keys
        if mt == "edna":
            mt = "edna"
        if mt not in {"camera": ["image", "video"], "bioacoustic": ["audio"], "edna": ["eDNA"]}:
            raise ValueError(
                f"Invalid measurement_type: {value!r}. "
                f"Must be one of {all_keys} or None/'all'."
            )
        if mt not in result:
            result.append(mt)
    return result


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
    requested = resolve_measurement_types(measurement_types)
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
        frames.append(normalise_species_frame(frame, fallback))

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=SPECIES_OBS_CORE_COLUMNS).copy()

    result = pd.concat(non_empty, ignore_index=True, sort=False)
    if include_iucn_status:
        result = enrich_with_iucn_status(result, build_iucn_map(hdr))
    return result


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def normalise_species_name(name: str) -> str:
    """Normalise a species name for reliable dictionary lookup.

    Strips leading/trailing whitespace (including non-breaking spaces) and
    lowercases the string so that case and spacing differences between the
    IUCN table and observation data do not silently break the join.
    """
    return name.replace("\xa0", " ").strip().lower()


def build_iucn_map(hdr: AuthHeaders) -> dict[str, str | None]:
    """Fetch the full IUCN label table and return a normalised species-name → status mapping.

    Keys are lowercased and stripped (via :func:`normalise_species_name`) so
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
            iucn_map[normalise_species_name(species_name)] = status
        if label_name:
            iucn_map[normalise_species_name(label_name)] = status
    if not iucn_map:
        print(
            "Warning: IUCN map is empty. The 'species' attribute may not exist on "
            "SpeciesLight — check the schema field name (e.g. 'scientific_name', "
            "'latin_name') and update build_iucn_map accordingly."
        )
    return iucn_map


def enrich_with_iucn_status(df: pd.DataFrame, iucn_map: dict[str, str | None]) -> pd.DataFrame:
    """Add an ``iucn_redlist_status`` column to a DataFrame by joining on ``species``.

    Both sides of the join are normalised via :func:`normalise_species_name`
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
        normalised_species = df.loc[mask, "species"].dropna().map(normalise_species_name)
        df.loc[normalised_species.index, "iucn_redlist_status"] = normalised_species.map(iucn_map)

    if "label" in df.columns:
        mask = df["iucn_redlist_status"].isna()
        normalised_label = df.loc[mask, "label"].dropna().map(normalise_species_name)
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
    url = f"{hdr.root}getProjectLabels/{labeltype}/{hdr.key}"
    response = httpx.get(url, timeout=180.0)
    response.raise_for_status()
    df = pd.DataFrame(extract_rows_from_payload(response.json()))
    if include_iucn_status and not df.empty:
        df = enrich_with_iucn_status(df, build_iucn_map(hdr))
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
    response = httpx.post(url, json=[label.model_dump(mode="json") for label in labels], timeout=180.0)
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
    response = httpx.get(url, params=params, timeout=180.0)
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
        payload = [label.model_dump(mode="json", by_alias=True, exclude_none=True) for label in chunk]
        response = httpx.post(url, json=payload, timeout=180.0)
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
    response = httpx.put(url, json=[label.model_dump(mode="json", exclude_none=True, exclude_unset=True) for label in labels], timeout=180.0)
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
    response = httpx.put(url, json=segment_record_ids, timeout=180.0)
    response.raise_for_status()
    resp = response.json()
    print(resp.get("message", ""))
    return resp


def set_segment_published_status(
    hdr: AuthHeaders,
    published_status: bool,
    segment_record_ids: list[int],
) -> dict[str, str]:
    """Set publication visibility for one or more segment records.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`auth_headers`.
    published_status:
        ``True`` to publish segments, ``False`` to unpublish.
    segment_record_ids:
        A list of segment record IDs to update.

    Returns
    -------
    dict[str, str]
        A response message indicating success.

    Examples
    --------
    >>> set_segment_published_status(hdr, published_status=False, segment_record_ids=[101, 102])  # doctest: +SKIP
    """
    # Convert to list if needed (handles pandas Series, etc.)
    if hasattr(segment_record_ids, "tolist"):
        segment_record_ids = segment_record_ids.tolist()
    segment_record_ids = list(segment_record_ids)
    
    url = f"{hdr.root}segmentRecordsPublishStatus/{hdr.key}/{published_status}"
    response = httpx.put(
        url,
        json=segment_record_ids,
        timeout=180.0,
    )
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
    response = httpx.put(url, json=payload, timeout=180.0)
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
    response = httpx.post(url, json=payload, timeout=180.0)
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
    response = httpx.post(url, json=records, timeout=180.0)
    response.raise_for_status()
    response_df = pd.DataFrame(response.json())
    result = data.reset_index(drop=True)
    for col in response_df.columns:
        result[col] = response_df[col].values
    if "class_" in result.columns:
        result = result.drop(columns=["class_"])
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
    response = httpx.post(url, json=payload, timeout=180.0)
    response.raise_for_status()
    result = [eDNAUploadResponse.model_validate(item) for item in response.json()]
    print(f"Upload complete: {len(result)} records processed.")
    return result