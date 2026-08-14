"""
Phone observation builders and upload functions for the Okala dashboard.

This module provides functions for constructing structured observation records
from mobile devices (photos, videos, audio, and form data) and uploading them
to the Okala platform.
"""

from __future__ import annotations

import json
import mimetypes
import re
import unicodedata
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from naturecubepy.schema import AuthHeaders

#: Valid item types for phone observations.
PHONE_TYPES = (
    "phone-photo",
    "phone-video",
    "phone-audio",
    "choice",
    "text",
    "numeric",
    "label",
    "instruction",
)

_VALID_GEOM_TYPES = ("Point", "Polygon", "LineString")
_MEDIA_TYPES = ("phone-photo", "phone-video", "phone-audio")
_DEVICE_REQUIRED = (
    "device_id",
    "phone_model",
    "phone_operating_system",
    "carrier",
    "build_number",
    "build_id",
)
_FEATURE_REQUIRED = (
    "feature_uuid",
    "project_system_id",
    "procedure_id",
    "procedure_start_timestamp",
    "procedure_end_timestamp",
    "created_by_method",
    "geometry",
    "observations",
)

_MIME_MAP: dict[str, str] = {
    "phone-photo": "image/jpeg",
    "phone-video": "video/mp4",
    "phone-audio": "audio/mpeg",
}


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_dt(dt: datetime | None) -> str:
    """Format a datetime (or now) to ISO 8601."""
    if dt is None:
        return _now_iso()
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_device_settings(
    device_id: str,
    phone_model: str,
    phone_os: str,
    carrier: str,
    build_number: str,
    build_id: str,
    battery_level: float = 100,
    device_last_used: datetime | None = None,
) -> dict[str, Any]:
    """Construct validated device settings for API submission.

    Parameters
    ----------
    device_id:
        Unique identifier for the device.
    phone_model:
        Model name of the phone (e.g. ``"iPhone 14 Pro"``).
    phone_os:
        Operating system (e.g. ``"iOS 17.2"``).
    carrier:
        Network carrier (e.g. ``"Vodafone"``).
    build_number:
        App build number (e.g. ``"1.2.3"``).
    build_id:
        App build identifier.
    battery_level:
        Battery percentage (0–100). Defaults to 100.
    device_last_used:
        Timestamp of last device use. Defaults to the current time.

    Returns
    -------
    dict
        Device settings dict ready for API submission.

    Raises
    ------
    ValueError
        If any required field is empty or ``battery_level`` is out of range.

    Examples
    --------
    >>> device = build_device_settings(
    ...     device_id="abc123",
    ...     phone_model="iPhone 14 Pro",
    ...     phone_os="iOS 17.2",
    ...     carrier="Vodafone",
    ...     build_number="1.2.3",
    ...     build_id="build-456",
    ... )
    """
    for name, value in [
        ("device_id", device_id),
        ("phone_model", phone_model),
        ("phone_os", phone_os),
        ("carrier", carrier),
        ("build_number", build_number),
        ("build_id", build_id),
    ]:
        if not value or not str(value).strip():
            raise ValueError(f"{name} is required")

    if not (0 <= battery_level <= 100):
        raise ValueError("battery_level must be a number between 0 and 100")

    return {
        "device_id": str(device_id),
        "phone_model": str(phone_model),
        "phone_operating_system": str(phone_os),
        "carrier": str(carrier),
        "build_number": str(build_number),
        "build_id": str(build_id),
        "battery_level": float(battery_level),
        "device_created_at": _now_iso(),
        "device_last_used": _format_dt(device_last_used),
    }


def build_observation(
    item_uuid: str,
    item_type: str,
    data: Any,
    geometry: dict[str, Any],
    observation_uuid: str | None = None,
    observation_created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a single observation record (GeoJSON Feature).

    Parameters
    ----------
    item_uuid:
        UUID of the form item this observation relates to.
    item_type:
        Type of observation. Must be one of :data:`PHONE_TYPES`.
    data:
        Observation data. For media types, a list of filenames.
    geometry:
        GeoJSON geometry dict (``Point``, ``Polygon``, or ``LineString``).
    observation_uuid:
        UUID for this observation. Auto-generated if ``None``.
    observation_created_at:
        Creation timestamp. Defaults to the current UTC time.

    Returns
    -------
    dict
        A GeoJSON Feature dict representing the observation.

    Raises
    ------
    ValueError
        If ``item_type`` or ``geometry`` is invalid.

    Examples
    --------
    >>> obs = build_observation(
    ...     item_uuid="f47ac10b-58cc-4372-a567-0e02b2c3d479",
    ...     item_type="phone-photo",
    ...     data=["photo1.jpg"],
    ...     geometry={"type": "Point", "coordinates": [-1.5, 53.4]},
    ... )
    """
    if item_type not in PHONE_TYPES:
        raise ValueError(
            f"item_type must be one of: {', '.join(PHONE_TYPES)}"
        )

    if not item_uuid or not str(item_uuid).strip():
        raise ValueError("item_uuid is required")

    if not geometry or not isinstance(geometry, dict):
        raise ValueError("geometry is required and must be a dict")
    if "type" not in geometry:
        raise ValueError("geometry must be a GeoJSON object with a 'type' property")
    if geometry["type"] not in _VALID_GEOM_TYPES:
        raise ValueError(
            f"geometry type must be one of: {', '.join(_VALID_GEOM_TYPES)}"
        )

    if observation_uuid is None:
        observation_uuid = str(uuid_lib.uuid4())

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "item_uuid": str(item_uuid),
            "item_type": str(item_type),
            "observation_uuid": str(observation_uuid),
            "observation_created_at": _format_dt(observation_created_at),
            "data": list(data) if not isinstance(data, list) else data,
        },
    }


def build_feature_record(
    feature_uuid: str,
    project_system_id: int,
    procedure_id: int,
    start_time: datetime,
    end_time: datetime,
    created_by_method: str,
    geometry: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct a feature record (FieldRecord) containing observations.

    Parameters
    ----------
    feature_uuid:
        UUID for this feature record.
    project_system_id:
        ID of the project system.
    procedure_id:
        ID of the procedure being followed.
    start_time:
        When the procedure started.
    end_time:
        When the procedure ended.
    created_by_method:
        How the feature was created: ``"drawn"`` or ``"traced"``.
    geometry:
        GeoJSON geometry dict (``Point``, ``Polygon``, or ``LineString``).
    observations:
        List of observation records from :func:`build_observation`.

    Returns
    -------
    dict
        A FieldRecord dict ready for API submission.

    Raises
    ------
    ValueError
        If any required field is missing or invalid.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> feature = build_feature_record(
    ...     feature_uuid="feature-uuid-1",
    ...     project_system_id=42,
    ...     procedure_id=7,
    ...     start_time=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
    ...     end_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    ...     created_by_method="drawn",
    ...     geometry={"type": "Point", "coordinates": [-1.5, 53.4]},
    ...     observations=[],
    ... )
    """
    if not feature_uuid or not str(feature_uuid).strip():
        raise ValueError("feature_uuid is required")
    if project_system_id is None:
        raise ValueError("project_system_id is required")
    if procedure_id is None:
        raise ValueError("procedure_id is required")
    if start_time is None:
        raise ValueError("start_time is required")
    if end_time is None:
        raise ValueError("end_time is required")
    if not created_by_method:
        raise ValueError("created_by_method is required")
    if created_by_method not in ("drawn", "traced"):
        raise ValueError("created_by_method must be 'drawn' or 'traced'")
    if not geometry or not isinstance(geometry, dict):
        raise ValueError("geometry is required and must be a dict")
    if "type" not in geometry:
        raise ValueError("geometry must be a GeoJSON object with a 'type' property")
    if geometry["type"] not in _VALID_GEOM_TYPES:
        raise ValueError(
            f"geometry type must be one of: {', '.join(_VALID_GEOM_TYPES)}"
        )
    if observations is None:
        raise ValueError("observations is required")

    return {
        "feature_uuid": str(feature_uuid),
        "project_system_id": int(project_system_id),
        "procedure_id": int(procedure_id),
        "procedure_start_timestamp": _format_dt(start_time),
        "procedure_end_timestamp": _format_dt(end_time),
        "created_by_method": str(created_by_method),
        "geometry": geometry,
        "observations": observations,
    }


def _collect_pending_media(
    observations: list[dict[str, Any]],
    media_dir: str | Path,
) -> list[dict[str, Any]]:
    """Collect pending local media descriptors for signed-URL upload."""
    media_dir = Path(media_dir)
    media_uploads: list[dict[str, Any]] = []
    for obs in observations:
        item_type = obs.get("properties", {}).get("item_type", "")
        if item_type not in _MEDIA_TYPES:
            continue
        for filename in obs.get("properties", {}).get("data", []):
            filepath = media_dir / filename
            media_uploads.append(
                {
                    "filepath": filepath,
                    "data_type": item_type,
                    "content_type": _MIME_MAP.get(
                        item_type,
                        mimetypes.guess_type(str(filename))[0]
                        or "application/octet-stream",
                    ),
                }
            )
    return media_uploads


def _get_field_media_upload_urls(
    hdr: AuthHeaders,
    files: list[dict[str, str]],
    *,
    timeout: float = 180.0,
) -> list[dict[str, Any]]:
    """Request signed field-media upload URLs in API-sized batches."""
    signed: list[dict[str, Any]] = []
    for start in range(0, len(files), 50):
        response = httpx.post(
            f"{hdr.root}getFieldMediaUploadUrls/{hdr.key}",
            json={"files": files[start : start + 50]},
            timeout=timeout,
        )
        response.raise_for_status()
        signed.extend(response.json().get("files", []))
    return signed


def _upload_pending_media(
    hdr: AuthHeaders,
    media_uploads: list[dict[str, Any]],
    *,
    timeout: float = 180.0,
) -> dict[str, str]:
    """Upload local media and return each filename's cloud ``blob_path``."""
    if not media_uploads:
        return {}

    media_by_filename: dict[str, dict[str, Any]] = {}
    for media in media_uploads:
        filepath = Path(media["filepath"])
        filename = filepath.name
        existing = media_by_filename.get(filename)
        if existing is not None and Path(existing["filepath"]).resolve() != filepath.resolve():
            raise ValueError(
                f"Multiple media files are named '{filename}'. "
                "Filenames must be unique within one upload."
            )
        media_by_filename[filename] = media

    signed = _get_field_media_upload_urls(
        hdr,
        [
            {
                "filename": filename,
                "data_type": str(media["data_type"]),
                "context": "field_record",
            }
            for filename, media in media_by_filename.items()
        ],
        timeout=timeout,
    )

    blob_by_filename: dict[str, str] = {}
    for entry in signed:
        filename = str(entry["filename"])
        media = media_by_filename.get(filename)
        if media is None:
            raise RuntimeError(f"Presign returned unexpected filename: {filename}")

        filepath = Path(media["filepath"])
        content_type = str(
            entry.get("content_type")
            or media.get("content_type")
            or "application/octet-stream"
        )
        response = httpx.put(
            entry["signed_url"],
            content=filepath.read_bytes(),
            headers={"Content-Type": content_type},
            timeout=timeout,
        )
        response.raise_for_status()
        blob_by_filename[filename] = str(entry["blob_path"])

    missing = set(media_by_filename) - set(blob_by_filename)
    if missing:
        raise RuntimeError(
            "No signed upload result returned for: " + ", ".join(sorted(missing))
        )
    return blob_by_filename


def validate_observation_payload(
    feature_payload: list[dict[str, Any]],
    device_settings: dict[str, Any],
    media_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate device settings and feature payload before API submission.

    Parameters
    ----------
    feature_payload:
        List of feature records from :func:`build_feature_record`.
    device_settings:
        Device settings from :func:`build_device_settings`.
    media_dir:
        Path to media files directory. Required if observations contain media.

    Returns
    -------
    dict
        A dict with ``"valid"`` (bool) and ``"errors"`` (list of str).

    Examples
    --------
    >>> result = validate_observation_payload(features, device, media_dir="/tmp/media")
    >>> if not result["valid"]:
    ...     raise ValueError("\\n".join(result["errors"]))
    """
    errors: list[str] = []

    missing_device = [f for f in _DEVICE_REQUIRED if f not in device_settings]
    if missing_device:
        errors.append(f"Missing device settings fields: {', '.join(missing_device)}")

    if not isinstance(feature_payload, list) or len(feature_payload) == 0:
        errors.append("feature_payload must be a non-empty list of feature records")
        return {"valid": False, "errors": errors}

    for i, feature in enumerate(feature_payload):
        feature_id = feature.get("feature_uuid") or f"Feature {i + 1}"

        missing_feature = [f for f in _FEATURE_REQUIRED if f not in feature]
        if missing_feature:
            errors.append(f"[{feature_id}] Missing fields: {', '.join(missing_feature)}")

        method = feature.get("created_by_method")
        if method is not None and method not in ("drawn", "traced"):
            errors.append(f"[{feature_id}] created_by_method must be 'drawn' or 'traced'")

        geom = feature.get("geometry")
        if geom is not None:
            if not isinstance(geom, dict) or "type" not in geom:
                errors.append(f"[{feature_id}] geometry must be a valid GeoJSON object")
            elif geom["type"] not in _VALID_GEOM_TYPES:
                errors.append(
                    f"[{feature_id}] geometry type must be Point, Polygon, or LineString"
                )

        observations = feature.get("observations") or []
        for j, obs in enumerate(observations):
            obs_id = obs.get("properties", {}).get("observation_uuid") or f"Observation {j + 1}"

            item_type = obs.get("properties", {}).get("item_type")
            if item_type is None:
                errors.append(f"[{feature_id}/{obs_id}] item_type is required")
            elif item_type not in PHONE_TYPES:
                errors.append(
                    f"[{feature_id}/{obs_id}] Invalid item_type '{item_type}'. "
                    f"Must be one of: {', '.join(PHONE_TYPES)}"
                )

            obs_geom = obs.get("geometry")
            if obs_geom is not None:
                if not isinstance(obs_geom, dict) or "type" not in obs_geom:
                    errors.append(
                        f"[{feature_id}/{obs_id}] observation geometry must be a valid GeoJSON object"
                    )

            if item_type in _MEDIA_TYPES:
                if media_dir is None:
                    errors.append(
                        f"[{feature_id}/{obs_id}] media_dir is required for media type observations"
                    )
                else:
                    for filename in obs.get("properties", {}).get("data", []):
                        filepath = Path(media_dir) / filename
                        if not filepath.exists():
                            errors.append(
                                f"[{feature_id}/{obs_id}] Media file not found: {filepath}"
                            )

    return {"valid": len(errors) == 0, "errors": errors}


def upload_phone_observations(
    hdr: AuthHeaders,
    project_id: int,
    feature_payload: list[dict[str, Any]],
    device_settings: dict[str, Any],
    media_dir: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Upload phone observation records to the Okala platform.

    Media files are uploaded via signed GCS URLs first, then each feature's
    metadata is posted to ``pushPhoneObservations``. Partial failures are
    collected and returned rather than stopping the upload.

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`~naturecubepy.api.auth_headers`.
    project_id:
        The project ID to upload observations to.
    feature_payload:
        List of feature records from :func:`build_feature_record`.
    device_settings:
        Device settings from :func:`build_device_settings`.
    media_dir:
        Path to media files directory. Required for media-type observations.
    validate:
        Whether to validate the payload before uploading. Defaults to ``True``.

    Returns
    -------
    dict
        A dict with keys:

        - ``"successes"`` – mapping of feature UUID → response body
        - ``"failures"`` – mapping of feature UUID → error message
        - ``"summary"`` – human-readable summary string

    Examples
    --------
    >>> result = upload_phone_observations(
    ...     hdr=hdr,
    ...     project_id=42,
    ...     feature_payload=[feature1],
    ...     device_settings=device,
    ... )
    >>> print(result["summary"])  # doctest: +SKIP
    """
    if validate:
        validation = validate_observation_payload(feature_payload, device_settings, media_dir)
        if not validation["valid"]:
            raise ValueError("Validation failed:\n" + "\n".join(validation["errors"]))

    successes: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    n_features = len(feature_payload)
    print(f"Starting upload of {n_features} feature(s)...")

    media_uploads: list[dict[str, Any]] = []
    if media_dir is not None:
        for feature in feature_payload:
            media_uploads.extend(
                _collect_pending_media(feature.get("observations", []), media_dir)
            )
    if media_uploads:
        print(f"Uploading {len(media_uploads)} media file(s)...")
        _upload_pending_media(hdr, media_uploads)

    url = f"{hdr.root}pushPhoneObservations/{hdr.key}/{project_id}"
    for i, feature in enumerate(feature_payload, start=1):
        feature_uuid = feature.get("feature_uuid", f"feature-{i}")
        print(f"Uploading feature {i} of {n_features} ({feature_uuid})...")

        try:
            device_upload = {
                "feature_payload": [feature],
                "device_settings": device_settings,
            }
            response = httpx.post(
                url,
                data={"data": json.dumps(device_upload)},
                timeout=180.0,
            )
            response.raise_for_status()
            resp_body = response.json()
            successes[feature_uuid] = {"feature_uuid": feature_uuid, "response": resp_body}
            print(f"  ✓ Feature {feature_uuid} uploaded successfully")

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            failures[feature_uuid] = {"feature_uuid": feature_uuid, "error": error_msg}
            print(f"  ✗ Feature {feature_uuid} failed: {error_msg}")

    n_success = len(successes)
    n_failed = len(failures)
    summary = (
        f"Upload complete: {n_success} of {n_features} features uploaded successfully, "
        f"{n_failed} failed"
    )
    print(summary)

    return {"successes": successes, "failures": failures, "summary": summary}


# ---------------------------------------------------------------------------
# Schema discovery + CSV uploadObservations workflow
# ---------------------------------------------------------------------------


def _normalize_lookup_value(value: Any) -> str:
    """Lowercase, strip, and ASCII-fold a lookup key."""
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip().lower()


def _normalize_lookup_series(values: Any) -> list[str]:
    if isinstance(values, pd.Series):
        return [_normalize_lookup_value(v) for v in values.tolist()]
    return [_normalize_lookup_value(v) for v in list(values)]


def _collect_item_nodes(node: Any, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Recursively collect dict nodes that expose an ``item_uuid``."""
    if out is None:
        out = []
    if not isinstance(node, (dict, list)):
        return out
    if isinstance(node, dict):
        if node.get("item_uuid"):
            out.append(node)
        for child in node.values():
            _collect_item_nodes(child, out)
    else:
        for child in node:
            _collect_item_nodes(child, out)
    return out


def _choice_labels(choice_vals: Any) -> str:
    if choice_vals is None:
        return ""
    labels: list[str] = []
    for choice in list(choice_vals):
        if isinstance(choice, dict):
            label = (
                choice.get("label")
                or choice.get("value")
                or choice.get("name")
                or choice.get("choice_label")
                or choice
            )
            labels.append(str(label))
        else:
            labels.append(str(choice))
    return " | ".join(labels)


def _format_id_list(values: list[Any]) -> str:
    return ", ".join(str(v) for v in values if v is not None) or "none"


def _resolve_schema_indices(
    schema: dict[str, Any],
    *,
    system_index: int | None = None,
    procedure_index: int | None = None,
    system_name: str | None = None,
    procedure_name: str | None = None,
    system_id: int | None = None,
    procedure_id: int | None = None,
) -> tuple[int, int]:
    systems = schema.get("systems") or []
    if not systems:
        raise ValueError("schema does not contain any systems")

    system_ids = [s.get("project_system_id") for s in systems]
    if system_index is None:
        if system_id is not None:
            try:
                system_index = [str(sid) for sid in system_ids].index(str(system_id)) + 1
            except ValueError as exc:
                raise ValueError(
                    f"system_id {system_id} not found in schema. "
                    f"Available system_id values: {_format_id_list(system_ids)}"
                ) from exc
        elif system_name:
            names = [_normalize_lookup_value(s.get("system_name", "")) for s in systems]
            target = _normalize_lookup_value(system_name)
            try:
                system_index = names.index(target) + 1
            except ValueError as exc:
                available = ", ".join(str(s.get("system_name") or "") for s in systems)
                raise ValueError(
                    f"system_name '{system_name}' not found in schema. Available: {available}"
                ) from exc
        else:
            system_index = 1

    if system_index < 1 or system_index > len(systems):
        hint = ""
        if str(system_index) in {str(sid) for sid in system_ids if sid is not None}:
            hint = (
                f" {system_index} looks like a project_system_id, not a position; "
                f"pass system_id={system_index} instead."
            )
        raise ValueError(
            f"system_index must be between 1 and {len(systems)}, got {system_index}."
            + hint
            + " Use list_systems(schema) to see positions, ids, and names."
        )

    procedures = systems[system_index - 1].get("procedures") or []
    if not procedures:
        raise ValueError("selected system does not contain any procedures")

    procedure_ids = [p.get("procedure_id") for p in procedures]
    if procedure_index is None:
        if procedure_id is not None:
            try:
                procedure_index = [str(pid) for pid in procedure_ids].index(str(procedure_id)) + 1
            except ValueError as exc:
                raise ValueError(
                    f"procedure_id {procedure_id} not found in the selected system. "
                    f"Available procedure_id values: {_format_id_list(procedure_ids)}"
                ) from exc
        elif procedure_name:
            names = [_normalize_lookup_value(p.get("procedure_name", "")) for p in procedures]
            target = _normalize_lookup_value(procedure_name)
            try:
                procedure_index = names.index(target) + 1
            except ValueError as exc:
                available = ", ".join(str(p.get("procedure_name") or "") for p in procedures)
                raise ValueError(
                    f"procedure_name '{procedure_name}' not found in the selected system. "
                    f"Available: {available}"
                ) from exc
        else:
            procedure_index = 1

    if procedure_index < 1 or procedure_index > len(procedures):
        hint = ""
        if str(procedure_index) in {str(pid) for pid in procedure_ids if pid is not None}:
            hint = (
                f" {procedure_index} looks like a procedure_id, not a position; "
                f"pass procedure_id={procedure_index} instead."
            )
        raise ValueError(
            f"procedure_index must be between 1 and {len(procedures)}, got {procedure_index}."
            + hint
        )

    return system_index, procedure_index


def _schema_item_dictionary(
    schema: dict[str, Any],
    *,
    system_index: int = 1,
    procedure_index: int = 1,
) -> pd.DataFrame:
    systems = schema.get("systems") or []
    if system_index < 1 or system_index > len(systems):
        raise ValueError("system_index is out of bounds for schema['systems']")
    procedures = systems[system_index - 1].get("procedures") or []
    if procedure_index < 1 or procedure_index > len(procedures):
        raise ValueError("procedure_index is out of bounds for selected system procedures")

    rows: list[dict[str, str]] = []
    for node in _collect_item_nodes(procedures[procedure_index - 1]):
        item_name = (
            node.get("item_name")
            or node.get("name")
            or node.get("label")
            or node.get("title")
            or ""
        )
        item_uuid = str(node.get("item_uuid") or "")
        if not item_uuid:
            continue
        rows.append(
            {
                "item_uuid": item_uuid,
                "item_name": str(item_name),
                "data_type": str(node.get("data_type") or ""),
            }
        )

    dictionary = pd.DataFrame(
        rows, columns=["item_uuid", "item_name", "data_type"]
    )
    if dictionary.empty:
        return dictionary
    return dictionary.drop_duplicates(subset=["item_uuid"]).reset_index(drop=True)


def get_project_systems(hdr: AuthHeaders, timeout: float = 180.0) -> dict[str, Any]:
    """Fetch the project schema (systems, procedures, and item codebook).

    Parameters
    ----------
    hdr:
        Authentication context returned by :func:`~naturecubepy.api.auth_headers`.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON schema used to resolve ``project_system_id``,
        ``procedure_id``, and item UUIDs.
    """
    url = f"{hdr.root}getProjectSchema/{hdr.key}"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch project schema from {url}. "
            "Confirm GET /api/getProjectSchema/{{api_key}} is available. "
            f"Original error: {exc}"
        ) from exc
    return response.json()


def list_systems(schema: dict[str, Any]) -> pd.DataFrame:
    """Summarise systems and procedures from a project schema.

    Returns
    -------
    pandas.DataFrame
        Columns ``system_index``, ``system_name``, ``system_id``,
        ``procedure_index``, ``procedure_name``, ``procedure_id``, ``form``.
    """
    systems = schema.get("systems") or []
    if not systems:
        print("No systems found in schema.")
        return pd.DataFrame(
            columns=[
                "system_index",
                "system_name",
                "system_id",
                "procedure_index",
                "procedure_name",
                "procedure_id",
                "form",
            ]
        )

    rows: list[dict[str, Any]] = []
    for si, system in enumerate(systems, start=1):
        sys_name = str(system.get("system_name") or "")
        sys_id = system.get("project_system_id")
        procedures = system.get("procedures") or []
        if not procedures:
            rows.append(
                {
                    "system_index": si,
                    "system_name": sys_name,
                    "system_id": int(sys_id) if sys_id is not None else pd.NA,
                    "procedure_index": pd.NA,
                    "procedure_name": pd.NA,
                    "procedure_id": pd.NA,
                    "form": pd.NA,
                }
            )
            continue
        for pi, procedure in enumerate(procedures, start=1):
            rows.append(
                {
                    "system_index": si,
                    "system_name": sys_name,
                    "system_id": int(sys_id) if sys_id is not None else pd.NA,
                    "procedure_index": pi,
                    "procedure_name": str(procedure.get("procedure_name") or ""),
                    "procedure_id": (
                        int(procedure["procedure_id"])
                        if procedure.get("procedure_id") is not None
                        else pd.NA
                    ),
                    "form": procedure.get("form"),
                }
            )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


def get_procedure(
    schema: dict[str, Any],
    *,
    system_name: str | None = None,
    system_index: int | None = None,
    system_id: int | None = None,
    procedure_name: str | None = None,
    procedure_index: int | None = None,
    procedure_id: int | None = None,
) -> dict[str, Any]:
    """Return procedure metadata and an items table for upload mapping.

    Select the system and procedure by name, by API id (``system_id`` /
    ``procedure_id``), or by 1-based position in the schema (``system_index`` /
    ``procedure_index``). Run :func:`list_systems` to see all three.

    Returns
    -------
    dict
        Keys ``system_id``, ``procedure_id``, ``system_name``,
        ``procedure_name``, ``form``, and ``items`` (DataFrame).
    """
    si, pi = _resolve_schema_indices(
        schema,
        system_index=system_index,
        procedure_index=procedure_index,
        system_name=system_name,
        procedure_name=procedure_name,
        system_id=system_id,
        procedure_id=procedure_id,
    )
    system = schema["systems"][si - 1]
    procedure = system["procedures"][pi - 1]
    item_nodes = _collect_item_nodes(procedure)

    rows: list[dict[str, Any]] = []
    for node in item_nodes:
        item_name = (
            node.get("item_name")
            or node.get("name")
            or node.get("label")
            or node.get("title")
            or ""
        )
        choice_vals = node.get("choices") or node.get("options") or node.get("items")
        rows.append(
            {
                "item_id": int(node["item_id"]) if node.get("item_id") is not None else pd.NA,
                "item_uuid": str(node.get("item_uuid") or ""),
                "item_name": str(item_name),
                "item_description": (
                    str(node["item_description"])
                    if node.get("item_description") is not None
                    else pd.NA
                ),
                "data_type": str(node.get("data_type") or ""),
                "nullable": (
                    bool(node["nullable"]) if node.get("nullable") is not None else pd.NA
                ),
                "choices": _choice_labels(choice_vals),
            }
        )

    items = pd.DataFrame(
        rows,
        columns=[
            "item_id",
            "item_uuid",
            "item_name",
            "item_description",
            "data_type",
            "nullable",
            "choices",
        ],
    )
    if items.empty:
        print("No items found in selected procedure.")
        return {
            "system_id": system.get("project_system_id"),
            "procedure_id": procedure.get("procedure_id"),
            "system_name": str(system.get("system_name") or f"System {si}"),
            "procedure_name": str(procedure.get("procedure_name") or f"Procedure {pi}"),
            "form": procedure.get("form"),
            "items": items,
        }

    out = {
        "system_id": (
            int(system["project_system_id"])
            if system.get("project_system_id") is not None
            else None
        ),
        "procedure_id": (
            int(procedure["procedure_id"]) if procedure.get("procedure_id") is not None else None
        ),
        "system_name": str(system.get("system_name") or f"System {si}"),
        "procedure_name": str(procedure.get("procedure_name") or f"Procedure {pi}"),
        "form": procedure.get("form"),
        "items": items.reset_index(drop=True),
    }
    print(
        f"System: {out['system_name']} (id: {out['system_id']})\n"
        f"Procedure: {out['procedure_name']} (id: {out['procedure_id']}, form: {out['form']})\n"
        f"Items ({len(items)}):"
    )
    print(items.to_string(index=False))
    return out


#: Delimiters used in CSVs when a choice field holds multiple selected values.
_MULTI_VALUE_SPLIT = re.compile(r"\s*[;|]\s*")


def _split_multi_value(value: str) -> list[str]:
    """Split a multi-select cell such as ``calling;land`` into its parts."""
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return []
    return [part for part in _MULTI_VALUE_SPLIT.split(text) if part]


def _canonical_choice(value: str, valid_choices: list[str]) -> str | None:
    """Return the procedure's spelling of a choice, matching case-insensitively."""
    lookup = {_normalize_lookup_value(c): c for c in valid_choices}
    return lookup.get(_normalize_lookup_value(value))


def _coerce_choice_value(
    raw: Any,
    valid_choices: list[str] | None = None,
) -> str | list[str] | None:
    """Turn a choice cell into a scalar or list for ``uploadObservations``.

    Cells may hold several selections separated by ``;`` or ``|``
    (e.g. ``calling;land``). Multiple selections become a list so they upload
    as one observation; a single selection stays a string. When
    ``valid_choices`` is provided, each part is rewritten to the procedure's
    spelling (case-insensitive match).
    """
    parts = _split_multi_value(raw)
    if not parts:
        return None
    if valid_choices:
        parts = [_canonical_choice(part, valid_choices) or part for part in parts]
    return parts if len(parts) > 1 else parts[0]


def _check_value_against_type(
    *,
    item_name_norm: str,
    effective_val: str,
    row_num: int,
    dt_lookup: dict[str, str],
    ch_lookup: dict[str, str],
) -> dict[str, Any] | None:
    data_type = dt_lookup.get(item_name_norm, "")
    if not data_type or not effective_val or effective_val == "NA":
        return None
    dt_lower = data_type.lower()
    problem = ""
    if re.search(r"num|int|float|double|decimal|real", dt_lower):
        try:
            float(effective_val)
        except ValueError:
            problem = f"expected numeric for data_type '{data_type}', got: '{effective_val}'"
    elif "bool" in dt_lower:
        if effective_val.lower() not in {"true", "false", "yes", "no", "1", "0"}:
            problem = f"expected boolean for data_type '{data_type}', got: '{effective_val}'"
    elif "choice" in dt_lower:
        choices_raw = ch_lookup.get(item_name_norm, "")
        parts = _split_multi_value(effective_val)
        if choices_raw and parts:
            valid_choices = [c.strip() for c in choices_raw.split("|") if c.strip()]
            bad = [p for p in parts if _canonical_choice(p, valid_choices) is None]
            if bad:
                quoted = ", ".join(f"'{b}'" for b in bad)
                problem = f"{quoted} not in choices: {' | '.join(valid_choices)}"
    if not problem:
        return None
    return {
        "row": row_num,
        "data_type": data_type,
        "value": effective_val,
        "problem": problem,
    }


def _feature_groups_long(
    csv: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    item_name_col: str,
) -> pd.DataFrame:
    if lon_col not in csv.columns or lat_col not in csv.columns:
        return pd.DataFrame(columns=["longitude", "latitude", "n_rows", "item_names"])
    work = csv.copy()
    work["_lon"] = pd.to_numeric(work[lon_col], errors="coerce").round(7)
    work["_lat"] = pd.to_numeric(work[lat_col], errors="coerce").round(7)
    groups = []
    for (_, _), g in work.groupby(["_lon", "_lat"], dropna=False):
        names = (
            " | ".join(g[item_name_col].astype(str).str.strip().tolist())
            if item_name_col in g.columns
            else ""
        )
        groups.append(
            {
                "longitude": float(g["_lon"].iloc[0]) if pd.notna(g["_lon"].iloc[0]) else pd.NA,
                "latitude": float(g["_lat"].iloc[0]) if pd.notna(g["_lat"].iloc[0]) else pd.NA,
                "n_rows": len(g),
                "item_names": names,
            }
        )
    return pd.DataFrame(groups)


#: Alternative header spellings accepted for the metadata columns.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "longitude": ("longitude", "lon", "lng", "long"),
    "latitude": ("latitude", "lat"),
    "recorded_at": ("recorded_at", "timestamp", "datetime", "date_time"),
    "item_name": ("item_name", "field_name"),
    "data": ("data", "value"),
    "numbers": ("numbers", "numeric_value"),
    "observation_id": ("observation_id", "obs_id", "survey_id"),
    "item_uuid": ("item_uuid",),
}


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM and surrounding whitespace from column headers."""
    df = df.copy()
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def _resolve_column(columns: Any, requested: str) -> str:
    """Find the actual column matching ``requested``, tolerating case and aliases.

    Returns the real column name when found; otherwise returns ``requested``
    unchanged so downstream "missing column" reporting names what was asked for.
    """
    lookup = {_normalize_lookup_value(c): str(c) for c in columns}
    exact = lookup.get(_normalize_lookup_value(requested))
    if exact is not None:
        return exact
    for alias in _COLUMN_ALIASES.get(requested, ()):  # only for canonical names
        hit = lookup.get(alias)
        if hit is not None:
            return hit
    return requested


def validate_csv_against_procedure(
    procedure: dict[str, Any],
    csv_path: str | Path,
    *,
    format: str = "auto",
    item_name_col: str = "item_name",
    value_col: str = "data",
    numeric_value_col: str = "numbers",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    recorded_at_col: str = "recorded_at",
) -> dict[str, Any]:
    """Validate an observation CSV against a :func:`get_procedure` result.

    Supports long format (``item_name`` + ``data``) and wide format (item names
    as columns). Format is auto-detected unless ``format`` is ``"long"`` or
    ``"wide"``.
    """
    if not isinstance(procedure, dict) or procedure.get("items") is None:
        raise ValueError("procedure must be a non-empty dict returned by get_procedure()")
    items = procedure["items"]
    if not isinstance(items, pd.DataFrame) or items.empty:
        raise ValueError("procedure must be a non-empty dict returned by get_procedure()")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path does not exist: {csv_path}")
    if format not in {"auto", "long", "wide"}:
        raise ValueError("format must be one of: auto, long, wide")

    csv = _clean_headers(pd.read_csv(csv_path, dtype=str, keep_default_na=False))

    # Tolerate different header spellings (e.g. "timestamp" for recorded_at,
    # "lat"/"lon", any casing) before detecting the layout.
    lon_col = _resolve_column(csv.columns, lon_col)
    lat_col = _resolve_column(csv.columns, lat_col)
    recorded_at_col = _resolve_column(csv.columns, recorded_at_col)
    item_name_col = _resolve_column(csv.columns, item_name_col)
    value_col = _resolve_column(csv.columns, value_col)
    numeric_value_col = _resolve_column(csv.columns, numeric_value_col)

    if format == "auto":
        detected = "long" if item_name_col in csv.columns else "wide"
    else:
        detected = format
    print(f"Detected CSV format: {detected}")

    proc_norm = _normalize_lookup_series(items["item_name"])
    dt_lookup = {
        n: str(t)
        for n, t in zip(proc_norm, items["data_type"].tolist(), strict=False)
    }
    ch_lookup = {
        n: str(c)
        for n, c in zip(proc_norm, items["choices"].fillna("").tolist(), strict=False)
    }

    issues: list[str] = []
    type_rows: list[dict[str, Any]] = []

    if detected == "long":
        required_cols = [lon_col, lat_col, recorded_at_col, item_name_col, value_col]
        missing_std = [c for c in required_cols if c not in csv.columns]
        if missing_std:
            issues.append(f"Missing columns: {', '.join(missing_std)}")

        csv_item_names = (
            csv[item_name_col].astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
            if item_name_col in csv.columns
            else []
        )
        csv_norm = _normalize_lookup_series(csv_item_names)
        matched_mask = [n in csv_norm for n in proc_norm]
        matched_items = items.loc[matched_mask].copy()
        missing_items = items.loc[[not m for m in matched_mask]].copy()
        unrecognised_names = [n for n, nn in zip(csv_item_names, csv_norm, strict=False) if nn not in proc_norm]

        if item_name_col in csv.columns and value_col in csv.columns:
            for r, row in enumerate(csv.itertuples(index=False), start=1):
                row_dict = row._asdict()
                nm = _normalize_lookup_value(row_dict.get(item_name_col, ""))
                if not nm:
                    continue
                v_txt = str(row_dict.get(value_col, "")).strip()
                v_num = row_dict.get(numeric_value_col, "") if numeric_value_col in csv.columns else ""
                effective = v_txt if v_txt else str(v_num).strip()
                hit = _check_value_against_type(
                    item_name_norm=nm,
                    effective_val=effective,
                    row_num=r,
                    dt_lookup=dt_lookup,
                    ch_lookup=ch_lookup,
                )
                if hit is not None:
                    type_rows.append({"item_name": str(row_dict.get(item_name_col, "")), **hit})
        feature_groups = _feature_groups_long(csv, lon_col, lat_col, item_name_col)
    else:
        meta_cols = [lon_col, lat_col, recorded_at_col]
        missing_meta = [c for c in meta_cols if c not in csv.columns]
        if missing_meta:
            issues.append(f"Missing metadata columns: {', '.join(missing_meta)}")
        item_cols = [c for c in csv.columns if c not in meta_cols]
        col_norm = _normalize_lookup_series(item_cols)
        matched_mask = [n in col_norm for n in proc_norm]
        matched_items = items.loc[matched_mask].copy()
        missing_items = items.loc[[not m for m in matched_mask]].copy()
        unrecognised_names = [c for c, nn in zip(item_cols, col_norm, strict=False) if nn not in proc_norm]

        for col, nm in zip(item_cols, col_norm, strict=False):
            if nm not in proc_norm:
                continue
            for r, value in enumerate(csv[col].tolist(), start=1):
                hit = _check_value_against_type(
                    item_name_norm=nm,
                    effective_val=str(value).strip(),
                    row_num=r,
                    dt_lookup=dt_lookup,
                    ch_lookup=ch_lookup,
                )
                if hit is not None:
                    type_rows.append({"item_name": col, **hit})
        # Each row is already one feature. Build against the CSV index so a
        # missing coordinate column still yields a full-length column.
        feature_groups = pd.DataFrame(index=csv.index)
        feature_groups["longitude"] = (
            pd.to_numeric(csv[lon_col], errors="coerce") if lon_col in csv.columns else pd.NA
        )
        feature_groups["latitude"] = (
            pd.to_numeric(csv[lat_col], errors="coerce") if lat_col in csv.columns else pd.NA
        )
        feature_groups["n_items"] = len(item_cols)

    required_missing = missing_items[
        missing_items["nullable"].isna() | (missing_items["nullable"] == False)  # noqa: E712
    ]
    optional_missing = missing_items[missing_items["nullable"] == True]  # noqa: E712
    if not required_missing.empty:
        label = "rows" if detected == "long" else "column"
        issues.append(
            f"{len(required_missing)} required item(s) have no {label} in the CSV: "
            + ", ".join(required_missing["item_name"].astype(str))
        )
    if not optional_missing.empty:
        print(
            f"Note: {len(optional_missing)} nullable item(s) absent from CSV (allowed): "
            + ", ".join(optional_missing["item_name"].astype(str))
        )
    warnings: list[str] = []
    if unrecognised_names:
        kind = "item name(s)" if detected == "long" else "column(s)"
        # Extra columns are dropped on upload (they map to no item), so this is
        # a warning rather than a blocking issue that would fail validation.
        warnings.append(
            f"{len(unrecognised_names)} CSV {kind} not in procedure (ignored on upload): "
            + ", ".join(unrecognised_names)
        )

    type_issues = pd.DataFrame(
        type_rows,
        columns=["row", "item_name", "data_type", "value", "problem"],
    )
    if not type_issues.empty:
        issues.append(f"{len(type_issues)} data type issue(s) found (see type_issues)")

    result = {
        "format": detected,
        "system_id": procedure.get("system_id"),
        "procedure_id": procedure.get("procedure_id"),
        "matched_items": matched_items.reset_index(drop=True),
        "unrecognised_names": unrecognised_names,
        "missing_items": missing_items.reset_index(drop=True),
        "type_issues": type_issues,
        "feature_groups": feature_groups.reset_index(drop=True),
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }

    print(f"\n--- CSV Validation Report ({detected} format) ---")
    print(f"System: {procedure.get('system_name')} (id: {procedure.get('system_id')})")
    print(f"Procedure: {procedure.get('procedure_name')} (id: {procedure.get('procedure_id')})")
    print(f"Matched items ({len(matched_items)}/{len(items)}):")
    if not matched_items.empty:
        print(matched_items[["item_name", "data_type"]].to_string(index=False))
    if not required_missing.empty:
        print(f"\nMissing required items ({len(required_missing)}):")
        print(required_missing[["item_name", "data_type", "nullable"]].to_string(index=False))
    if not optional_missing.empty:
        print(f"\nMissing nullable/optional items ({len(optional_missing)}) — allowed:")
        print(optional_missing[["item_name", "data_type", "nullable"]].to_string(index=False))
    if not type_issues.empty:
        print(f"\nData type issues ({len(type_issues)}, showing first 15):")
        print(type_issues.head(15).to_string(index=False))
    print(f"\nFeature groups: {len(feature_groups)} parent feature(s)")
    print(f"\nValid: {result['valid']}")
    if warnings:
        print("Warnings (non-blocking):\n  " + "\n  ".join(warnings))
    if issues:
        print("Issues:\n  " + "\n  ".join(issues))
    return result


def _choice_values_by_uuid(items: pd.DataFrame) -> dict[str, list[str]]:
    """Map choice-item UUIDs to their procedure choice labels."""
    if items is None or items.empty:
        return {}
    if "data_type" not in items.columns or "item_uuid" not in items.columns:
        return {}
    has_choices = "choices" in items.columns
    out: dict[str, list[str]] = {}
    for uid, dtype, choices_raw in zip(
        items["item_uuid"],
        items["data_type"],
        items["choices"].fillna("") if has_choices else [""] * len(items),
        strict=False,
    ):
        if "choice" not in str(dtype).lower():
            continue
        uid_str = str(uid).strip()
        if not uid_str:
            continue
        out[uid_str] = [c.strip() for c in str(choices_raw).split("|") if c.strip()]
    return out


def build_observation_record(
    procedure: dict[str, Any],
    values: dict[str, Any],
    recorded_at: str | datetime,
    lon: float,
    lat: float,
    survey_uuid: str | None = None,
) -> dict[str, Any]:
    """Build one ``uploadObservations`` record from a procedure + UUID-keyed values."""
    if not isinstance(procedure, dict) or procedure.get("system_id") is None or procedure.get("procedure_id") is None:
        raise ValueError("procedure must be a dict returned by get_procedure()")
    if not values:
        raise ValueError("values must be a non-empty dict keyed by item UUID")
    if any(not k for k in values):
        raise ValueError("values must be named with item UUID keys")
    if recorded_at is None:
        raise ValueError("recorded_at is required")
    if isinstance(recorded_at, datetime):
        recorded_at = recorded_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        recorded_at = str(recorded_at)
    if not isinstance(lon, (int, float)) or not (-180 <= float(lon) <= 180):
        raise ValueError("lon must be a single numeric value between -180 and 180")
    if not isinstance(lat, (int, float)) or not (-90 <= float(lat) <= 90):
        raise ValueError("lat must be a single numeric value between -90 and 90")
    return {
        "survey_uuid": str(survey_uuid or uuid_lib.uuid4()),
        "project_system_id": int(procedure["system_id"]),
        "procedure_id": int(procedure["procedure_id"]),
        "recorded_at": recorded_at,
        "lon": float(lon),
        "lat": float(lat),
        "values": dict(values),
    }


#: Maximum observations accepted by ``POST /uploadObservations`` per request.
UPLOAD_OBSERVATIONS_MAX_BATCH = 500


def _as_result_records(results: Any) -> list[dict[str, Any]]:
    """Normalise an upload API payload into a list of result dicts."""
    if results is None:
        return []
    if isinstance(results, dict):
        return [results]
    if not isinstance(results, list):
        return [{"status": "error", "message": f"Unexpected upload response: {results!r}"}]
    return [r if isinstance(r, dict) else {"status": "error", "message": str(r)} for r in results]


def summarise_upload_results(
    results: Any,
    *,
    print_summary: bool = True,
    max_error_examples: int = 10,
) -> dict[str, Any]:
    """Summarise per-record ``uploadObservations`` results.

    The API returns HTTP 200 with a mix of ``success`` / ``error`` rows, so
    callers should inspect this summary rather than treating the request itself
    as all-or-nothing.
    """
    records = _as_result_records(results)
    succeeded = [r for r in records if str(r.get("status", "")).lower() == "success"]
    failed = [r for r in records if str(r.get("status", "")).lower() != "success"]

    error_counts: dict[str, int] = {}
    for row in failed:
        message = str(row.get("message") or "unknown error").strip()
        error_counts[message] = error_counts.get(message, 0) + 1

    summary = {
        "total": len(records),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "all_succeeded": len(failed) == 0 and len(records) > 0,
        "any_succeeded": len(succeeded) > 0,
        "error_counts": error_counts,
        "errors": failed,
        "successes": succeeded,
    }

    if print_summary:
        print(
            f"Upload complete: {summary['succeeded']} succeeded, "
            f"{summary['failed']} failed ({summary['total']} total)."
        )
        if failed:
            print("Not uploaded (rolled back per observation):")
            for i, (message, count) in enumerate(error_counts.items()):
                if i >= max_error_examples:
                    remaining = len(error_counts) - max_error_examples
                    print(f"  … and {remaining} more distinct error message(s)")
                    break
                label = "observation" if count == 1 else "observations"
                print(f"  - {message} ({count} {label})")
        elif records:
            print("All observations uploaded successfully.")

    return summary


def upload_observations(
    hdr: AuthHeaders,
    observations: list[dict[str, Any]],
    *,
    batch_size: int = UPLOAD_OBSERVATIONS_MAX_BATCH,
    dry_run_payload: bool = False,
    timeout: float = 180.0,
) -> dict[str, Any] | list[Any]:
    """POST observation records to ``uploadObservations``.

    The API accepts at most 500 observations per request. Larger lists are
    split into batches of ``batch_size`` (default 500) and posted sequentially;
    the combined per-record results are returned as one list. A short success /
    failure summary is printed after the final batch.
    """
    if not observations:
        raise ValueError("observations must be a non-empty list")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > UPLOAD_OBSERVATIONS_MAX_BATCH:
        raise ValueError(
            f"batch_size cannot exceed the API limit of {UPLOAD_OBSERVATIONS_MAX_BATCH}"
        )
    if dry_run_payload:
        return {"observations": observations}

    url = f"{hdr.root}uploadObservations/{hdr.key}"
    total = len(observations)
    n_batches = (total + batch_size - 1) // batch_size
    combined: list[Any] = []

    for batch_num, start in enumerate(range(0, total, batch_size), start=1):
        batch = observations[start : start + batch_size]
        print(
            f"Uploading batch {batch_num}/{n_batches} "
            f"({len(batch)} observations, {start + len(batch)}/{total} total)..."
        )
        response = httpx.post(url, json={"observations": batch}, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(
                f"HTTP {response.status_code} from uploadObservations "
                f"(batch {batch_num}/{n_batches}, observations "
                f"{start + 1}-{start + len(batch)}).\n"
                f"Response body:\n{response.text}"
            )
        payload = response.json()
        if isinstance(payload, list):
            combined.extend(payload)
        else:
            combined.append(payload)

    summarise_upload_results(combined)
    return combined

#: Procedure data types whose values should upload as numbers rather than text.
_NUMERIC_DATA_TYPE = re.compile(r"num|int|float|double|decimal|real")


def _numeric_item_names(items: pd.DataFrame) -> set[str]:
    """Normalised names of procedure items declared with a numeric data type."""
    if "data_type" not in items.columns:
        return set()
    return {
        _normalize_lookup_value(name)
        for name, dtype in zip(items["item_name"], items["data_type"], strict=False)
        if _NUMERIC_DATA_TYPE.search(str(dtype).lower())
    }


def _wide_to_long(
    data: pd.DataFrame,
    item_names: list[str],
    *,
    numeric_names: set[str],
    lon_col: str,
    lat_col: str,
    recorded_at_col: str,
    item_name_col: str,
    value_col: str,
    numeric_value_col: str,
    observation_id_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Reshape a wide table (one row per observation) into item-per-row form.

    Only columns whose headers match a procedure item name are carried over;
    the remaining column names are returned so callers can report them.
    """
    name_norm = {_normalize_lookup_value(n) for n in item_names}
    meta_cols = {lon_col, lat_col, recorded_at_col, observation_id_col}
    candidates = [c for c in data.columns if c not in meta_cols]
    matched = [c for c in candidates if _normalize_lookup_value(c) in name_norm]
    ignored = [c for c in candidates if c not in matched]
    if not matched:
        raise ValueError(
            "No CSV column names match the procedure item names, so no values "
            "could be uploaded. Procedure items: " + ", ".join(map(str, item_names))
        )

    work = data.copy()
    # Each wide row is one observation, so give every row its own grouping key
    # rather than letting rows merge on identical coordinates and timestamps.
    fallback_ids = [f"row-{i}" for i in range(1, len(work) + 1)]
    if observation_id_col in work.columns:
        existing = work[observation_id_col].fillna("").astype(str).str.strip()
        work[observation_id_col] = existing.where(existing != "", fallback_ids)
    else:
        work[observation_id_col] = fallback_ids

    id_vars = [
        c
        for c in (lon_col, lat_col, recorded_at_col, observation_id_col)
        if c in work.columns
    ]
    long = work.melt(
        id_vars=id_vars,
        value_vars=matched,
        var_name=item_name_col,
        value_name=value_col,
    )

    # Numeric items must travel in the numeric column so they upload as numbers.
    is_numeric_item = long[item_name_col].map(
        lambda n: _normalize_lookup_value(n) in numeric_names
    )
    long[numeric_value_col] = pd.NA
    long.loc[is_numeric_item, numeric_value_col] = long.loc[is_numeric_item, value_col]
    long.loc[is_numeric_item, value_col] = ""
    return long, ignored


def build_upload_observations_from_table(
    data: pd.DataFrame,
    *,
    procedure: dict[str, Any] | None = None,
    schema: dict[str, Any] | None = None,
    system_index: int | None = None,
    procedure_index: int | None = None,
    system_name: str | None = None,
    procedure_name: str | None = None,
    system_id: int | None = None,
    procedure_id: int | None = None,
    format: str = "auto",
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    recorded_at_col: str = "recorded_at",
    item_uuid_col: str = "item_uuid",
    item_name_col: str = "item_name",
    value_col: str = "data",
    numeric_value_col: str = "numbers",
    observation_id_col: str = "observation_id",
    recorded_at_format: str = "%d/%m/%Y %H:%M",
    media_dir: str | Path | None = None,
    column_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert an observation table into ``uploadObservations`` payloads.

    Accepts long format (one row per item, using ``item_name_col``/``value_col``)
    and wide format (one row per observation, item names as column headers).
    The layout is auto-detected unless ``format`` is ``"long"`` or ``"wide"``.

    The returned dict includes ``resolved_rows`` (number of source records that
    became observations) and ``resolved_values`` (number of individual item
    values mapped); in wide format the latter is ``resolved_rows`` times the
    number of matched item columns.
    """
    if data is None or data.empty:
        raise ValueError("data must be a non-empty DataFrame")
    if format not in {"auto", "long", "wide"}:
        raise ValueError("format must be one of: auto, long, wide")

    data = _clean_headers(data)
    if column_map:
        data = data.rename(
            columns={
                source: target
                for source, target in column_map.items()
                if source in data.columns
            }
        )
    lon_col = _resolve_column(data.columns, lon_col)
    lat_col = _resolve_column(data.columns, lat_col)
    recorded_at_col = _resolve_column(data.columns, recorded_at_col)
    item_uuid_col = _resolve_column(data.columns, item_uuid_col)
    item_name_col = _resolve_column(data.columns, item_name_col)
    value_col = _resolve_column(data.columns, value_col)
    numeric_value_col = _resolve_column(data.columns, numeric_value_col)
    observation_id_col = _resolve_column(data.columns, observation_id_col)

    required_cols = [lon_col, lat_col, recorded_at_col]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {', '.join(missing_cols)}")

    use_procedure = isinstance(procedure, dict) and procedure.get("system_id") is not None
    if use_procedure:
        assert procedure is not None
        sys_id = int(procedure["system_id"])
        proc_id = int(procedure["procedure_id"])
        dictionary = procedure["items"].copy()
        if "data_type" not in dictionary.columns:
            dictionary["data_type"] = ""
        dictionary = dictionary[["item_uuid", "item_name", "data_type"]]
    else:
        if schema is None:
            raise ValueError("Either procedure or schema must be provided")
        si, pi = _resolve_schema_indices(
            schema,
            system_index=system_index,
            procedure_index=procedure_index,
            system_name=system_name,
            procedure_name=procedure_name,
            system_id=system_id,
            procedure_id=procedure_id,
        )
        dictionary = _schema_item_dictionary(schema, system_index=si, procedure_index=pi)
        system = schema["systems"][si - 1]
        proc = system["procedures"][pi - 1]
        sys_id = int(system["project_system_id"])
        proc_id = int(proc["procedure_id"])

    if format == "auto":
        detected = (
            "long"
            if item_name_col in data.columns or item_uuid_col in data.columns
            else "wide"
        )
    else:
        detected = format

    ignored_columns: list[str] = []
    if detected == "wide":
        numeric_names = (
            _numeric_item_names(procedure["items"])
            if use_procedure and isinstance(procedure, dict)
            else set()
        )
        data, ignored_columns = _wide_to_long(
            data,
            dictionary["item_name"].astype(str).tolist(),
            numeric_names=numeric_names,
            lon_col=lon_col,
            lat_col=lat_col,
            recorded_at_col=recorded_at_col,
            item_name_col=item_name_col,
            value_col=value_col,
            numeric_value_col=numeric_value_col,
            observation_id_col=observation_id_col,
        )

    work = data.copy()
    if item_uuid_col not in work.columns:
        work[item_uuid_col] = ""
    if item_name_col not in work.columns:
        work[item_name_col] = ""
    if value_col not in work.columns:
        work[value_col] = ""
    if numeric_value_col not in work.columns:
        work[numeric_value_col] = pd.NA
    if observation_id_col not in work.columns:
        work[observation_id_col] = ""

    work[item_uuid_col] = work[item_uuid_col].fillna("").astype(str)
    work[item_name_col] = work[item_name_col].fillna("").astype(str)
    work[value_col] = work[value_col].fillna("").astype(str)
    work[observation_id_col] = work[observation_id_col].fillna("").astype(str)

    name_lookup = {
        _normalize_lookup_value(name): str(uid)
        for uid, name in zip(dictionary["item_uuid"], dictionary["item_name"], strict=False)
        if str(uid)
    }
    data_type_lookup = {
        str(uid): str(data_type).lower()
        for uid, data_type in zip(
            dictionary["item_uuid"], dictionary["data_type"], strict=False
        )
        if str(uid)
    }
    choice_by_uuid = (
        _choice_values_by_uuid(procedure["items"])
        if use_procedure and isinstance(procedure, dict)
        else {}
    )
    resolved_from_name = [
        name_lookup.get(_normalize_lookup_value(name), "") for name in work[item_name_col]
    ]
    explicit_uuid = [u.strip() for u in work[item_uuid_col].tolist()]
    work["_resolved_item_uuid"] = [
        eu if eu else rn for eu, rn in zip(explicit_uuid, resolved_from_name, strict=False)
    ]
    work["_resolved_data_type"] = [
        data_type_lookup.get(item_uuid, "")
        for item_uuid in work["_resolved_item_uuid"]
    ]

    has_value = work[value_col].astype(str).str.strip() != ""
    num_vals = pd.to_numeric(work[numeric_value_col], errors="coerce")
    has_numeric = num_vals.notna()
    unresolved_mask = work["_resolved_item_uuid"].astype(str).str.strip() == ""
    empty_value_mask = ~(has_value | has_numeric)
    valid = work[~(unresolved_mask | empty_value_mask)].copy()
    if valid.empty:
        unresolved_names = sorted(
            {n for n in work.loc[unresolved_mask, item_name_col].astype(str) if n.strip()}
        )
        detail = (
            f"{int(unresolved_mask.sum())} row(s) had an item name that is not in the "
            f"procedure ({', '.join(unresolved_names[:5])}); "
            if unresolved_names
            else ""
        )
        raise ValueError(
            f"No rows could be converted into observations ({detected} format detected). "
            + detail
            + f"{int(empty_value_mask.sum())} row(s) had no value. "
            "Procedure items: "
            + ", ".join(dictionary["item_name"].astype(str).tolist())
        )

    parsed = pd.to_datetime(valid[recorded_at_col], format=recorded_at_format, errors="coerce", utc=True)
    iso_timestamp = [
        ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(ts) else str(raw)
        for ts, raw in zip(parsed, valid[recorded_at_col], strict=False)
    ]
    lon_vals = pd.to_numeric(valid[lon_col], errors="coerce")
    lat_vals = pd.to_numeric(valid[lat_col], errors="coerce")
    obs_ids = valid[observation_id_col].astype(str).str.strip()
    grouping_key = [
        f"obs-id::{oid}" if oid else f"coord-time::{round(float(lon), 7)}::{round(float(lat), 7)}::{ts}"
        for oid, lon, lat, ts in zip(obs_ids, lon_vals, lat_vals, iso_timestamp, strict=False)
    ]
    valid = valid.assign(_group=grouping_key, _iso=iso_timestamp)

    observations: list[dict[str, Any]] = []
    media_uploads: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for _, chunk in valid.groupby("_group", sort=False):
        chunk_values: dict[str, Any] = {}
        chunk_media: list[dict[str, Any]] = []
        rejection_reasons: list[str] = []
        for _, row in chunk.iterrows():
            item_uuid = str(row["_resolved_item_uuid"])
            data_type = str(row["_resolved_data_type"]).lower()
            value_text = str(row.get(value_col, "")).strip()
            value_num = row.get(numeric_value_col)
            if data_type in _MEDIA_TYPES:
                filepath = Path(value_text)
                if media_dir is not None and not filepath.is_absolute():
                    filepath = Path(media_dir) / filepath
                if not filepath.is_file():
                    rejection_reasons.append(
                        f"{row.get(item_name_col, item_uuid)}: "
                        f"media file not found: {filepath}"
                    )
                else:
                    chunk_media.append(
                        {
                            "item_uuid": item_uuid,
                            "filepath": filepath,
                            "data_type": data_type,
                            "content_type": _MIME_MAP.get(
                                data_type,
                                mimetypes.guess_type(filepath.name)[0]
                                or "application/octet-stream",
                            ),
                        }
                    )
                continue
            if "choice" in data_type and value_text:
                coerced = _coerce_choice_value(
                    value_text, choice_by_uuid.get(item_uuid) or None
                )
                if coerced is None:
                    continue
                chunk_values[item_uuid] = coerced
                continue
            if value_text:
                value_to_use: Any = value_text
            elif value_num is not None and not (isinstance(value_num, float) and pd.isna(value_num)):
                try:
                    value_to_use = float(value_num)
                    if value_to_use.is_integer():
                        value_to_use = int(value_to_use)
                except (TypeError, ValueError):
                    value_to_use = value_num
            else:
                continue
            chunk_values[item_uuid] = value_to_use

        if rejection_reasons:
            rejected.append(
                {
                    "row": int(chunk.index.min()) + 2,
                    "reasons": rejection_reasons,
                }
            )
            continue

        observation_index = len(observations)
        observations.append(
            {
                "survey_uuid": str(uuid_lib.uuid4()),
                "project_system_id": sys_id,
                "procedure_id": proc_id,
                "recorded_at": str(chunk["_iso"].iloc[0]),
                "lon": float(chunk[lon_col].iloc[0]),
                "lat": float(chunk[lat_col].iloc[0]),
                "values": chunk_values,
            }
        )
        for media in chunk_media:
            media_uploads.append(
                {
                    "obs_index": observation_index,
                    **media,
                }
            )

    unresolved_rows = work.loc[unresolved_mask, [item_name_col, item_uuid_col]].copy()
    return {
        "format": detected,
        "observations": observations,
        # In wide format ``valid`` counts item-values (one per matched column),
        # so report the number of source records that produced observations.
        "resolved_rows": len(observations),
        "resolved_values": len(valid),
        "unresolved_rows": unresolved_rows.reset_index(drop=True),
        "ignored_columns": ignored_columns,
        "media_uploads": media_uploads,
        "rejected": rejected,
    }


def upload_observations_from_csv(
    hdr: AuthHeaders,
    csv_path: str | Path,
    *,
    procedure: dict[str, Any] | None = None,
    system_name: str | None = None,
    procedure_name: str | None = None,
    system_index: int | None = None,
    procedure_index: int | None = None,
    system_id: int | None = None,
    procedure_id: int | None = None,
    dry_run: bool = False,
    batch_size: int = UPLOAD_OBSERVATIONS_MAX_BATCH,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read a CSV, build observation payloads, and optionally upload them.

    Pass a ``procedure`` from :func:`get_procedure` (preferred), or omit it and
    resolve via ``system_name`` / ``procedure_name`` against a freshly fetched
    schema.

    Uploads larger than ``batch_size`` (API max 500) are sent in sequential
    batches. Media values are treated as local file paths, uploaded through
    signed URLs, and replaced with cloud ``blob_path`` values before the
    observation payload is submitted. Pass ``media_dir`` through ``kwargs``
    when the CSV contains relative media filenames.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path must exist: {csv_path}")

    observation_data = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    schema = None if procedure is not None else get_project_systems(hdr)
    built = build_upload_observations_from_table(
        data=observation_data,
        procedure=procedure,
        schema=schema,
        system_name=system_name,
        procedure_name=procedure_name,
        system_index=system_index,
        procedure_index=procedure_index,
        system_id=system_id,
        procedure_id=procedure_id,
        **kwargs,
    )
    if dry_run:
        return {
            "uploaded": False,
            "format": built["format"],
            "observations": built["observations"],
            "resolved_rows": built["resolved_rows"],
            "resolved_values": built["resolved_values"],
            "unresolved_rows": built["unresolved_rows"],
            "ignored_columns": built["ignored_columns"],
            "media_uploads": built["media_uploads"],
            "rejected": built["rejected"],
        }

    if not built["observations"]:
        raise ValueError(
            "No observations are eligible for upload. "
            "Check the returned rejected rows with dry_run=True."
        )

    if built["media_uploads"]:
        print(f"Uploading {len(built['media_uploads'])} media file(s)...")
        blob_by_filename = _upload_pending_media(hdr, built["media_uploads"])
        for media in built["media_uploads"]:
            filename = Path(media["filepath"]).name
            built["observations"][media["obs_index"]]["values"][
                media["item_uuid"]
            ] = blob_by_filename[filename]

    response = upload_observations(
        hdr, built["observations"], batch_size=batch_size
    )
    # upload_observations already printed a summary; rebuild without reprinting.
    summary = summarise_upload_results(response, print_summary=False)
    return {
        "uploaded": summary["any_succeeded"],
        "all_succeeded": summary["all_succeeded"],
        "succeeded": summary["succeeded"],
        "failed": summary["failed"],
        "error_counts": summary["error_counts"],
        "errors": summary["errors"],
        "response": response,
        "format": built["format"],
        "observations": built["observations"],
        "resolved_rows": built["resolved_rows"],
        "resolved_values": built["resolved_values"],
        "unresolved_rows": built["unresolved_rows"],
        "ignored_columns": built["ignored_columns"],
        "media_uploads": built["media_uploads"],
        "rejected": built["rejected"],
    }
