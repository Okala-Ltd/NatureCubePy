"""
Phone observation builders and upload functions for the Okala dashboard.

This module provides functions for constructing structured observation records
from mobile devices (photos, videos, audio, and form data) and uploading them
to the Okala platform.
"""

from __future__ import annotations

import mimetypes
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

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


def _collect_media_files(
    observations: list[dict[str, Any]],
    media_dir: str | Path,
) -> dict[str, tuple[bytes, str]]:
    """Extract media file bytes from observations for multipart upload.

    Parameters
    ----------
    observations:
        List of observation records.
    media_dir:
        Directory containing the media files.

    Returns
    -------
    dict
        Mapping of ``filename -> (bytes, mime_type)`` for each media file found.
    """
    media_dir = Path(media_dir)
    media_files: dict[str, tuple[bytes, str]] = {}

    for obs in observations:
        item_type = obs.get("properties", {}).get("item_type", "")
        if item_type in _MEDIA_TYPES:
            filenames = obs.get("properties", {}).get("data", [])
            for filename in filenames:
                filepath = media_dir / filename
                if filepath.exists():
                    mime_type = _MIME_MAP.get(item_type, mimetypes.guess_type(filename)[0] or "application/octet-stream")
                    media_files[filename] = (filepath.read_bytes(), mime_type)

    return media_files


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
    hdr: dict[str, Any],
    project_id: int,
    feature_payload: list[dict[str, Any]],
    device_settings: dict[str, Any],
    media_dir: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Upload phone observation records to the Okala platform.

    Processes one feature at a time. Partial failures are collected and
    returned rather than stopping the upload.

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

    for i, feature in enumerate(feature_payload, start=1):
        feature_uuid = feature.get("feature_uuid", f"feature-{i}")
        print(f"Uploading feature {i} of {n_features} ({feature_uuid})...")

        try:
            device_upload = {
                "feature_payload": [feature],
                "device_settings": device_settings,
            }

            url = f"{hdr['root']}pushObservation/{hdr['key']}/{project_id}"

            media_files = {}
            if media_dir is not None:
                media_files = _collect_media_files(
                    feature.get("observations", []), media_dir
                )

            if media_files:
                import json

                files: dict[str, Any] = {
                    "device_upload": (None, json.dumps(device_upload), "application/json"),
                }
                for filename, (file_bytes, mime_type) in media_files.items():
                    files[filename] = (filename, file_bytes, mime_type)
                response = httpx.post(url, files=files)
            else:
                response = httpx.post(url, json=device_upload)

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
