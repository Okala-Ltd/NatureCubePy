"""Tests for naturecubepy.phone_observations module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from naturecubepy.phone_observations import (
    PHONE_TYPES,
    _collect_media_files,
    build_device_settings,
    build_feature_record,
    build_observation,
    upload_phone_observations,
    validate_observation_payload,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hdr():
    from naturecubepy.api import auth_headers
    return auth_headers("test-api-key")


@pytest.fixture()
def valid_device():
    return build_device_settings(
        device_id="dev-001",
        phone_model="iPhone 14",
        phone_os="iOS 17",
        carrier="Vodafone",
        build_number="1.0.0",
        build_id="build-001",
    )


@pytest.fixture()
def valid_geometry():
    return {"type": "Point", "coordinates": [-1.5, 53.4]}


@pytest.fixture()
def text_observation(valid_geometry):
    return build_observation(
        item_uuid="item-uuid-1",
        item_type="text",
        data=["Hello, world!"],
        geometry=valid_geometry,
    )


@pytest.fixture()
def valid_feature(valid_geometry, text_observation):
    return build_feature_record(
        feature_uuid="feature-uuid-1",
        project_system_id=10,
        procedure_id=5,
        start_time=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        created_by_method="drawn",
        geometry=valid_geometry,
        observations=[text_observation],
    )


# ---------------------------------------------------------------------------
# PHONE_TYPES
# ---------------------------------------------------------------------------

class TestPhoneTypes:
    def test_contains_expected_types(self):
        expected = {
            "phone-photo", "phone-video", "phone-audio",
            "choice", "text", "numeric", "label", "instruction",
        }
        assert set(PHONE_TYPES) == expected


# ---------------------------------------------------------------------------
# build_device_settings
# ---------------------------------------------------------------------------

class TestBuildDeviceSettings:
    def test_returns_dict_with_expected_keys(self, valid_device):
        expected_keys = {
            "device_id", "phone_model", "phone_operating_system",
            "carrier", "build_number", "build_id",
            "battery_level", "device_created_at", "device_last_used",
        }
        assert expected_keys.issubset(valid_device.keys())

    def test_default_battery_level(self, valid_device):
        assert valid_device["battery_level"] == 100.0

    def test_custom_battery_level(self):
        device = build_device_settings(
            device_id="d1", phone_model="Pixel", phone_os="Android",
            carrier="EE", build_number="2.0", build_id="b2",
            battery_level=75,
        )
        assert device["battery_level"] == 75.0

    def test_raises_missing_device_id(self):
        with pytest.raises(ValueError, match="device_id"):
            build_device_settings(
                device_id="",
                phone_model="iPhone", phone_os="iOS",
                carrier="Vodafone", build_number="1.0", build_id="b1",
            )

    def test_raises_invalid_battery_level(self):
        with pytest.raises(ValueError, match="battery_level"):
            build_device_settings(
                device_id="d1", phone_model="Phone", phone_os="OS",
                carrier="Net", build_number="1.0", build_id="b1",
                battery_level=150,
            )

    def test_timestamps_in_iso_format(self, valid_device):
        import re
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(pattern, valid_device["device_created_at"])
        assert re.match(pattern, valid_device["device_last_used"])


# ---------------------------------------------------------------------------
# build_observation
# ---------------------------------------------------------------------------

class TestBuildObservation:
    def test_returns_geojson_feature(self, valid_geometry):
        obs = build_observation(
            item_uuid="uuid-1",
            item_type="text",
            data=["some text"],
            geometry=valid_geometry,
        )
        assert obs["type"] == "Feature"
        assert obs["geometry"] == valid_geometry
        assert obs["properties"]["item_type"] == "text"

    def test_auto_generates_uuid(self, valid_geometry):
        obs = build_observation(
            item_uuid="uuid-1",
            item_type="text",
            data=["data"],
            geometry=valid_geometry,
        )
        assert obs["properties"]["observation_uuid"]

    def test_accepts_custom_uuid(self, valid_geometry):
        obs = build_observation(
            item_uuid="uuid-1",
            item_type="text",
            data=["data"],
            geometry=valid_geometry,
            observation_uuid="custom-uuid",
        )
        assert obs["properties"]["observation_uuid"] == "custom-uuid"

    def test_raises_invalid_item_type(self, valid_geometry):
        with pytest.raises(ValueError, match="item_type"):
            build_observation(
                item_uuid="uuid-1",
                item_type="invalid-type",
                data=["data"],
                geometry=valid_geometry,
            )

    def test_raises_missing_item_uuid(self, valid_geometry):
        with pytest.raises(ValueError, match="item_uuid"):
            build_observation(
                item_uuid="",
                item_type="text",
                data=["data"],
                geometry=valid_geometry,
            )

    def test_raises_invalid_geometry_type(self):
        with pytest.raises(ValueError, match="geometry type"):
            build_observation(
                item_uuid="uuid-1",
                item_type="text",
                data=["data"],
                geometry={"type": "Circle", "coordinates": [0, 0]},
            )

    def test_raises_missing_geometry(self):
        with pytest.raises(ValueError, match="geometry"):
            build_observation(
                item_uuid="uuid-1",
                item_type="text",
                data=["data"],
                geometry=None,
            )

    def test_data_stored_as_list(self, valid_geometry):
        obs = build_observation(
            item_uuid="uuid-1",
            item_type="text",
            data="single string",
            geometry=valid_geometry,
        )
        assert isinstance(obs["properties"]["data"], list)

    @pytest.mark.parametrize("item_type", PHONE_TYPES)
    def test_all_valid_item_types(self, item_type, valid_geometry):
        obs = build_observation(
            item_uuid="uuid-1",
            item_type=item_type,
            data=["val"],
            geometry=valid_geometry,
        )
        assert obs["properties"]["item_type"] == item_type


# ---------------------------------------------------------------------------
# build_feature_record
# ---------------------------------------------------------------------------

class TestBuildFeatureRecord:
    def test_returns_dict_with_required_keys(self, valid_feature):
        expected = {
            "feature_uuid", "project_system_id", "procedure_id",
            "procedure_start_timestamp", "procedure_end_timestamp",
            "created_by_method", "geometry", "observations",
        }
        assert expected.issubset(valid_feature.keys())

    def test_project_system_id_is_int(self, valid_feature):
        assert isinstance(valid_feature["project_system_id"], int)

    def test_raises_invalid_created_by_method(self, valid_geometry):
        with pytest.raises(ValueError, match="created_by_method"):
            build_feature_record(
                feature_uuid="f1",
                project_system_id=1,
                procedure_id=1,
                start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
                created_by_method="flying",
                geometry=valid_geometry,
                observations=[],
            )

    def test_raises_missing_feature_uuid(self, valid_geometry):
        with pytest.raises(ValueError, match="feature_uuid"):
            build_feature_record(
                feature_uuid="",
                project_system_id=1,
                procedure_id=1,
                start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
                end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
                created_by_method="drawn",
                geometry=valid_geometry,
                observations=[],
            )

    def test_timestamps_in_iso_format(self, valid_feature):
        import re
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(pattern, valid_feature["procedure_start_timestamp"])
        assert re.match(pattern, valid_feature["procedure_end_timestamp"])

    def test_accepts_traced_method(self, valid_geometry):
        feature = build_feature_record(
            feature_uuid="f1",
            project_system_id=1,
            procedure_id=1,
            start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
            created_by_method="traced",
            geometry=valid_geometry,
            observations=[],
        )
        assert feature["created_by_method"] == "traced"


# ---------------------------------------------------------------------------
# validate_observation_payload
# ---------------------------------------------------------------------------

class TestValidateObservationPayload:
    def test_valid_payload(self, valid_feature, valid_device):
        result = validate_observation_payload([valid_feature], valid_device)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_empty_feature_payload(self, valid_device):
        result = validate_observation_payload([], valid_device)
        assert result["valid"] is False
        assert any("non-empty" in e for e in result["errors"])

    def test_missing_device_field(self, valid_feature):
        bad_device = {"device_id": "d1", "phone_model": "Phone"}  # incomplete
        result = validate_observation_payload([valid_feature], bad_device)
        assert result["valid"] is False
        assert any("Missing device settings" in e for e in result["errors"])

    def test_invalid_geometry_type(self, valid_device, text_observation):
        feature = build_feature_record(
            feature_uuid="f1",
            project_system_id=1,
            procedure_id=1,
            start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
            created_by_method="drawn",
            geometry={"type": "Point", "coordinates": [0, 0]},
            observations=[text_observation],
        )
        # Manually corrupt geometry type
        feature["geometry"]["type"] = "Blob"
        result = validate_observation_payload([feature], valid_device)
        # The geometry is already built; the validate function should catch the bad type
        assert result["valid"] is False

    def test_media_without_media_dir(self, valid_device):
        photo_obs = build_observation(
            item_uuid="uuid-photo",
            item_type="phone-photo",
            data=["photo.jpg"],
            geometry={"type": "Point", "coordinates": [0, 0]},
        )
        feature = build_feature_record(
            feature_uuid="f1",
            project_system_id=1,
            procedure_id=1,
            start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
            created_by_method="drawn",
            geometry={"type": "Point", "coordinates": [0, 0]},
            observations=[photo_obs],
        )
        result = validate_observation_payload([feature], valid_device, media_dir=None)
        assert result["valid"] is False
        assert any("media_dir is required" in e for e in result["errors"])

    def test_media_file_not_found(self, valid_device, tmp_path):
        photo_obs = build_observation(
            item_uuid="uuid-photo",
            item_type="phone-photo",
            data=["missing.jpg"],
            geometry={"type": "Point", "coordinates": [0, 0]},
        )
        feature = build_feature_record(
            feature_uuid="f1",
            project_system_id=1,
            procedure_id=1,
            start_time=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 1, tzinfo=timezone.utc),
            created_by_method="drawn",
            geometry={"type": "Point", "coordinates": [0, 0]},
            observations=[photo_obs],
        )
        result = validate_observation_payload([feature], valid_device, media_dir=str(tmp_path))
        assert result["valid"] is False
        assert any("Media file not found" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# _collect_media_files
# ---------------------------------------------------------------------------

class TestCollectMediaFiles:
    def test_collects_photo(self, tmp_path, valid_geometry):
        (tmp_path / "photo.jpg").write_bytes(b"FAKE_IMAGE")
        obs = build_observation(
            item_uuid="u1",
            item_type="phone-photo",
            data=["photo.jpg"],
            geometry=valid_geometry,
        )
        media = _collect_media_files([obs], tmp_path)
        assert "photo.jpg" in media
        assert media["photo.jpg"][1] == "image/jpeg"

    def test_skips_nonexistent_files(self, tmp_path, valid_geometry):
        obs = build_observation(
            item_uuid="u1",
            item_type="phone-photo",
            data=["missing.jpg"],
            geometry=valid_geometry,
        )
        media = _collect_media_files([obs], tmp_path)
        assert media == {}

    def test_non_media_observation_ignored(self, tmp_path, valid_geometry):
        obs = build_observation(
            item_uuid="u1",
            item_type="text",
            data=["hello"],
            geometry=valid_geometry,
        )
        media = _collect_media_files([obs], tmp_path)
        assert media == {}


# ---------------------------------------------------------------------------
# upload_phone_observations
# ---------------------------------------------------------------------------

class TestUploadPhoneObservations:
    def test_successful_upload(self, hdr, valid_device, valid_feature):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        with patch("naturecubepy.phone_observations.httpx.post", return_value=mock_response):
            result = upload_phone_observations(
                hdr=hdr,
                project_id=42,
                feature_payload=[valid_feature],
                device_settings=valid_device,
                validate=True,
            )
        assert len(result["successes"]) == 1
        assert len(result["failures"]) == 0
        assert "1 of 1" in result["summary"]

    def test_validation_failure_raises(self, hdr):
        bad_device = {"device_id": "d1"}
        with pytest.raises(ValueError, match="Validation failed"):
            upload_phone_observations(
                hdr=hdr,
                project_id=1,
                feature_payload=[{"feature_uuid": "f1"}],
                device_settings=bad_device,
                validate=True,
            )

    def test_partial_failure_collected(self, hdr, valid_device, valid_feature):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        with patch("naturecubepy.phone_observations.httpx.post", return_value=mock_response):
            result = upload_phone_observations(
                hdr=hdr,
                project_id=42,
                feature_payload=[valid_feature],
                device_settings=valid_device,
                validate=False,
            )
        assert len(result["failures"]) == 1
        assert "1 failed" in result["summary"]

    def test_validate_false_skips_validation(self, hdr, valid_feature):
        bad_device = {}  # would fail validation
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        with patch("naturecubepy.phone_observations.httpx.post", return_value=mock_response):
            # Should not raise even with bad device because validate=False
            result = upload_phone_observations(
                hdr=hdr,
                project_id=1,
                feature_payload=[valid_feature],
                device_settings=bad_device,
                validate=False,
            )
        assert len(result["successes"]) == 1
