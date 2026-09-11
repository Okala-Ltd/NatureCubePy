"""Tests for naturecubepy.phone_observations module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from naturecubepy.phone_observations import (
    PHONE_TYPES,
    _collect_pending_media,
    build_device_settings,
    build_feature_record,
    build_observation,
    extract_taxonomic_label,
    get_phone_observation_data,
    phone_observations_to_wide,
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
# _collect_pending_media
# ---------------------------------------------------------------------------

class TestCollectPendingMedia:
    def test_collects_photo(self, tmp_path, valid_geometry):
        (tmp_path / "photo.jpg").write_bytes(b"FAKE_IMAGE")
        obs = build_observation(
            item_uuid="u1",
            item_type="phone-photo",
            data=["photo.jpg"],
            geometry=valid_geometry,
        )
        media = _collect_pending_media([obs], tmp_path)
        assert len(media) == 1
        assert media[0]["filepath"] == tmp_path / "photo.jpg"
        assert media[0]["data_type"] == "phone-photo"
        assert media[0]["content_type"] == "image/jpeg"

    def test_includes_missing_files_for_later_rejection(self, tmp_path, valid_geometry):
        obs = build_observation(
            item_uuid="u1",
            item_type="phone-photo",
            data=["missing.jpg"],
            geometry=valid_geometry,
        )
        media = _collect_pending_media([obs], tmp_path)
        assert len(media) == 1
        assert media[0]["filepath"] == tmp_path / "missing.jpg"

    def test_non_media_observation_ignored(self, tmp_path, valid_geometry):
        obs = build_observation(
            item_uuid="u1",
            item_type="text",
            data=["hello"],
            geometry=valid_geometry,
        )
        media = _collect_pending_media([obs], tmp_path)
        assert media == []


# ---------------------------------------------------------------------------
# get_phone_observation_data
# ---------------------------------------------------------------------------

class TestGetPhoneObservationData:
    def test_paginates_csv_with_keyset_cursor(self, hdr):
        page1 = MagicMock()
        page1.status_code = 200
        page1.headers = {
            "content-type": "text/csv",
            "X-Next-Observation-Id": "102",
        }
        page1.raise_for_status = MagicMock()
        page1.content = (
            b"observation_id,item_name,data_type,data\n"
            b"101,Notes,text,hello\n"
            b"102,Notes,text,world\n"
        )

        page2 = MagicMock()
        page2.status_code = 200
        page2.headers = {"content-type": "text/csv"}
        page2.raise_for_status = MagicMock()
        page2.content = (
            b"observation_id,item_name,data_type,data\n"
            b"103,Notes,text,again\n"
        )

        with patch(
            "naturecubepy.phone_observations.httpx.get",
            side_effect=[page1, page2],
        ) as get:
            result = get_phone_observation_data(
                hdr,
                procedure_id=7,
                data_type="text",
                page_size=2,
                project_id=42,
            )

        assert list(result["observation_id"]) == [101, 102, 103]
        assert get.call_count == 2
        first_params = get.call_args_list[0].kwargs["params"]
        second_params = get.call_args_list[1].kwargs["params"]
        assert first_params["limit"] == 2
        assert "after_observation_id" not in first_params
        assert second_params["after_observation_id"] == 102

    def test_empty_media_page_returns_empty_dataframe(self, hdr):
        empty = MagicMock()
        empty.status_code = 200
        empty.headers = {"content-type": "application/json"}
        empty.raise_for_status = MagicMock()
        empty.json.return_value = {
            "download_url": "",
            "expires_at": "",
            "row_count": 0,
            "media_count": 0,
            "format": "zip",
            "filename": "",
            "size_bytes": 0,
        }
        with patch("naturecubepy.phone_observations.httpx.get", return_value=empty):
            result = get_phone_observation_data(
                hdr, procedure_id=7, data_type="phone-photo", project_id=42
            )
        assert result.empty

    def test_rejects_invalid_data_type(self, hdr):
        with pytest.raises(ValueError, match="Unsupported data_type"):
            get_phone_observation_data(
                hdr, procedure_id=7, data_type="instruction", project_id=42
            )

    def test_media_export_reads_csv_from_zip(self, hdr, tmp_path):
        import io
        from zipfile import ZipFile, ZIP_DEFLATED

        buf = io.BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
            zf.writestr(
                "observations.csv",
                "observation_id,item_name,data_type,data\n"
                "201,Photo,phone-photo,path/a.jpg\n",
            )
            zf.writestr("media/a.jpg", b"fake-image-bytes")
        zip_bytes = buf.getvalue()

        meta = MagicMock()
        meta.status_code = 200
        meta.headers = {"content-type": "application/json"}
        meta.raise_for_status = MagicMock()
        meta.json.return_value = {
            "download_url": "https://example.test/export.zip",
            "row_count": 1,
            "media_count": 1,
            "format": "zip",
            "filename": "export.zip",
            "size_bytes": len(zip_bytes),
            "expires_at": "2026-01-01T00:00:00Z",
        }

        zip_resp = MagicMock()
        zip_resp.raise_for_status = MagicMock()
        zip_resp.content = zip_bytes

        with patch(
            "naturecubepy.phone_observations.httpx.get",
            side_effect=[meta, zip_resp],
        ):
            result = get_phone_observation_data(
                hdr,
                procedure_id=7,
                data_type="phone-photo",
                media_dir=tmp_path / "media",
                project_id=42,
            )

        assert len(result) == 1
        assert int(result["observation_id"].iloc[0]) == 201
        assert (tmp_path / "media" / "phone-photo" / "a.jpg").read_bytes() == b"fake-image-bytes"

    def test_resolves_project_id_from_api_key(self, hdr):
        project_resp = MagicMock()
        project_resp.raise_for_status = MagicMock()
        project_resp.json.return_value = {
            "boundary": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"project_id": 99, "project_name": "Demo"},
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                    }
                ],
            }
        }

        page = MagicMock()
        page.status_code = 200
        page.headers = {"content-type": "text/csv"}
        page.raise_for_status = MagicMock()
        page.content = b"observation_id,item_name,data_type,data\n11,Notes,text,hi\n"

        with patch(
            "naturecubepy.phone_observations.httpx.get",
            side_effect=[project_resp, page],
        ) as get:
            result = get_phone_observation_data(
                hdr, procedure_id=7, data_type="text"
            )

        assert list(result["observation_id"]) == [11]
        assert "/getProject/" in get.call_args_list[0].args[0]
        assert "/getPhoneObservations/" in get.call_args_list[1].args[0]
        assert "/99/" in get.call_args_list[1].args[0]

    def test_phone_observations_to_wide(self):
        long = pd.DataFrame(
            [
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Notes",
                    "data": "hello",
                },
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Count",
                    "data": "3",
                },
                {
                    "feature_uuid": "f2",
                    "longitude": 3.0,
                    "latitude": 4.0,
                    "item_name": "Notes",
                    "data": "bye",
                },
            ]
        )
        wide = phone_observations_to_wide(long)
        assert list(wide["feature_uuid"]) == ["f1", "f2"]
        assert "Notes" in wide.columns and "Count" in wide.columns
        row1 = wide.set_index("feature_uuid").loc["f1"]
        assert row1["Notes"] == "hello"
        assert row1["Count"] == "3"

    def test_wide_keeps_all_items_when_coords_drift(self):
        """GPS noise within a feature must not split/sparsify wide rows."""
        long = pd.DataFrame(
            [
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Notes",
                    "data": "hello",
                },
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0000001,
                    "latitude": 2.0,
                    "item_name": "Count",
                    "data": "3",
                },
                {
                    "feature_uuid": "f2",
                    "longitude": 3.0,
                    "latitude": 4.0,
                    "item_name": "Notes",
                    "data": "bye",
                },
            ]
        )
        wide = phone_observations_to_wide(long)
        assert len(wide) == 2
        row1 = wide.set_index("feature_uuid").loc["f1"]
        assert row1["Notes"] == "hello"
        assert row1["Count"] == "3"

    def test_wide_preserves_list_valued_data(self):
        long = pd.DataFrame(
            [
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Notes",
                    "data_type": "text",
                    "data": ["hello", "world"],
                }
            ]
        )
        wide = phone_observations_to_wide(long)
        assert wide.loc[0, "Notes"] == "hello, world"

    def test_extracts_taxonomic_label_in_wide(self):
        long = pd.DataFrame(
            [
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Taxonomic label",
                    "data_type": "label",
                    "data": json.dumps(
                        [
                            {
                                "label_id": 1,
                                "label": "Panthera leo",
                                "common_name": "Lion",
                                "species": "Panthera leo",
                            }
                        ]
                    ),
                    "labels": json.dumps(
                        [
                            {
                                "label_id": 1,
                                "label": "Panthera leo",
                                "common_name": "Lion",
                            }
                        ]
                    ),
                },
                {
                    "feature_uuid": "f1",
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "item_name": "Notes",
                    "data_type": "text",
                    "data": "seen at dusk",
                    "labels": "",
                },
            ]
        )
        assert extract_taxonomic_label(long.loc[0, "data"]) == "Panthera leo"
        wide = phone_observations_to_wide(long)
        assert wide.loc[0, "Taxonomic label"] == "Panthera leo"
        assert wide.loc[0, "Notes"] == "seen at dusk"

    def test_retries_on_http_429(self, hdr):
        limited = MagicMock()
        limited.status_code = 429
        limited.headers = {"Retry-After": "0"}
        limited.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "429", request=MagicMock(), response=limited
            )
        )

        ok = MagicMock()
        ok.status_code = 200
        ok.headers = {"content-type": "text/csv"}
        ok.raise_for_status = MagicMock()
        ok.content = b"observation_id,item_name,data_type,data\n11,Notes,text,hi\n"

        with (
            patch(
                "naturecubepy.phone_observations.httpx.get",
                side_effect=[limited, ok],
            ),
            patch("naturecubepy.phone_observations.time.sleep") as sleep,
        ):
            result = get_phone_observation_data(
                hdr, procedure_id=7, data_type="text", project_id=42
            )

        assert list(result["observation_id"]) == [11]
        sleep.assert_called_once()


# ---------------------------------------------------------------------------
# upload_phone_observations
# ---------------------------------------------------------------------------

class TestUploadPhoneObservations:
    def test_successful_upload(self, hdr, valid_device, valid_feature):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        with patch("naturecubepy.phone_observations.httpx.post", return_value=mock_response) as post:
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
        assert post.call_args.args[0].endswith("pushPhoneObservations/test-api-key/42")
        assert "data" in post.call_args.kwargs["data"]

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

    def test_media_uses_signed_url_then_metadata_route(
        self, hdr, valid_device, valid_geometry, tmp_path: Path
    ):
        image = tmp_path / "roadkill.jpg"
        image.write_bytes(b"jpeg-data")
        photo = build_observation(
            item_uuid="photo-item",
            item_type="phone-photo",
            data=[image.name],
            geometry=valid_geometry,
        )
        feature = build_feature_record(
            feature_uuid="feature-photo",
            project_system_id=10,
            procedure_id=5,
            start_time=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
            created_by_method="drawn",
            geometry=valid_geometry,
            observations=[photo],
        )

        presign = MagicMock()
        presign.json.return_value = {
            "files": [
                {
                    "filename": image.name,
                    "blob_path": f"field_records/test/{image.name}",
                    "signed_url": "https://storage.example/upload",
                    "content_type": "image/jpeg",
                }
            ]
        }
        metadata = MagicMock()
        metadata.json.return_value = [{"status": "success"}]

        with (
            patch(
                "naturecubepy.phone_observations.httpx.post",
                side_effect=[presign, metadata],
            ) as post,
            patch("naturecubepy.phone_observations.httpx.put") as put,
        ):
            result = upload_phone_observations(
                hdr=hdr,
                project_id=42,
                feature_payload=[feature],
                device_settings=valid_device,
                media_dir=tmp_path,
            )

        assert not result["failures"]
        assert post.call_args_list[0].args[0].endswith(
            "getFieldMediaUploadUrls/test-api-key"
        )
        assert post.call_args_list[1].args[0].endswith(
            "pushPhoneObservations/test-api-key/42"
        )
        assert "data" in post.call_args_list[1].kwargs["data"]
        put.assert_called_once_with(
            "https://storage.example/upload",
            content=b"jpeg-data",
            headers={"Content-Type": "image/jpeg"},
            timeout=180.0,
        )
# ---------------------------------------------------------------------------
# Schema / CSV uploadObservations workflow
# ---------------------------------------------------------------------------

def _sample_schema() -> dict:
    return {
        "systems": [
            {
                "system_name": "Plante Ivindo",
                "project_system_id": 10,
                "procedures": [
                    {
                        "procedure_name": "Arbre",
                        "procedure_id": 5,
                        "form": True,
                        "items": [
                            {
                                "item_id": 1,
                                "item_uuid": "uuid-tax",
                                "item_name": "Taxonomic label",
                                "data_type": "text",
                                "nullable": False,
                            },
                            {
                                "item_id": 2,
                                "item_uuid": "uuid-common",
                                "item_name": "Nom commun",
                                "data_type": "text",
                                "nullable": True,
                            },
                        ],
                    }
                ],
            }
        ]
    }


class TestProjectSchemaHelpers:
    def test_list_systems_and_get_procedure(self):
        from naturecubepy.phone_observations import get_procedure, list_systems

        schema = _sample_schema()
        systems = list_systems(schema)
        assert len(systems) == 1
        assert systems.iloc[0]["system_name"] == "Plante Ivindo"

        procedure = get_procedure(
            schema,
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        assert procedure["system_id"] == 10
        assert procedure["procedure_id"] == 5
        assert set(procedure["items"]["item_name"]) == {"Taxonomic label", "Nom commun"}

    def test_get_procedure_by_api_ids(self):
        from naturecubepy.phone_observations import get_procedure

        procedure = get_procedure(_sample_schema(), system_id=10, procedure_id=5)

        assert procedure["system_id"] == 10
        assert procedure["procedure_id"] == 5

    def test_get_procedure_rejects_id_passed_as_index(self):
        """Passing an api id as a position should say so rather than just fail."""
        from naturecubepy.phone_observations import get_procedure

        with pytest.raises(ValueError, match="looks like a project_system_id"):
            get_procedure(_sample_schema(), system_index=10)

        with pytest.raises(ValueError, match="looks like a procedure_id"):
            get_procedure(_sample_schema(), system_index=1, procedure_index=5)

    def test_get_procedure_unknown_id_lists_available(self):
        from naturecubepy.phone_observations import get_procedure

        with pytest.raises(ValueError, match="Available system_id values: 10"):
            get_procedure(_sample_schema(), system_id=999)


class TestCsvUploadWorkflow:
    def test_csv_media_is_presigned_then_uploaded_as_blob_path(
        self, hdr, tmp_path: Path
    ):
        from naturecubepy.phone_observations import upload_observations_from_csv

        media_dir = tmp_path / "media"
        media_dir.mkdir()
        image = media_dir / "roadkill.jpg"
        image.write_bytes(b"jpeg-data")
        csv_path = tmp_path / "roadkill.csv"
        csv_path.write_text(
            "recorded_at,longitude,latitude,image\n"
            "01/01/2024,-8.5,41.0,roadkill.jpg\n",
            encoding="utf-8",
        )
        procedure = {
            "system_id": 10,
            "procedure_id": 5,
            "items": pd.DataFrame(
                [
                    {
                        "item_uuid": "uuid-photo",
                        "item_name": "animal_image",
                        "data_type": "phone-photo",
                    }
                ]
            ),
        }

        presign = MagicMock()
        presign.status_code = 200
        presign.json.return_value = {
            "files": [
                {
                    "filename": image.name,
                    "blob_path": "organisations/test/field_records/roadkill.jpg",
                    "signed_url": "https://storage.example/upload",
                    "content_type": "image/jpeg",
                }
            ]
        }
        uploaded = MagicMock()
        uploaded.status_code = 200
        uploaded.json.return_value = [
            {"survey_uuid": "survey-1", "status": "success", "message": None}
        ]

        with (
            patch(
                "naturecubepy.phone_observations.httpx.post",
                side_effect=[presign, uploaded],
            ) as post,
            patch("naturecubepy.phone_observations.httpx.put") as put,
        ):
            result = upload_observations_from_csv(
                hdr,
                csv_path,
                procedure=procedure,
                media_dir=media_dir,
                column_map={"image": "animal_image"},
                recorded_at_format="%d/%m/%Y",
            )

        assert result["succeeded"] == 1
        assert post.call_args_list[0].args[0].endswith(
            "getFieldMediaUploadUrls/test-api-key"
        )
        uploaded_values = post.call_args_list[1].kwargs["json"]["observations"][0][
            "values"
        ]
        assert uploaded_values == {
            "uuid-photo": "organisations/test/field_records/roadkill.jpg"
        }
        put.assert_called_once_with(
            "https://storage.example/upload",
            content=b"jpeg-data",
            headers={"Content-Type": "image/jpeg"},
            timeout=180.0,
        )

    def test_build_from_long_table(self):
        from naturecubepy.phone_observations import (
            build_upload_observations_from_table,
            get_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        data = pd.DataFrame(
            [
                {
                    "longitude": 13.7,
                    "latitude": 0.93,
                    "recorded_at": "15/04/2026 08:39",
                    "item_name": "Taxonomic label",
                    "data": "Myrianthus",
                },
                {
                    "longitude": 13.7,
                    "latitude": 0.93,
                    "recorded_at": "15/04/2026 08:39",
                    "item_name": "Nom commun",
                    "data": "Oboba",
                },
            ]
        )
        built = build_upload_observations_from_table(
            data=data,
            procedure=procedure,
            recorded_at_format="%d/%m/%Y %H:%M",
        )
        assert built["format"] == "long"
        assert built["resolved_rows"] == 1
        assert built["resolved_values"] == 2
        assert len(built["observations"]) == 1
        obs = built["observations"][0]
        assert obs["project_system_id"] == 10
        assert obs["procedure_id"] == 5
        assert obs["values"]["uuid-tax"] == "Myrianthus"
        assert obs["values"]["uuid-common"] == "Oboba"
        assert obs["recorded_at"] == "2026-04-15T08:39:00Z"

    def test_build_from_wide_table(self):
        """Wide rows become one observation each, with extra columns reported."""
        from naturecubepy.phone_observations import (
            build_upload_observations_from_table,
            get_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        data = pd.DataFrame(
            [
                {
                    "longitude": 13.7,
                    "latitude": 0.93,
                    "timestamp": "15/04/2026 08:39",
                    "Taxonomic label": "Myrianthus",
                    "Nom commun": "Oboba",
                    "observer": "PF",
                },
                {
                    "longitude": 13.7,
                    "latitude": 0.93,
                    "timestamp": "15/04/2026 08:39",
                    "Taxonomic label": "Santiria Trimera",
                    "Nom commun": "Ebo",
                    "observer": "PF",
                },
            ]
        )
        built = build_upload_observations_from_table(
            data=data,
            procedure=procedure,
            recorded_at_format="%d/%m/%Y %H:%M",
        )

        assert built["format"] == "wide"
        assert built["ignored_columns"] == ["observer"]
        # Identical coordinates and timestamps must not merge separate rows.
        assert len(built["observations"]) == 2
        values = [obs["values"] for obs in built["observations"]]
        assert {"uuid-tax": "Myrianthus", "uuid-common": "Oboba"} in values
        assert {"uuid-tax": "Santiria Trimera", "uuid-common": "Ebo"} in values

    def test_build_from_wide_table_sends_numeric_items_as_numbers(self):
        from naturecubepy.phone_observations import build_upload_observations_from_table

        procedure = {
            "system_id": 434,
            "procedure_id": 863,
            "items": pd.DataFrame(
                {
                    "item_uuid": ["u-label", "u-count"],
                    "item_name": ["label", "MIN.N.ADULT"],
                    "data_type": ["text", "numeric"],
                }
            ),
        }
        data = pd.DataFrame(
            [
                {
                    "longitude": -0.72,
                    "latitude": 50.95,
                    "timestamp": "04/04/2025",
                    "label": "Vulpes vulpes",
                    "MIN.N.ADULT": "3",
                }
            ]
        )

        built = build_upload_observations_from_table(
            data=data,
            procedure=procedure,
            recorded_at_format="%d/%m/%Y",
        )

        assert built["observations"][0]["values"] == {"u-label": "Vulpes vulpes", "u-count": 3}

    def test_validate_accepts_multi_value_choice_cell(self, tmp_path: Path):
        """A combined choice cell like 'calling;land' is valid when each part matches."""
        from naturecubepy.phone_observations import (
            get_procedure,
            validate_csv_against_procedure,
        )

        schema = {
            "systems": [
                {
                    "system_name": "Bird",
                    "project_system_id": 434,
                    "procedures": [
                        {
                            "procedure_name": "bird survey",
                            "procedure_id": 863,
                            "form": False,
                            "items": [
                                {
                                    "item_id": 1,
                                    "item_uuid": "u-beh",
                                    "item_name": "behavior",
                                    "data_type": "choice",
                                    "nullable": True,
                                    "choices": [
                                        {"label": "singing"},
                                        {"label": "calling"},
                                        {"label": "land"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        procedure = get_procedure(schema, system_id=434, procedure_id=863)
        csv_path = tmp_path / "beh.csv"
        csv_path.write_text(
            "longitude,latitude,timestamp,behavior\n"
            "-1.43,52.72,24/04/2015 07:15,Calling\n"
            "-1.45,52.74,24/04/2015 07:15,calling;land\n"
            "-1.46,52.75,24/04/2015 07:15,calling;unknown\n",
            encoding="utf-8",
        )

        result = validate_csv_against_procedure(procedure, csv_path)

        assert result["valid"] is False
        issues = result["type_issues"]
        assert len(issues) == 1
        assert issues.iloc[0]["value"] == "calling;unknown"
        assert "'unknown'" in issues.iloc[0]["problem"]

    @staticmethod
    def _choice_schema() -> dict:
        return {
            "systems": [
                {
                    "system_name": "Bird",
                    "project_system_id": 434,
                    "procedures": [
                        {
                            "procedure_name": "bird survey",
                            "procedure_id": 863,
                            "form": False,
                            "items": [
                                {
                                    "item_id": 1,
                                    "item_uuid": "u-label",
                                    "item_name": "label",
                                    "data_type": "label",
                                },
                                {
                                    "item_id": 2,
                                    "item_uuid": "u-beh",
                                    "item_name": "behavior",
                                    "data_type": "choice",
                                    "choices": [
                                        {"label": "singing"},
                                        {"label": "calling"},
                                        {"label": "land"},
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }

    def test_build_multi_value_choice_as_list_wide(self):
        from naturecubepy.phone_observations import (
            build_upload_observations_from_table,
            get_procedure,
        )

        procedure = get_procedure(self._choice_schema(), system_id=434, procedure_id=863)
        data = pd.DataFrame(
            [
                {
                    "longitude": -1.43,
                    "latitude": 52.72,
                    "timestamp": "24/04/2015 07:15",
                    "label": "Blackcap",
                    "behavior": "Calling;Land",
                },
                {
                    "longitude": -1.44,
                    "latitude": 52.73,
                    "timestamp": "24/04/2015 07:15",
                    "label": "Robin",
                    "behavior": "singing",
                },
            ]
        )

        built = build_upload_observations_from_table(
            data, procedure=procedure, recorded_at_format="%d/%m/%Y %H:%M"
        )
        assert len(built["observations"]) == 2
        values = [obs["values"] for obs in built["observations"]]
        assert {"u-label": "Blackcap", "u-beh": ["calling", "land"]} in values
        assert {"u-label": "Robin", "u-beh": "singing"} in values

    def test_build_multi_value_choice_as_list_long(self):
        from naturecubepy.phone_observations import (
            build_upload_observations_from_table,
            get_procedure,
        )

        procedure = get_procedure(self._choice_schema(), system_id=434, procedure_id=863)
        data = pd.DataFrame(
            [
                {
                    "longitude": -1.43,
                    "latitude": 52.72,
                    "recorded_at": "24/04/2015 07:15",
                    "item_name": "label",
                    "data": "Blackcap",
                },
                {
                    "longitude": -1.43,
                    "latitude": 52.72,
                    "recorded_at": "24/04/2015 07:15",
                    "item_name": "behavior",
                    "data": "calling;land",
                },
            ]
        )

        built = build_upload_observations_from_table(
            data, procedure=procedure, recorded_at_format="%d/%m/%Y %H:%M"
        )
        assert len(built["observations"]) == 1
        assert built["observations"][0]["values"] == {
            "u-label": "Blackcap",
            "u-beh": ["calling", "land"],
        }

    def test_build_from_wide_table_without_matching_columns(self):
        """A wide sheet sharing no headers with the procedure explains itself."""
        from naturecubepy.phone_observations import (
            build_upload_observations_from_table,
            get_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        data = pd.DataFrame(
            [{"longitude": 13.7, "latitude": 0.93, "timestamp": "15/04/2026 08:39", "species": "x"}]
        )

        with pytest.raises(ValueError, match="No CSV column names match"):
            build_upload_observations_from_table(data=data, procedure=procedure)

    def test_validate_csv_long(self, tmp_path: Path):
        from naturecubepy.phone_observations import (
            get_procedure,
            validate_csv_against_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        csv_path = tmp_path / "obs.csv"
        csv_path.write_text(
            "longitude,latitude,recorded_at,item_name,data\n"
            "13.7,0.93,15/04/2026 08:39,Taxonomic label,Myrianthus\n"
            "13.7,0.93,15/04/2026 08:39,Nom commun,Oboba\n",
            encoding="utf-8",
        )
        result = validate_csv_against_procedure(procedure, csv_path)
        assert result["format"] == "long"
        assert result["valid"] is True
        assert len(result["matched_items"]) == 2

    def test_validate_extra_columns_warn_not_invalid(self, tmp_path: Path):
        """Columns not in the procedure are a warning, not a blocking issue."""
        from naturecubepy.phone_observations import (
            get_procedure,
            validate_csv_against_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        csv_path = tmp_path / "extra.csv"
        csv_path.write_text(
            "longitude,latitude,recorded_at,Taxonomic label,Nom commun,observer,site_id\n"
            "13.7,0.93,15/04/2026 08:39,Myrianthus,Oboba,PF,A1\n",
            encoding="utf-8",
        )

        result = validate_csv_against_procedure(procedure, csv_path)

        assert result["valid"] is True
        assert set(result["unrecognised_names"]) == {"observer", "site_id"}
        assert any("observer" in w for w in result["warnings"])
        assert result["issues"] == []

    def test_upload_observations_from_csv_dry_run(self, hdr, tmp_path: Path):
        from naturecubepy.phone_observations import (
            get_procedure,
            upload_observations_from_csv,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        csv_path = tmp_path / "obs.csv"
        csv_path.write_text(
            "longitude,latitude,recorded_at,item_name,data\n"
            "13.7,0.93,15/04/2026 08:39,Taxonomic label,Myrianthus\n",
            encoding="utf-8",
        )
        result = upload_observations_from_csv(
            hdr,
            csv_path,
            procedure=procedure,
            dry_run=True,
            recorded_at_format="%d/%m/%Y %H:%M",
        )
        assert result["uploaded"] is False
        assert len(result["observations"]) == 1
        assert result["resolved_rows"] == 1

    def test_upload_observations_posts_payload(self, hdr):
        from naturecubepy.phone_observations import upload_observations

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"survey_uuid": "s1", "status": "success"}]
        with patch("naturecubepy.phone_observations.httpx.post", return_value=mock_response) as post:
            out = upload_observations(
                hdr,
                [
                    {
                        "survey_uuid": "s1",
                        "project_system_id": 10,
                        "procedure_id": 5,
                        "recorded_at": "2026-04-15T08:39:00Z",
                        "lon": 13.7,
                        "lat": 0.93,
                        "values": {"uuid-tax": "Myrianthus"},
                    }
                ],
            )
        assert out == [{"survey_uuid": "s1", "status": "success"}]
        assert post.call_args.kwargs["json"]["observations"][0]["values"]["uuid-tax"] == "Myrianthus"

    def test_summarise_upload_results_groups_failures(self, capsys):
        from naturecubepy.phone_observations import summarise_upload_results

        results = [
            {
                "survey_uuid": "a",
                "status": "error",
                "message": "taxonomy label 'bogus' was not found",
            },
            {
                "survey_uuid": "b",
                "status": "error",
                "message": "taxonomy label 'bogus' was not found",
            },
            {"survey_uuid": "c", "status": "success", "message": None},
        ]
        summary = summarise_upload_results(results)

        assert summary["total"] == 3
        assert summary["succeeded"] == 1
        assert summary["failed"] == 2
        assert summary["all_succeeded"] is False
        assert summary["any_succeeded"] is True
        assert summary["error_counts"] == {
            "taxonomy label 'bogus' was not found": 2
        }
        printed = capsys.readouterr().out
        assert "1 succeeded, 2 failed" in printed
        assert "Not uploaded" in printed

    def test_upload_observations_batches_over_api_limit(self, hdr):
        """Lists larger than the API max are posted in sequential batches."""
        from naturecubepy.phone_observations import upload_observations

        def _obs(i: int) -> dict:
            return {
                "survey_uuid": f"s{i}",
                "project_system_id": 10,
                "procedure_id": 5,
                "recorded_at": "2026-04-15T08:39:00Z",
                "lon": 13.7,
                "lat": 0.93,
                "values": {"uuid-tax": f"tree-{i}"},
            }

        observations = [_obs(i) for i in range(1200)]
        posted_sizes: list[int] = []

        def _fake_post(url, json=None, timeout=None):  # noqa: A002
            batch = json["observations"]
            posted_sizes.append(len(batch))
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = [
                {"survey_uuid": o["survey_uuid"], "status": "success"} for o in batch
            ]
            return mock

        with patch("naturecubepy.phone_observations.httpx.post", side_effect=_fake_post):
            out = upload_observations(hdr, observations, batch_size=500)

        assert posted_sizes == [500, 500, 200]
        assert len(out) == 1200
        assert out[0]["survey_uuid"] == "s0"
        assert out[-1]["survey_uuid"] == "s1199"

    def test_validate_csv_wide_missing_coordinate_columns(self, tmp_path: Path):
        """Missing lon/lat must be reported, not raise from an all-scalar frame."""
        from naturecubepy.phone_observations import (
            get_procedure,
            validate_csv_against_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        csv_path = tmp_path / "wide.csv"
        csv_path.write_text(
            "recorded_at,Taxonomic label,Nom commun\n"
            "15/04/2026 08:39,Myrianthus,Oboba\n",
            encoding="utf-8",
        )

        result = validate_csv_against_procedure(procedure, csv_path)

        assert result["format"] == "wide"
        assert result["valid"] is False
        assert any("Missing metadata columns" in issue for issue in result["issues"])
        assert len(result["feature_groups"]) == 1

    def test_validate_csv_wide_resolves_column_aliases(self, tmp_path: Path):
        """Headers like timestamp/Lat/Lon (any casing) map onto the standard columns."""
        from naturecubepy.phone_observations import (
            get_procedure,
            validate_csv_against_procedure,
        )

        procedure = get_procedure(
            _sample_schema(),
            system_name="Plante Ivindo",
            procedure_name="Arbre",
        )
        csv_path = tmp_path / "wide_aliases.csv"
        csv_path.write_text(
            "timestamp,Latitude,Lon,Taxonomic label,Nom commun\n"
            "15/04/2026 08:39,0.93,13.7,Myrianthus,Oboba\n",
            encoding="utf-8",
        )

        result = validate_csv_against_procedure(procedure, csv_path)

        assert result["format"] == "wide"
        assert result["valid"] is True
        assert float(result["feature_groups"]["longitude"].iloc[0]) == 13.7
        assert float(result["feature_groups"]["latitude"].iloc[0]) == 0.93
