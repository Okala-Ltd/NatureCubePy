"""Tests for naturecubepy.api module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import httpx
import naturecubepy.api as api_module
import pandas as pd
import pytest

from naturecubepy.api import (
    add_iucn_labels,
    add_project_labels,
    auth_headers,
    check_edna_labels,
    check_edna_labels_df,
    get_audio_observation_data,
    get_edna_assets,
    get_iucn_labels,
    get_key,
    get_media_assets,
    get_media_assets_df,
    get_media_segments,
    get_project,
    get_project_labels,
    get_species_observations,
    get_station_info,
    push_new_labels,
    push_new_timestamps,
    set_segment_blank_status,
    set_segment_published_status,
    update_media_timestamps,
    upload_edna_records,
)
from naturecubepy.schema import Label, SegmentRecordAPIFlat, MediaRecordAPIFlat, MediaTimestampUpdate, SpeciesLight, eDNAUploadResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hdr():
    return auth_headers("test-api-key")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestGetKey:
    def test_returns_key_when_set(self, monkeypatch):
        monkeypatch.setenv("OKALA_API_KEY", "abc123")
        assert get_key() == "abc123"

    def test_raises_when_not_set(self, monkeypatch):
        monkeypatch.delenv("OKALA_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OKALA_API_KEY"):
            get_key()


class TestAuthHeaders:
    def test_default_production_url(self):
        hdr = auth_headers("mykey")
        assert hdr.key == "mykey"
        assert hdr.root == "https://naturecube.io/api/"

    def test_root_url_defaults_to_api_prefix(self):
        hdr = auth_headers("mykey", okala_url="http://127.0.0.1:8000")
        assert hdr.root == "http://127.0.0.1:8000/api/"

    def test_custom_url(self):
        hdr = auth_headers("mykey", okala_url="https://custom.example.com/api/")
        assert hdr.root == "https://custom.example.com/api/"

    def test_trailing_slash_normalised(self):
        hdr = auth_headers("mykey", okala_url="https://naturecube.io/api")
        assert hdr.root.endswith("/")


# ---------------------------------------------------------------------------
# get_project
# ---------------------------------------------------------------------------

class TestGetProject:
    def test_prints_project_name(self, hdr, capsys):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "boundary": {
                "features": [{"properties": {"project_name": "My Project"}}]
            }
        }
        with (
            patch("naturecubepy.api.httpx.get", return_value=mock_response),
            patch("naturecubepy.api.GetProjectGeometryResponse.model_validate", return_value=MagicMock()),
        ):
            get_project(hdr)
        captured = capsys.readouterr()
        assert "My Project" in captured.out

    def test_accepts_payload_when_project_status_missing(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "boundary": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                        },
                        "properties": {
                            "project_name": "My Project",
                            "project_description": "desc",
                            "project_start_timestamp": "2024-01-01T00:00:00Z",
                            "project_end_timestamp": "2024-12-31T00:00:00Z",
                        },
                    }
                ],
            },
            "rois": {"type": "FeatureCollection", "features": []},
            "locations": {"type": "FeatureCollection", "features": []},
        }

        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_project(hdr)

        assert result.boundary.features[0].properties.project_status is None


# ---------------------------------------------------------------------------
# get_station_info
# ---------------------------------------------------------------------------

class TestGetStationInfo:
    def test_returns_geodataframe(self, hdr):
        image_payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-1.5, 53.4]},
                    "properties": {
                        "device_id": "dev-image",
                        "record_count": 5,
                        "measurement_type": "Camera",
                        "data_type": "image",
                    },
                }
            ],
        }
        video_payload = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-1.6, 53.5]},
                    "properties": {
                        "device_id": "dev-video",
                        "record_count": 7,
                        "measurement_type": "Camera",
                        "data_type": "video",
                    },
                }
            ],
        }

        image_response = MagicMock()
        image_response.raise_for_status = MagicMock()
        image_response.json.return_value = image_payload

        video_response = MagicMock()
        video_response.raise_for_status = MagicMock()
        video_response.json.return_value = video_payload

        with patch("naturecubepy.api.httpx.get", side_effect=[image_response, video_response]) as mock_get:
            result = get_station_info(hdr, "camera")

        import geopandas as gpd

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert set(result["data_type"]) == {"image", "video"}
        assert mock_get.call_count == 2

    def test_paginates_stations_for_single_datatype(self, hdr):
        def make_feature(idx: int) -> dict:
            return {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-1.5, 53.4]},
                "properties": {
                    "project_system_record_id": idx,
                    "device_id": f"dev-{idx}",
                    "record_count": 1,
                    "measurement_type": "Bioacoustic",
                    "data_type": "audio",
                },
            }

        first_page = {
            "type": "FeatureCollection",
            "features": [make_feature(i) for i in range(1000)],
        }
        second_page = {
            "type": "FeatureCollection",
            "features": [make_feature(1001)],
        }

        first_response = MagicMock()
        first_response.raise_for_status = MagicMock()
        first_response.json.return_value = first_page

        second_response = MagicMock()
        second_response.raise_for_status = MagicMock()
        second_response.json.return_value = second_page

        with patch("naturecubepy.api.httpx.get", side_effect=[first_response, second_response]) as mock_get:
            result = get_station_info(hdr, "bioacoustic")

        assert len(result) == 1001
        assert mock_get.call_count == 2



# ---------------------------------------------------------------------------
# get_species_observations
# ---------------------------------------------------------------------------

class TestGetSpeciesObservations:
    def test_combines_requested_domains(self, hdr):
        camera_df = pd.DataFrame([{"species": "Panthera leo", "data_type": "image", "project_system_record_id": 1}])
        edna_df = pd.DataFrame([{"species": "Canis lupus", "data_type": "eDNA", "project_system_record_id": 2, "class_": "Mammalia"}])

        with (
            patch("naturecubepy.api.get_camera_trap_data", return_value=camera_df),
            patch("naturecubepy.api.get_edna_observation_data", return_value=edna_df),
        ):
            result = get_species_observations(hdr, measurement_types=["camera", "edna"])

        assert len(result) == 2
        assert set(result["data_type"]) == {"image", "eDNA"}
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        assert "class" in result.columns

    def test_raises_for_invalid_measurement_type(self, hdr):
        with pytest.raises(ValueError, match="Invalid measurement_type"):
            get_species_observations(hdr, measurement_types=["invalid"])


# ---------------------------------------------------------------------------
# get_media_assets
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd


class TestGetMediaAssets:
    def test_returns_list(self, hdr):
        validated_records = [MagicMock(), MagicMock()]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"media_file_record_id": 1, "filename": "vid1.mp4"},
            {"media_file_record_id": 2, "filename": "vid2.mp4"},
        ]

        with (
            patch("naturecubepy.api.httpx.post", return_value=mock_response),
            patch(
                "naturecubepy.api.MediaRecordAPIFlat.model_validate",
                side_effect=validated_records,
            ),
        ):
            result = get_media_assets(hdr, "video", psr_ids=[123])

        assert isinstance(result, list)
        assert len(result) == 2


class TestMediaPagination:
    def _mock_paginated_endpoint(self):
        def fake_post(_url, json=None, params=None, timeout=None):
            response = MagicMock()
            response.raise_for_status = MagicMock()

            offset = (params or {}).get("offset", 0)
            limit = (params or {}).get("limit", 1000)

            if offset == 0:
                response.json.return_value = [{"id": i} for i in range(limit)]
            elif offset == limit:
                response.json.return_value = [{"id": i} for i in range(100)]
            else:
                response.json.return_value = []

            return response

        return fake_post

    def _mock_wrapped_paginated_endpoint(self):
        def fake_post(_url, json=None, params=None, timeout=None):
            response = MagicMock()
            response.raise_for_status = MagicMock()

            offset = (params or {}).get("offset", 0)
            limit = (params or {}).get("limit", 1000)

            if offset == 0:
                rows = [{"id": i} for i in range(limit)]
            elif offset == limit:
                rows = [{"id": i} for i in range(100)]
            else:
                rows = []

            response.json.return_value = {
                "rows": rows,
                "total": limit + 100,
            }

            return response

        return fake_post

    def test_get_media_segments_uses_offset_pagination(self, hdr):
        validated = [MagicMock() for _ in range(1100)]

        with (
            patch(
                "naturecubepy.api.httpx.post",
                side_effect=self._mock_paginated_endpoint(),
            ) as mock_post,
            patch(
                "naturecubepy.api.SegmentRecordAPIFlat.model_validate",
                side_effect=validated,
            ),
        ):
            result = get_media_segments(hdr, "video", psr_ids=[1, 2])

        assert isinstance(result, list)
        assert len(result) == 1100
        assert mock_post.call_count == 2

    def test_get_media_segments_supports_wrapped_pagination_payload(self, hdr):
        validated = [MagicMock() for _ in range(1100)]

        with (
            patch(
                "naturecubepy.api.httpx.post",
                side_effect=self._mock_wrapped_paginated_endpoint(),
            ) as mock_post,
            patch(
                "naturecubepy.api.SegmentRecordAPIFlat.model_validate",
                side_effect=validated,
            ),
        ):
            result = get_media_segments(hdr, "video", psr_ids=[1, 2])

        assert isinstance(result, list)
        assert len(result) == 1100
        assert mock_post.call_count == 2

    def test_get_media_assets_df_uses_offset_pagination(self, hdr):
        validated = [
            MagicMock(model_dump=lambda: {"id": i})
            for i in range(1100)
        ]

        with (
            patch(
                "naturecubepy.api.httpx.post",
                side_effect=self._mock_paginated_endpoint(),
            ) as mock_post,
            patch(
                "naturecubepy.api.MediaRecordAPIFlat.model_validate",
                side_effect=validated,
            ),
        ):
            result = get_media_assets_df(hdr, "video", psr_ids=[1, 2])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1100
        assert mock_post.call_count == 2

    def test_get_media_assets_df_supports_wrapped_pagination_payload(self, hdr):
        validated = [
            MagicMock(model_dump=lambda: {"id": i})
            for i in range(1100)
        ]

        with (
            patch(
                "naturecubepy.api.httpx.post",
                side_effect=self._mock_wrapped_paginated_endpoint(),
            ) as mock_post,
            patch(
                "naturecubepy.api.MediaRecordAPIFlat.model_validate",
                side_effect=validated,
            ),
        ):
            result = get_media_assets_df(hdr, "video", psr_ids=[1, 2])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1100
        assert mock_post.call_count == 2

    def test_timeout_splits_chunk_and_recovers(self, hdr):
        validated = [
            MagicMock(
                model_dump=lambda psr_id=psr_id: {
                    "project_system_record_id_fk": psr_id
                }
            )
            for psr_id in (101, 202)
        ]

        def fake_post(_url, json=None, params=None, timeout=None):
            if len(json) > 1:
                raise httpx.TimeoutException("request timed out")

            response = MagicMock()
            response.raise_for_status = MagicMock()
            response.json.return_value = [{"id": json[0]}]
            return response

        with (
            patch(
                "naturecubepy.api.httpx.post",
                side_effect=fake_post,
            ),
            patch(
                "naturecubepy.api.MediaRecordAPIFlat.model_validate",
                side_effect=validated,
            ),
        ):
            result = get_media_assets_df(
                hdr,
                "audio",
                psr_ids=[101, 202],
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result["project_system_record_id_fk"]) == {101, 202}


class TestSegmentRecordValidation:
    def _base_row(self):
        return {
            "segment_record_id": 1,
            "prediction_accuracy": 0.9,
            "manager_verified": False,
            "labeller_verified": False,

            # Add all required MediaRecordSimple fields
            "media_file_record_id": 1,
            "project_system_record_id_fk": 1,
            "media_file_reference_location": "s3://bucket/file.wav",
            "media_file_created_at": "2024-01-01T00:00:00Z",

            # Add all required SegmentSimple fields
            "segment_start_timestamp": "2024-01-01T00:00:00Z",
            "segment_end_timestamp": "2024-01-01T00:00:10Z",
        }

    def test_defaults_to_ai_derived(self):
        row = self._base_row()

        result = SegmentRecordAPIFlat.model_validate(row)

        assert result.segment_verification_status == "ai_derived"

    def test_labeller_verified_sets_status(self):
        row = self._base_row()
        row["labeller_verified"] = True

        result = SegmentRecordAPIFlat.model_validate(row)

        assert result.segment_verification_status == "labeller_verified"

    def test_manager_verified_takes_precedence(self):
        row = self._base_row()
        row["manager_verified"] = True
        row["labeller_verified"] = True

        result = SegmentRecordAPIFlat.model_validate(row)

        assert result.segment_verification_status == "manager_verified"


# ---------------------------------------------------------------------------
# get_edna_assets
# ---------------------------------------------------------------------------

class TestGetEdnaAssets:
    def test_supports_flat_list_payload(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"label": "Panthera leo", "species": "Panthera leo", "label_id": 1},
        ]

        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_edna_assets(hdr, project_system_record_id=123)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.iloc[0]["label"] == "Panthera leo"

    def test_supports_wrapped_table_payload(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "table": [
                {"label": "Canis lupus", "species": "Canis lupus", "label_id": 2},
            ],
            "sankey": [],
        }

        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_edna_assets(hdr, project_system_record_id=456)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.iloc[0]["label"] == "Canis lupus"


    def test_raises_after_retries_exhausted(self, hdr):
        rate_limited = MagicMock()
        rate_limited.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests",
            request=httpx.Request("GET", "https://example.invalid"),
            response=httpx.Response(429, headers={"Retry-After": "0"}),
        )

        with (
            patch("naturecubepy.api.httpx.get", return_value=rate_limited),
            patch("naturecubepy.api.time.sleep"),
            pytest.raises(httpx.HTTPStatusError),
        ):
            get_edna_assets(hdr, project_system_record_id=123)


class TestGetEdnaObservationData:
    def _edna_station_frame(self, psr_ids, record_counts):
        return gpd.GeoDataFrame(
            {
                "project_system_record_id": psr_ids,
                "device_id": [f"dev-{i}" for i in psr_ids],
                "measurement_type": ["eDNA"] * len(psr_ids),
                "data_type": ["eDNA"] * len(psr_ids),
                "record_count": record_counts,
            },
            geometry=gpd.points_from_xy([150.0 + i for i in range(len(psr_ids))], [-33.0] * len(psr_ids)),
            crs="EPSG:4326",
        )

    def test_skips_zero_record_stations(self, hdr):
        stations = self._edna_station_frame([1001, 1002], [2, 0])

        def fake_get_edna_assets(_hdr, project_system_record_id):
            assert project_system_record_id == 1001
            return pd.DataFrame([
                {"label": "Panthera leo", "species": "Panthera leo", "label_id": 1}
            ])

        with (
            patch("naturecubepy.api.get_station_info", return_value=stations),
            patch("naturecubepy.api.get_edna_assets", side_effect=fake_get_edna_assets),
        ):
            from naturecubepy.api import get_edna_observation_data

            result = get_edna_observation_data(hdr)

        assert not result.empty
        assert set(result["project_system_record_id"]) == {1001}

    def test_http_500_is_treated_as_no_data(self, hdr, capsys):
        stations = self._edna_station_frame([6731], [1])

        def raise_500(_hdr, project_system_record_id):
            request = httpx.Request("GET", f"https://example.invalid/{project_system_record_id}")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("Server error", request=request, response=response)

        with (
            patch("naturecubepy.api.get_station_info", return_value=stations),
            patch("naturecubepy.api.get_edna_assets", side_effect=raise_500),
        ):
            from naturecubepy.api import get_edna_observation_data

            result = get_edna_observation_data(hdr)

        captured = capsys.readouterr()
        assert "Warning: could not retrieve eDNA species labels" not in captured.out
        assert "project_system_record_id" in result.columns
        assert not result.empty


# ---------------------------------------------------------------------------
# get_project_labels
# ---------------------------------------------------------------------------

class TestGetProjectLabels:
    def test_returns_dataframe(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"label_id": 1, "label": "Robin"},
        ]
        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_project_labels(hdr, "Bioacoustic")
        assert isinstance(result, pd.DataFrame)
        assert result.loc[0, "label"] == "Robin"


# ---------------------------------------------------------------------------
# add_project_labels
# ---------------------------------------------------------------------------

class TestAddProjectLabels:
    def test_posts_and_returns_response(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        with patch("naturecubepy.api.httpx.post", return_value=mock_response):
            result = add_project_labels(hdr, "Camera", labels=[Label()])
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# get_iucn_labels
# ---------------------------------------------------------------------------

class TestGetIucnLabels:
    def test_returns_dict_with_data(self, hdr):
        validated_table = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "table": [{"label_id": 1, "label": "Panthera leo"}],
            "pagination_state": {"total": 1, "offset": 0, "limit": 10},
        }
        with (
            patch("naturecubepy.api.httpx.get", return_value=mock_response),
            patch("naturecubepy.api.SpeciesTable.model_validate", return_value=validated_table),
        ):
            result = get_iucn_labels(hdr, offset=0, limit=10)
        assert result is validated_table

    def test_raises_when_limit_too_large(self, hdr):
        with pytest.raises(ValueError, match="20000"):
            get_iucn_labels(hdr, offset=0, limit=20001)

    def test_search_term_passed(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "table": [],
            "pagination_state": {"total": 0, "offset": 0, "limit": 10},
        }
        with patch("naturecubepy.api.httpx.get", return_value=mock_response) as mock_get:
            get_iucn_labels(hdr, offset=0, limit=10, search_term="horse")
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["search_term"] == "horse"


# ---------------------------------------------------------------------------
# update_media_timestamps
# ---------------------------------------------------------------------------

class TestUpdateMediaTimestamps:
    def _valid_updates(self):
        return [
            MediaTimestampUpdate(media_file_record_id=123, new_timestamp="2024-01-15T10:30:00Z"),
            MediaTimestampUpdate(media_file_record_id=456, new_timestamp="2024-01-15T14:20:00Z"),
        ]

    def test_returns_response(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "media_file_record_id": 123,
                "media_updated": True,
                "segments_updated": 1,
                "message": "ok",
            },
            {
                "media_file_record_id": 456,
                "media_updated": True,
                "segments_updated": 2,
                "message": "ok",
            },
        ]
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            result = update_media_timestamps(hdr, self._valid_updates())
        assert len(result) == 2
        assert result[0].media_file_record_id == 123

    def test_raises_invalid_timestamp_format(self):
        with pytest.raises(ValueError):
            MediaTimestampUpdate(media_file_record_id=1, new_timestamp="not-a-timestamp")

    def test_accepts_timezone_offset(self, hdr):
        updates = [
            MediaTimestampUpdate(media_file_record_id=1, new_timestamp="2024-01-15T10:30:00+01:00"),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "media_file_record_id": 1,
                "media_updated": True,
                "segments_updated": 0,
                "message": "ok",
            }
        ]
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            result = update_media_timestamps(hdr, updates)
        assert result[0].media_file_record_id == 1


# ---------------------------------------------------------------------------
# push_new_timestamps
# ---------------------------------------------------------------------------

class TestPushNewTimestamps:
    def test_splits_into_chunks(self, hdr, capsys):
        updates = [
            MediaTimestampUpdate(
                media_file_record_id=i,
                new_timestamp="2024-01-15T10:30:00Z",
            )
            for i in range(1, 11)
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "media_file_record_id": 1,
                "media_updated": True,
                "segments_updated": 0,
                "message": "ok",
            }
        ]
        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            push_new_timestamps(hdr, updates, chunksize=5)
        assert mock_put.call_count == 2

    def test_adjusts_chunksize_if_too_large(self, hdr, capsys):
        updates = [
            MediaTimestampUpdate(media_file_record_id=1, new_timestamp="2024-01-15T10:30:00Z"),
            MediaTimestampUpdate(media_file_record_id=2, new_timestamp="2024-01-15T10:30:00Z"),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "media_file_record_id": 1,
                "media_updated": True,
                "segments_updated": 0,
                "message": "ok",
            }
        ]
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            push_new_timestamps(hdr, updates, chunksize=100)
        captured = capsys.readouterr()
        assert "altering chunksize" in captured.out


# ---------------------------------------------------------------------------
# set_segment_blank_status
# ---------------------------------------------------------------------------

class TestSetSegmentBlankStatus:
    def test_sends_true(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": "Updated"}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            result = set_segment_blank_status(hdr, True, [101, 102])
        url = mock_put.call_args.args[0]
        assert "true" in url
        assert result["message"] == "Updated"

    def test_sends_false(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": "Updated"}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            set_segment_blank_status(hdr, False, [101])
        url = mock_put.call_args.args[0]
        assert "false" in url


# ---------------------------------------------------------------------------
# set_segment_published_status
# ---------------------------------------------------------------------------

class TestSetSegmentPublishedStatus:
    def test_sends_false(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": "Publish status updated successfully"}

        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            result = set_segment_published_status(hdr, False, [101, 102])

        assert result["message"] == "Publish status updated successfully"
        url = mock_put.call_args.args[0]
        assert "segmentRecordsPublishStatus" in url
        assert "/False" in url
        json_body = mock_put.call_args.kwargs["json"]
        assert json_body == [101, 102]

    def test_sends_true(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": "Publish status updated successfully"}

        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            set_segment_published_status(hdr, True, [101])

        url = mock_put.call_args.args[0]
        assert "segmentRecordsPublishStatus" in url
        assert "/True" in url
        json_body = mock_put.call_args.kwargs["json"]
        assert json_body == [101]


# ---------------------------------------------------------------------------
# check_edna_labels
# ---------------------------------------------------------------------------

class TestCheckEdnaLabels:
    def _valid_df(self):
        return pd.DataFrame({
            "marker_name": ["COI"],
            "sequence": ["ACGT"],
            "primer": ["mlCOIintF"],
            "timestamp": ["2024-01-15T10:30:00"],
            "species": ["Panthera leo"],
        })

    def test_returns_dataframe(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"marker_name": "COI", "status": "success", "label": "Panthera leo", "label_id": 1, "message": "ok"},
        ]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response):
            result = check_edna_labels_df(hdr, self._valid_df())
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["status"] == "success"

    def test_raises_missing_required_columns(self, hdr):
        df = pd.DataFrame({"marker_name": ["COI"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            check_edna_labels_df(hdr, df)

    def test_adds_default_confidence(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{"status": "success"}]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            check_edna_labels_df(hdr, self._valid_df())
        payload = mock_post.call_args.kwargs["json"]
        assert payload[0]["confidence"] == 100

    def test_handles_class_underscore(self, hdr):
        df = self._valid_df()
        df["class_"] = "Mammalia"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{"status": "success"}]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            result = check_edna_labels_df(hdr, df)
        payload = mock_post.call_args.kwargs["json"]
        assert payload[0].get("class") == "Mammalia"
        assert "class" in result.columns
        assert "class_" not in result.columns


# ---------------------------------------------------------------------------
# upload_edna_records
# ---------------------------------------------------------------------------

class TestUploadEdnaRecords:
    def _validated_data(self):
        return [
            eDNAUploadResponse(
                marker_name="COI",
                sequence="ACGT",
                primer="mlCOIintF",
                timestamp="2024-01-15T10:30:00",
                status="success",
                label="Panthera leo",
                label_id=1,
                message="ok",
            ),
            eDNAUploadResponse(
                marker_name="COI",
                sequence="TTTT",
                primer="mlCOIintF",
                timestamp="2024-01-15T11:00:00",
                status="error",
                label=None,
                label_id=None,
                message="no match",
            ),
        ]

    def test_uploads_only_successful(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "marker_name": "COI",
                "sequence": "ACGT",
                "primer": "mlCOIintF",
                "timestamp": "2024-01-15T10:30:00",
                "status": "success",
                "label": "Panthera leo",
                "label_id": 1,
                "message": "uploaded",
            }
        ]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            result = upload_edna_records(hdr, self._validated_data(), project_system_record_id=99)
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload) == 1
        assert len(result) == 1

    def test_raises_if_not_validated(self, hdr):
        with pytest.raises(AttributeError):
            upload_edna_records(hdr, [{"marker_name": "COI"}], project_system_record_id=1)

    def test_raises_if_no_successful_records(self, hdr):
        df = [
            eDNAUploadResponse(
                marker_name="COI",
                sequence="ACGT",
                primer="mlCOIintF",
                timestamp="2024-01-15T10:30:00",
                status="error",
            )
        ]
        with pytest.raises(ValueError, match="No successful records"):
            upload_edna_records(hdr, df, project_system_record_id=1)


class TestTaxonomySerialization:
    def test_species_light_model_dump_uses_class_alias(self):
        species = SpeciesLight(label_id=1, label="Panthera leo", class_="Mammalia")
        dumped = species.model_dump()
        assert "class" in dumped
        assert "class_" not in dumped
