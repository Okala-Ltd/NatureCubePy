"""Tests for naturecubepy.api module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from naturecubepy.api import (
    add_iucn_labels,
    add_project_labels,
    auth_headers,
    auth_headers_dev,
    check_edna_labels,
    get_iucn_labels,
    get_key,
    get_media_assets,
    get_project,
    get_project_labels,
    get_station_info,
    push_new_labels,
    push_new_timestamps,
    set_segment_blank_status,
    update_media_timestamps,
    upload_edna_records,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def hdr():
    return auth_headers("test-api-key")


@pytest.fixture()
def hdr_dev():
    return auth_headers_dev("test-api-key")


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
        assert hdr["key"] == "mykey"
        assert hdr["root"] == "https://api.dashboard.okala.io/api/"

    def test_custom_url(self):
        hdr = auth_headers("mykey", okala_url="https://custom.example.com/api/")
        assert hdr["root"] == "https://custom.example.com/api/"

    def test_trailing_slash_normalised(self):
        hdr = auth_headers("mykey", okala_url="https://api.dashboard.okala.io/api")
        assert hdr["root"].endswith("/")

    def test_dev_default_url(self):
        hdr = auth_headers_dev("mykey")
        assert hdr["root"] == "https://dev.api.dashboard.okala.io/api/"


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
        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            get_project(hdr)
        captured = capsys.readouterr()
        assert "My Project" in captured.out


# ---------------------------------------------------------------------------
# get_station_info
# ---------------------------------------------------------------------------

class TestGetStationInfo:
    def test_returns_geodataframe(self, hdr):
        geojson_text = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-1.5, 53.4]},
                "properties": {"device_id": "dev1", "record_count": 5}
            }]
        }"""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = geojson_text
        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_station_info(hdr, "video")
        import geopandas as gpd
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# get_media_assets
# ---------------------------------------------------------------------------

class TestGetMediaAssets:
    def test_returns_dataframe(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"media_file_record_id": 1, "filename": "vid1.mp4"},
            {"media_file_record_id": 2, "filename": "vid2.mp4"},
        ]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response):
            result = get_media_assets(hdr, "video", psr_id=123)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# get_project_labels
# ---------------------------------------------------------------------------

class TestGetProjectLabels:
    def test_returns_dataframe(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"label_id": 1, "label_name": "Robin"},
        ]
        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_project_labels(hdr, "Bioacoustic")
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["label_name"] == "Robin"


# ---------------------------------------------------------------------------
# add_project_labels
# ---------------------------------------------------------------------------

class TestAddProjectLabels:
    def test_posts_and_returns_response(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        with patch("naturecubepy.api.httpx.post", return_value=mock_response):
            result = add_project_labels(hdr, "Camera", labels=[{"label_name": "Fox"}])
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# get_iucn_labels
# ---------------------------------------------------------------------------

class TestGetIucnLabels:
    def test_returns_dict_with_data(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "table": [{"label_id": 1, "scientific_name": "Panthera leo"}],
            "pagination_state": {"total": 1, "offset": 0, "limit": 10},
        }
        with patch("naturecubepy.api.httpx.get", return_value=mock_response):
            result = get_iucn_labels(hdr, offset=0, limit=10)
        assert isinstance(result["data"], pd.DataFrame)
        assert result["total"] == 1

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
    def _valid_df(self):
        return pd.DataFrame({
            "media_file_record_id": [123, 456],
            "new_timestamp": ["2024-01-15T10:30:00Z", "2024-01-15T14:20:00Z"],
        })

    def test_returns_response(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"updated": 2}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            result = update_media_timestamps(hdr, self._valid_df())
        assert result == {"updated": 2}

    def test_raises_missing_columns(self, hdr):
        df = pd.DataFrame({"media_file_record_id": [1]})
        with pytest.raises(ValueError, match="new_timestamp"):
            update_media_timestamps(hdr, df)

    def test_raises_non_numeric_id(self, hdr):
        df = pd.DataFrame({
            "media_file_record_id": ["abc"],
            "new_timestamp": ["2024-01-15T10:30:00Z"],
        })
        with pytest.raises(ValueError, match="numeric"):
            update_media_timestamps(hdr, df)

    def test_raises_negative_id(self, hdr):
        df = pd.DataFrame({
            "media_file_record_id": [-1],
            "new_timestamp": ["2024-01-15T10:30:00Z"],
        })
        with pytest.raises(ValueError, match="positive"):
            update_media_timestamps(hdr, df)

    def test_raises_invalid_timestamp_format(self, hdr):
        df = pd.DataFrame({
            "media_file_record_id": [1],
            "new_timestamp": ["not-a-timestamp"],
        })
        with pytest.raises(ValueError, match="ISO 8601"):
            update_media_timestamps(hdr, df)

    def test_accepts_timezone_offset(self, hdr):
        df = pd.DataFrame({
            "media_file_record_id": [1],
            "new_timestamp": ["2024-01-15T10:30:00+01:00"],
        })
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"updated": 1}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            result = update_media_timestamps(hdr, df)
        assert result == {"updated": 1}


# ---------------------------------------------------------------------------
# push_new_timestamps
# ---------------------------------------------------------------------------

class TestPushNewTimestamps:
    def test_splits_into_chunks(self, hdr, capsys):
        df = pd.DataFrame({
            "media_file_record_id": list(range(1, 11)),
            "new_timestamp": ["2024-01-15T10:30:00Z"] * 10,
        })
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response) as mock_put:
            push_new_timestamps(hdr, df, chunksize=5)
        assert mock_put.call_count == 2

    def test_adjusts_chunksize_if_too_large(self, hdr, capsys):
        df = pd.DataFrame({
            "media_file_record_id": [1, 2],
            "new_timestamp": ["2024-01-15T10:30:00Z", "2024-01-15T10:30:00Z"],
        })
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {}
        with patch("naturecubepy.api.httpx.put", return_value=mock_response):
            push_new_timestamps(hdr, df, chunksize=100)
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
            result = check_edna_labels(hdr, self._valid_df())
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["status"] == "success"

    def test_raises_missing_required_columns(self, hdr):
        df = pd.DataFrame({"marker_name": ["COI"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            check_edna_labels(hdr, df)

    def test_adds_default_confidence(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{"status": "success"}]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            check_edna_labels(hdr, self._valid_df())
        payload = mock_post.call_args.kwargs["json"]
        assert payload[0]["confidence"] == 100

    def test_handles_class_underscore(self, hdr):
        df = self._valid_df()
        df["class_"] = "Mammalia"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{"status": "success"}]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            check_edna_labels(hdr, df)
        payload = mock_post.call_args.kwargs["json"]
        assert payload[0].get("class") == "Mammalia"


# ---------------------------------------------------------------------------
# upload_edna_records
# ---------------------------------------------------------------------------

class TestUploadEdnaRecords:
    def _validated_df(self):
        return pd.DataFrame({
            "marker_name": ["COI", "COI"],
            "sequence": ["ACGT", "TTTT"],
            "primer": ["mlCOIintF", "mlCOIintF"],
            "timestamp": ["2024-01-15T10:30:00", "2024-01-15T11:00:00"],
            "status": ["success", "error"],
            "label": ["Panthera leo", None],
            "label_id": [1, None],
            "message": ["ok", "no match"],
        })

    def test_uploads_only_successful(self, hdr):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [{"status": "uploaded"}]
        with patch("naturecubepy.api.httpx.post", return_value=mock_response) as mock_post:
            result = upload_edna_records(hdr, self._validated_df(), project_system_record_id=99)
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload) == 1

    def test_raises_if_not_validated(self, hdr):
        df = pd.DataFrame({"marker_name": ["COI"]})
        with pytest.raises(ValueError, match="validated first"):
            upload_edna_records(hdr, df, project_system_record_id=1)

    def test_raises_if_no_successful_records(self, hdr):
        df = pd.DataFrame({
            "marker_name": ["COI"],
            "status": ["error"],
        })
        with pytest.raises(ValueError, match="No successful records"):
            upload_edna_records(hdr, df, project_system_record_id=1)
