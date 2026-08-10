from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import geopandas as gpd
import httpx
import pandas as pd
import pytest

from naturecubepy.api import auth_headers, get_camera_trap_data
from naturecubepy.schema import MediaRecordAPIFlat


def _station_frame(project_system_record_id: int, datatype: str, longitude: float, latitude: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "project_system_record_id": [project_system_record_id],
            "device_id": [f"device-{datatype}"],
            "measurement_type": ["Camera"],
            "data_type": [datatype],
        },
        geometry=gpd.points_from_xy([longitude], [latitude]),
        crs="EPSG:4326",
    )


def _media_record(project_system_record_id: int, segment_record_id: int) -> MediaRecordAPIFlat:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return MediaRecordAPIFlat(
        label_id=segment_record_id,
        label=f"label-{segment_record_id}",
        common_name=None,
        segment_start_timestamp=timestamp,
        segment_end_timestamp=timestamp,
        media_file_record_id=segment_record_id,
        media_file_reference_location=f"media-{segment_record_id}.jpg",
        media_file_created_at=timestamp,
        project_system_record_id_fk=project_system_record_id,
        duration_in_seconds=10.0,
        file_size=1.0,
        number_of_individuals=1,
        segment_record_id=segment_record_id,
        label_record_id=segment_record_id,
        prediction_accuracy=99.0,
        manager_verified=False,
        labeller_verified=False,
    )


@pytest.fixture()
def hdr():
    return auth_headers("test-api-key")


def test_get_camera_trap_data_returns_image_and_video_without_duplicate_columns(hdr):
    image_stations = _station_frame(101, "image", 12.1, -0.1)
    video_stations = _station_frame(202, "video", 12.2, -0.2)
    all_camera_stations = pd.concat([image_stations, video_stations], ignore_index=True)

    def fake_get_station_info(_hdr, measurement_type):
        assert measurement_type == "camera"
        return all_camera_stations

    def fake_get_media_assets_df(_hdr, datatype, project_system_record_ids=None):
        psr_ids = project_system_record_ids or []
        assert psr_ids
        return pd.DataFrame(
            [
                _media_record(int(psr_ids[0]), 1 if datatype == "image" else 2).model_dump(mode="json")
            ]
        )

    def fake_get_media_segments(_hdr, datatype, project_system_record_ids=None):
        return pd.DataFrame(
            [
                {
                    "segment_record_id": 1 if datatype == "image" else 2,
                    "prediction_accuracy": 88.0,
                    "manager_verified": False,
                    "labeller_verified": False,
                    "blank": False,
                    "segment_note": f"note-{datatype}",
                }
            ]
        )

    with (
        patch("naturecubepy.api.get_station_info", side_effect=fake_get_station_info),
        patch("naturecubepy.api.get_media_assets_df", side_effect=fake_get_media_assets_df),
        patch("naturecubepy.api.get_media_segments", side_effect=fake_get_media_segments),
    ):
        result = get_camera_trap_data(hdr)

    assert set(result["data_type"]) == {"image", "video"}
    assert "latitude" in result.columns
    assert "longitude" in result.columns
    assert not any(column.endswith("_x") or column.endswith("_y") for column in result.columns)
    assert result["segment_note"].tolist() == ["note-image", "note-video"]


def test_get_camera_trap_data_raises_when_station_ids_missing(hdr):
    stations = gpd.GeoDataFrame(
        {"device_id": ["device-image"]},
        geometry=gpd.points_from_xy([12.1], [-0.1]),
        crs="EPSG:4326",
    )

    with patch("naturecubepy.api.get_station_info", return_value=stations):
        with pytest.raises(ValueError, match="project_system_record_id"):
            get_camera_trap_data(hdr)


def test_get_camera_trap_data_keeps_partial_results_when_one_station_fails(hdr):
    image_stations = _station_frame(101, "image", 12.1, -0.1)
    video_stations = _station_frame(202, "video", 12.2, -0.2)
    all_camera_stations = pd.concat([image_stations, video_stations], ignore_index=True)

    def fake_get_station_info(_hdr, measurement_type):
        return all_camera_stations

    def fake_get_media_assets_df(_hdr, datatype, project_system_record_ids=None):
        psr_ids = project_system_record_ids or []
        psr_id = int(psr_ids[0])
        if psr_id == 202:
            raise httpx.HTTPStatusError(
                "429",
                request=MagicMock(),
                response=MagicMock(status_code=429, headers={}),
            )
        return pd.DataFrame([_media_record(psr_id, 1).model_dump(mode="json")])

    def fake_get_media_segments(_hdr, datatype, project_system_record_ids=None):
        return [{"segment_record_id": 1, "segment_note": "note-image"}]

    with (
        patch("naturecubepy.api.get_station_info", side_effect=fake_get_station_info),
        patch("naturecubepy.api.get_media_assets_df", side_effect=fake_get_media_assets_df),
        patch("naturecubepy.api.get_media_segments", side_effect=fake_get_media_segments),
    ):
        result = get_camera_trap_data(hdr)

    assert len(result) == 1
    assert result["data_type"].iloc[0] == "image"
