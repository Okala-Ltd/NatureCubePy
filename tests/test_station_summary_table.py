"""Sensor summary record counts should match labelled observations."""

from __future__ import annotations

import pandas as pd

from naturecubepy.analysis import ObservationBundle, save_all_tables, station_summary_table


def _stations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "device_id": "cam1",
                "measurement_type": "Camera",
                "record_count": 100,
                "latitude": 1.0,
                "longitude": 2.0,
                "project_system_record_start_timestamp": "2025-01-01",
                "project_system_record_end_timestamp": "2025-01-11",
            },
            {
                "device_id": "bio1",
                "measurement_type": "Bioacoustic",
                "record_count": 200,
                "latitude": 1.1,
                "longitude": 2.1,
                "project_system_record_start_timestamp": "2025-01-01",
                "project_system_record_end_timestamp": "2025-01-06",
            },
            {
                "device_id": "edna1",
                "measurement_type": "eDNA",
                "record_count": 50,
                "latitude": 1.2,
                "longitude": 2.2,
                "project_system_record_start_timestamp": "2025-01-01",
                "project_system_record_end_timestamp": "2025-01-01",
            },
        ]
    )


def test_station_summary_falls_back_to_station_record_count():
    summary = station_summary_table(_stations())
    by_type = {
        row["Sensor Type"]: int(row["Number of Records"])
        for _, row in summary.iterrows()
    }
    assert by_type["Camera Traps"] == 100
    assert by_type["Bioacoustic Sensors"] == 200
    assert by_type["Environmental DNA"] == 50


def test_station_summary_prefers_observation_counts():
    summary = station_summary_table(
        _stations(),
        observation_counts={"camera": 12, "bioacoustic": 34, "edna": 5},
    )
    by_type = {
        row["Sensor Type"]: int(row["Number of Records"])
        for _, row in summary.iterrows()
    }
    assert by_type["Camera Traps"] == 12
    assert by_type["Bioacoustic Sensors"] == 34
    assert by_type["Environmental DNA"] == 5


def test_save_all_tables_uses_observation_lengths(tmp_path):
    bundle = ObservationBundle(
        camera=pd.DataFrame([{"species": "A"}, {"species": "B"}]),
        bioacoustic=pd.DataFrame([{"species": "C"}]),
        edna=pd.DataFrame([{"species": "D"}, {"species": "E"}, {"species": "F"}]),
        all_species=pd.DataFrame(
            [
                {"species": "A", "common_name": "a", "iucn_redlist_status": "Least Concern"},
                {"species": "C", "common_name": "c", "iucn_redlist_status": "Least Concern"},
                {"species": "D", "common_name": "d", "iucn_redlist_status": "Least Concern"},
            ]
        ),
        stations=_stations(),
        sensor_types=("camera", "bioacoustic", "edna"),
    )
    saved = save_all_tables(bundle, tmp_path)
    summary = pd.read_csv(saved["sensor_summary"])
    by_type = {
        row["Sensor Type"]: int(row["Number of Records"])
        for _, row in summary.iterrows()
    }
    assert by_type["Camera Traps"] == 2
    assert by_type["Bioacoustic Sensors"] == 1
    assert by_type["Environmental DNA"] == 3
