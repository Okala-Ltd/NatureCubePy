"""Species diversity and conservation measure calculations."""

from __future__ import annotations

from math import exp, log

import pandas as pd

from naturecubepy.analysis import (
    ObservationBundle,
    exponential_shannon_index,
    invasive_species_table,
    inverse_simpson_index,
    red_list_index,
    save_all_tables,
    species_measures_table,
)


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"species": "A", "common_name": "a", "iucn_redlist_status": "Least Concern", "tags": []},
            {"species": "A", "common_name": "a", "iucn_redlist_status": "Least Concern", "tags": []},
            {"species": "B", "common_name": "b", "iucn_redlist_status": "Vulnerable", "tags": ["Invasive"]},
            {"species": "C", "common_name": "c", "iucn_redlist_status": "Data Deficient", "tags": []},
            {"species": "D", "common_name": "d", "iucn_redlist_status": "Endangered", "tags": ["alien; invasive plant"]},
            {"species": "D", "common_name": "d", "iucn_redlist_status": "Endangered", "tags": ["alien; invasive plant"]},
            {"species": "D", "common_name": "d", "iucn_redlist_status": "Endangered", "tags": ["alien; invasive plant"]},
        ]
    )


def test_exponential_shannon_and_inverse_simpson_match_hill_numbers():
    counts = pd.Series({"A": 2, "B": 1, "C": 1, "D": 3})
    proportions = counts / counts.sum()
    expected_shannon = exp(float(-(proportions * proportions.map(log)).sum()))
    expected_simpson = float(1.0 / (proportions * proportions).sum())

    assert exponential_shannon_index(counts) == expected_shannon
    assert inverse_simpson_index(counts) == expected_simpson


def test_red_list_index_excludes_data_deficient():
    # Assessed: LC(0), VU(2), EN(3) → 1 - 5/(3*5) = 2/3
    statuses = ["Least Concern", "Vulnerable", "Data Deficient", "Endangered"]
    assert abs(red_list_index(statuses) - (2.0 / 3.0)) < 1e-9


def test_species_measures_table_values():
    table = species_measures_table(_obs())
    values = dict(zip(table["measure"], table["value"]))

    assert values["observed_richness"] == 4
    assert values["invasive_species_count"] == 2
    assert values["percent_invasive"] == 50.0
    assert values["invasive_species"] == "B, D"
    assert values["assessed_species_count"] == 3
    assert abs(float(values["red_list_index"]) - (2.0 / 3.0)) < 1e-6
    assert float(values["exponential_shannon"]) > 1
    assert float(values["inverse_simpson"]) > 1


def test_invasive_species_table_lists_tagged_species():
    table = invasive_species_table(_obs())
    assert list(table["species"]) == ["D", "B"]
    assert int(table.loc[table["species"] == "D", "observation_count"].iloc[0]) == 3


def test_invasive_tags_from_string_literal():
    df = pd.DataFrame(
        [
            {"species": "X", "common_name": "x", "iucn_redlist_status": "LC", "tags": "['Invasive species']"},
            {"species": "Y", "common_name": "y", "iucn_redlist_status": "LC", "tags": ""},
        ]
    )
    table = invasive_species_table(df)
    assert list(table["species"]) == ["X"]


def test_save_all_tables_writes_species_measures(tmp_path):
    bundle = ObservationBundle(
        camera=pd.DataFrame([{"species": "A"}]),
        bioacoustic=pd.DataFrame(),
        edna=None,
        all_species=_obs(),
        stations=pd.DataFrame(),
        sensor_types=("camera",),
    )
    saved = save_all_tables(bundle, tmp_path)
    assert "species_measures" in saved
    assert "invasive_species" in saved
    measures = pd.read_csv(saved["species_measures"])
    assert "exponential_shannon" in set(measures["measure"])
