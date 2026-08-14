"""
NatureCubePy – Python wrapper around the Okala dashboard API services.

This package mirrors the functionality of the OkalaR R package, providing
the same API wrapper capabilities as a Python library suitable for submission
to PyPI.

Quick start
-----------
>>> import os
>>> os.environ["OKALA_API_KEY"] = "your_api_key"
>>> from naturecubepy import auth_headers, get_project
>>> hdr = auth_headers("your_api_key")
>>> get_project(hdr)  # doctest: +SKIP
Setting your active project as - My Project
"""

from naturecubepy.api import (
    add_iucn_labels,
    add_project_labels,
    auth_headers,
    check_edna_labels,
    check_edna_labels_df,
    get_camera_trap_data,
    get_audio_observation_data,
    get_iucn_labels,
    get_key,
    get_media_assets,
    get_media_assets_df,
    get_media_segments,
    get_project,
    get_project_boundary,
    project_boundary_gdf,
    get_species_observations,
    get_project_labels,
    get_station_info,
    get_stations_typed,
    push_new_labels,
    push_new_timestamps,
    set_segment_blank_status,
    set_segment_published_status,
    update_media_timestamps,
    update_segment_labels,
    upload_edna_records,
    get_edna_observation_data
)
from naturecubepy.schema import (
    AuthHeaders,
    CameraTrapDataRecord,
    DataTypes,
    GetProjectGeometryResponse,
    IUCNSpeciesLabelInput,
    Label,
    LabelType,
    MediaRecordAPIFlat,
    MediaTimestampUpdate,
    SegmentRecordAPIFlat,
    SPECIES_OBS_CORE_COLUMNS,
    STATION_LOOKUP_COLUMNS,
    SpeciesLight,
    SpeciesTable,
    StationResponseAPI,
    TimestampUpdateResponse,
    eDNAUploadResponse,
    eDNAUploadSchema,
)
from naturecubepy.phone_observations import (
    PHONE_TYPES,
    build_device_settings,
    build_feature_record,
    build_observation,
    build_observation_record,
    build_upload_observations_from_table,
    get_procedure,
    get_project_systems,
    list_systems,
    summarise_upload_results,
    upload_observations,
    upload_observations_from_csv,
    upload_phone_observations,
    validate_csv_against_procedure,
    validate_observation_payload,
)
from naturecubepy.analysis import (
    ObservationBundle,
    ProjectAssetExport,
    export_project_assets,
    load_project_data,
    save_all_tables,
    save_observation_bundle,
)


__all__ = [
    # api
    "get_key",
    "auth_headers",
    "get_project",
    "get_project_boundary",
    "project_boundary_gdf",
    "get_species_observations",
    "get_station_info",
    "get_stations_typed",
    "get_media_assets",
    "get_media_assets_df",
    "get_camera_trap_data",
    "get_audio_observation_data",
    "get_media_segments",
    "get_project_labels",
    "add_project_labels",
    "get_iucn_labels",
    "add_iucn_labels",
    "push_new_labels",
    "update_media_timestamps",
    "push_new_timestamps",
    "update_segment_labels",
    "set_segment_blank_status",
    "set_segment_published_status",
    "check_edna_labels",
    "check_edna_labels_df",
    "upload_edna_records",
    "get_edna_observation_data",
    # schema types
    "AuthHeaders",
    "SPECIES_OBS_CORE_COLUMNS",
    "STATION_LOOKUP_COLUMNS",
    "CameraTrapDataRecord",
    "DataTypes",
    "GetProjectGeometryResponse",
    "IUCNSpeciesLabelInput",
    "Label",
    "LabelType",
    "MediaRecordAPIFlat",
    "MediaTimestampUpdate",
    "SegmentRecordAPIFlat",
    "SpeciesLight",
    "SpeciesTable",
    "StationResponseAPI",
    "TimestampUpdateResponse",
    "eDNAUploadResponse",
    "eDNAUploadSchema",
    # phone_observations
    "PHONE_TYPES",
    "build_device_settings",
    "build_observation",
    "build_feature_record",
    "validate_observation_payload",
    "upload_phone_observations",
    "get_project_systems",
    "list_systems",
    "get_procedure",
    "validate_csv_against_procedure",
    "summarise_upload_results",
    "build_observation_record",
    "build_upload_observations_from_table",
    "upload_observations",
    "upload_observations_from_csv",
    # analysis / asset export
    "ObservationBundle",
    "ProjectAssetExport",
    "load_project_data",
    "save_observation_bundle",
    "save_all_tables",
    "export_project_assets",
]
