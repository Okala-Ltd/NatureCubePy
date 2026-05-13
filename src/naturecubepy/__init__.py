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
    auth_headers_dev,
    check_edna_labels,
    check_edna_labels_df,
    get_camera_trap_data,
    get_audio_data,
    get_audio_observation_data,
    get_iucn_labels,
    get_key,
    get_media_assets,
    get_media_assets_df,
    get_media_segments,
    get_project,
    get_species_observations,
    get_project_labels,
    get_station_info,
    get_stations_typed,
    plot_stations,
    push_new_labels,
    push_new_timestamps,
    set_segment_blank_status,
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
    upload_phone_observations,
    validate_observation_payload,
)
from naturecubepy.report_builder import (
    ReportBuildResult,
    build_core_report_text,
    generate_reports,
    render_final_docx,
)

__all__ = [
    # api
    "get_key",
    "auth_headers",
    "auth_headers_dev",
    "get_project",
    "get_species_observations",
    "get_station_info",
    "get_stations_typed",
    "plot_stations",
    "get_media_assets",
    "get_media_assets_df",
    "get_camera_trap_data",
    "get_audio_data",
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
    "check_edna_labels",
    "check_edna_labels_df",
    "upload_edna_records",
    # schema types
    "AuthHeaders",
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
    # report_builder
    "ReportBuildResult",
    "build_core_report_text",
    "generate_reports",
    "render_final_docx",
]
