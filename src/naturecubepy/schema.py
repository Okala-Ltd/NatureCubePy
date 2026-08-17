"""
Pydantic schemas for NatureCubePy API requests and responses.

These schemas provide type-safe validation for data exchanged with the Okala API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_geojson import (
    FeatureCollectionModel,
    FeatureModel,
    LineStringModel,
    MultiPolygonModel,
    PointModel,
    PolygonModel,
)


# ============================================================
# Type Aliases / Literals
# ============================================================

DataTypes = Literal["video", "image", "audio", "eDNA"]
MobileDataTypes = Literal["phone-photo", "phone-video", "phone-audio"]
LabelType = Literal["Camera", "Bioacoustic", "Observation"]
MeasurementType = Literal["Camera", "Bioacoustic", "eDNA", "Observation"]
ModelClassification = Literal["Classification", "Detection"]
LabelStatuses = Literal["ai_derived", "verified", "modified"]
SegmentVerificationStatuses = Literal[
    "labeller_verified", "manager_verified", "ai_derived"
]

IUCNStatus = Literal[
    "Data Deficient",
    "Least Concern",
    "Near Threatened",
    "Vulnerable",
    "Endangered",
    "Critically Endangered",
    "Extinct in the Wild",
    "Extinct",
    "Not Evaluated",
]

FileStatus = Literal["processed", "healthy", "corrupted", "quality_flagged"]

SystemTypes = Literal["Sensor", "Sample", "Observation"]
PhoneTypes = Literal[
    "phone-photo",
    "phone-video",
    "phone-audio",
    "choice",
    "text",
    "numeric",
    "label",
    "instruction",
]
ProcedureType = Literal[
    "Sensor installation",
    "Sensor maintenance",
    "Sensor removal",
    "Field observation",
    "Field sample",
]

ProjectStatus = Literal[
    "proposed",
    "field-planning",
    "pre-deployment",
    "under-deployment",
    "active",
    "on-hold",
    "closed",
]


# ============================================================
# Observation DataFrame column contracts
# ============================================================

SPECIES_OBS_CORE_COLUMNS: list[str] = [
    "project_system_record_id",
    "device_id",
    "data_type",
    "measurement_type",
    "latitude",
    "longitude",
    "label",
    "label_id",
    "common_name",
    "species",
    "class",
    "genus",
    "family",
    "order",
]

STATION_LOOKUP_COLUMNS: list[str] = [
    "project_system_record_id",
    "device_id",
    "data_type",
    "measurement_type",
    "latitude",
    "longitude",
]


# ============================================================
# Authentication Schema
# ============================================================


class AuthHeaders(BaseModel):
    """Authentication context for API requests."""

    key: str = Field(..., description="API key for authentication")
    root: str = Field(..., description="Base URL for the API")


# ============================================================
# Pagination & Base Schemas
# ============================================================


class PaginationState(BaseModel):
    """Pagination state for paginated responses."""

    offset: int
    limit: int
    total: int


class SpeciesMinimal(BaseModel):
    """Minimal species information."""

    label_id: int
    label: str
    common_name: str | None = None


class SpeciesLight(SpeciesMinimal):
    """Light species information with taxonomy."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    class_: str | None = Field(default=None, alias="class")
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    species: str | None = None
    iucn_redlist_status: IUCNStatus | None = None
    tags: list[str] = Field(default_factory=list)
    global_labels_applied: bool | None = True

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(tag) for tag in v if tag is not None]
        return []


class SpeciesTable(BaseModel):
    """Paginated species table response."""

    table: list[SpeciesLight]
    pagination_state: PaginationState


# ============================================================
# Label Schemas
# ============================================================


class AddLabel(BaseModel):
    """Schema for adding a label."""

    segment_record_id_fk: int | None = None
    label_id_fk: int | None = None


class LabelDB(AddLabel):
    """Label database schema."""

    label_record_uuid: UUID | str | None = Field(default_factory=lambda: str(uuid4()))
    number_of_individuals: int | None = 1
    prediction_accuracy: float | None = 100
    segment_record_published: bool | None = None
    label_created_at: datetime | None = Field(default_factory=datetime.now)


class Label(LabelDB):
    """Full label schema."""

    label_record_id: int | None = None
    segment_record_location_slug: str | None = None
    label_record_uuid: UUID | str | None = Field(default_factory=lambda: uuid4().hex)


# ============================================================
# Media & Segment Schemas
# ============================================================


class SegmentSimple(BaseModel):
    """Simple segment with timestamps."""

    segment_start_timestamp: datetime
    segment_end_timestamp: datetime


class MediaRecordSimple(BaseModel):
    """Simple media record."""

    media_file_record_id: int | None = None
    media_file_record_uuid: UUID | str | None = Field(
        default_factory=lambda: uuid4().hex
    )
    media_file_record_created_at: datetime | None = Field(default_factory=datetime.now)
    media_file_reference_location: str
    media_file_created_at: datetime
    project_system_record_id_fk: int | None = None
    duration_in_seconds: float | None = 0
    file_size: float | None = 0


class MediaRecordAPIFlat(MediaRecordSimple, SegmentSimple, SpeciesLight):
    """Flat media record with all details for API responses.

    Label fields are nullable because unlabelled and blank segments are
    returned with no label attached.
    """

    label_id: int | None = None
    label: str | None = None
    number_of_individuals: int | None = None
    segment_record_id: int | None = None
    label_record_id: int | None = None
    prediction_accuracy: float | None = None
    manager_verified: bool | None = None
    labeller_verified: bool | None = None
    blank: bool | None = False
    segment_verification_status: SegmentVerificationStatuses = "ai_derived"

    @model_validator(mode="before")
    @classmethod
    def check_segment_verification_status(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            if values.get("manager_verified"):
                values["segment_verification_status"] = "manager_verified"
            elif values.get("labeller_verified") and not values.get("manager_verified"):
                values["segment_verification_status"] = "labeller_verified"
            else:
                values["segment_verification_status"] = "ai_derived"
        return values


class SegmentRecordAPIFlat(MediaRecordSimple, SegmentSimple):
    """Flat segment record for API responses."""

    segment_record_id: int | None = None
    prediction_accuracy: float | None = None
    manager_verified: bool | None = None
    labeller_verified: bool | None = None
    blank: bool | None = False
    segment_verification_status: SegmentVerificationStatuses = "ai_derived"

    @model_validator(mode="before")
    @classmethod
    def check_segment_verification_status(cls, values: dict[str, Any]) -> dict[str, Any]:
        if isinstance(values, dict):
            if values.get("manager_verified"):
                values["segment_verification_status"] = "manager_verified"
            elif values.get("labeller_verified") and not values.get("manager_verified"):
                values["segment_verification_status"] = "labeller_verified"
            else:
                values["segment_verification_status"] = "ai_derived"
        return values


class CameraTrapDataRecord(BaseModel):
    """Merged camera trap row with validated core fields."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    project_system_record_id: int
    data_type: Literal["image", "video"]
    device_id: str | None = None
    measurement_type: MeasurementType | None = None
    latitude: float | None = None
    longitude: float | None = None


# ============================================================
# Timestamp Update Schemas
# ============================================================


class MediaTimestampUpdate(BaseModel):
    """Schema for updating media file timestamp."""

    media_file_record_id: int
    new_timestamp: datetime


class TimestampUpdateResponse(BaseModel):
    """Response schema for timestamp update operation."""

    media_file_record_id: int
    media_updated: bool
    segments_updated: int
    message: str | None = None


# ============================================================
# eDNA Schemas
# ============================================================


class MarkerRecord(BaseModel):
    """Marker record for eDNA data."""

    marker_record_id: int | None = None
    marker_reference_name: str
    genetic_sequence: str
    marker_primer: str
    label_id_fk: int | None = None


class eDNATable(SpeciesLight, MarkerRecord):
    """eDNA table with species and marker information."""

    iucn_redlist_status: IUCNStatus


class eDNAUploadSchema(BaseModel):
    """Schema for uploading eDNA records."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    marker_reference_name: str = Field(alias="marker_name")
    genetic_sequence: str = Field(alias="sequence")
    marker_primer: str = Field(alias="primer")
    edna_record_timestamp: datetime = Field(alias="timestamp")
    kingdom: str | None = None
    phylum: str | None = None
    class_: str | None = Field(default=None, alias="class")
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    species: str | None = None
    confidence: float | None = 100


class eDNAUploadResponse(eDNAUploadSchema):
    """Response schema for eDNA upload with validation status."""

    label: str | None = None
    label_id: int | None = -1
    status: Literal["success", "error"] | None = "error"
    message: str | None = None

    @model_validator(mode="after")
    def check_label(self) -> "eDNAUploadResponse":
        if self.label is None:
            for field in [
                "species",
                "genus",
                "family",
                "order",
                "class_",
                "phylum",
                "kingdom",
            ]:
                value = getattr(self, field, None)
                if value is not None:
                    self.label = value
                    break
        return self


# ============================================================
# IUCN Species Label Schemas
# ============================================================


class IUCNSpeciesLabelEdit(BaseModel):
    """Schema for editing IUCN species labels."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    kingdom: str | None = None
    phylum: str | None = None
    class_: str | None = Field(default=None, alias="class")
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    species: str | None = None
    label: str | None = None
    common_name: str | None = None
    iucn_redlist_status: IUCNStatus
    date_assessed: datetime | None = Field(default_factory=datetime.now)
    year_published: int | None = Field(default_factory=lambda: datetime.now().year)
    extant_country_list: list[str] | None = Field(default_factory=lambda: ["global"])
    species_thumbnail: str | None = None
    species_url: str | None = "https://www.wikipedia.org/wiki/"
    auxillary_country_data: list[str] | None = None

    @model_validator(mode="after")
    def check_label(self) -> "IUCNSpeciesLabelEdit":
        if self.label is None:
            for field in [
                "species",
                "genus",
                "family",
                "order",
                "class_",
                "phylum",
                "kingdom",
            ]:
                value = getattr(self, field, None)
                if value is not None:
                    self.label = value
        return self


class IUCNSpeciesLabelInput(IUCNSpeciesLabelEdit):
    """Input schema for adding IUCN species labels."""

    label_uuid: str | None = Field(default_factory=lambda: uuid4().hex)
    label_created_at: datetime = Field(default_factory=datetime.now)
    label_updated_at: datetime | None = Field(default_factory=datetime.now)
    taxonomic_uid: str | None = None

    @model_validator(mode="after")
    def generate_taxonomic_id(self) -> "IUCNSpeciesLabelInput":
        parts = [
            self.kingdom,
            self.phylum,
            self.class_,
            self.order,
            self.family,
            self.genus,
            self.species,
        ]
        self.taxonomic_uid = "-".join([p for p in parts if p is not None])
        return self


# ============================================================
# Station & System Schemas
# ============================================================


class StationPropertiesInit(BaseModel):
    """Initial station properties."""

    system_type: SystemTypes
    feature_id: int
    feature_name: str | None = ""
    system_name: str
    device_id: str
    project_system_record_id: int
    record_count: int

    @field_validator("feature_name", mode="before")
    @classmethod
    def set_feature_name(cls, v: str | None) -> str:
        return v if v is not None else ""


class StationPropertiesBase(StationPropertiesInit):
    """Base station properties with measurement info."""

    measurement_type: MeasurementType
    data_type: DataTypes


class StationPropertiesAPI(StationPropertiesBase):
    """Station properties for API responses."""

    project_system_record_start_timestamp: datetime
    project_system_record_end_timestamp: datetime


class PointGeometryStationAPI(FeatureModel):
    """Point geometry with station properties."""

    properties: StationPropertiesAPI


class StationResponseAPI(FeatureCollectionModel):
    """Station response as GeoJSON feature collection."""

    features: list[PointGeometryStationAPI]


# ============================================================
# Project Geometry Schemas
# ============================================================


class ProjectBaseCentroidProperties(BaseModel):
    """Base project centroid properties."""

    project_name: str
    project_description: str
    project_start_timestamp: datetime
    project_end_timestamp: datetime
    # Older API payloads may omit project_status.
    project_status: ProjectStatus | None = None
    project_colour: str | None = None
    project_ha: float | None = None
    project_published: bool | None = False
    organisation_activation_code: str | None = None
    project_open_licence: bool | None = False
    project_activation_code: str | None = None


class BaseProjectProperties(ProjectBaseCentroidProperties):
    """Base project properties with ID."""

    project_id: int | None = -1


class ProjectGeomProperties(BaseProjectProperties):
    """Project geometry properties."""

    project_created_at: datetime | None = Field(default_factory=datetime.now)
    monthly_spend_last_triggered: datetime | None = None
    project_thumbnail: str | None = None


class BaseAreaProperties(BaseModel):
    """Base area properties."""

    area_id: int | None = None
    area_name: str
    area_description: str | None = None
    area_colour: str | None = None
    area_ha: float | None = 0


class MultiPolygonGeometryProject(BaseModel):
    """Multi-polygon geometry with project properties."""

    type: str
    geometry: MultiPolygonModel | PolygonModel
    properties: ProjectGeomProperties


class MultiPolygonGeometryArea(BaseModel):
    """Multi-polygon geometry with area properties."""

    type: str
    geometry: MultiPolygonModel
    id: int | UUID | None = None
    properties: BaseAreaProperties


class ProjectSystemRecordResponse(BaseModel):
    """Project system record response."""

    system_uuid: str | None = None
    system_id: int | None = None
    system_type: SystemTypes
    measurement_type: MeasurementType | None = None
    system_name: str
    tags: list[str] | None = None
    project_system_id: int | None = None
    project_system_instance_id: int
    device_id: str | int
    project_system_record_id: int | None = None
    project_system_record_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    project_system_record_created_at: datetime | None = Field(
        default_factory=datetime.now
    )
    project_system_record_start_timestamp: datetime | None = None
    project_system_record_end_timestamp: datetime | None = None
    shared: bool | None = True
    feature_id: int | None = None
    feature_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    feature_created_at: datetime | None = Field(default_factory=datetime.now)
    feature_name: str | None = None
    feature_description: str | None = None
    feature_type: Literal["Point", "LineString", "MultiPolygon"] | None = None


class PointGeometryLocation(BaseModel):
    """Point geometry with location properties."""

    type: str
    geometry: PointModel
    id: int
    properties: ProjectSystemRecordResponse


class ProjectGeometryResponse(FeatureCollectionModel):
    """Project geometry response."""

    features: list[MultiPolygonGeometryProject]


class ProjectAreaGeometryResponse(FeatureCollectionModel):
    """Project area geometry response."""

    features: list[MultiPolygonGeometryArea]


class ProjectLocations(FeatureCollectionModel):
    """Project locations response."""

    features: list[PointGeometryLocation]


class GetProjectGeometryResponse(BaseModel):
    """Complete project geometry response."""

    boundary: ProjectGeometryResponse
    rois: ProjectAreaGeometryResponse
    locations: ProjectLocations


# ============================================================
# System Schemas
# ============================================================


class ItemInput(BaseModel):
    """Item input schema."""

    item_id: int | None = None
    item_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    item_name: str
    item_description: str
    nullable: bool | None = False
    data_type: PhoneTypes
    choices: list[str] | None = None

    @model_validator(mode="after")
    def test_choice(self) -> "ItemInput":
        if self.data_type == "choice" and not self.choices:
            raise ValueError("choice must contain a list of choices")
        return self


class Item(ItemInput):
    """Item schema with metadata."""

    item_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    item_created_at: datetime | None = Field(default_factory=datetime.now)


class ItemDB(Item):
    """Item database schema."""

    item_id: int


class ItemParameters(ItemInput):
    """Item with parameters."""

    range: list[float] | None = None


class ProcedureFlat(BaseModel):
    """Flat procedure schema."""

    procedure_id: int
    procedure_name: str
    procedure_type: ProcedureType
    items: list[ItemParameters]
    labels: list[SpeciesLight]
    procedure_description: str
    tags: list[str] | None = None
    shared: bool | None = True
    form: bool | None = False
    geometry_type: Literal["Point", "Lines", "Polygon"]


class ProcedureInput(BaseModel):
    """Procedure input schema."""

    procedure_name: str
    procedure_type: ProcedureType
    procedure_description: str
    tags: list[str] | None = None
    shared: bool | None = True
    form: bool | None = False
    geometry_type: Literal["Point", "Lines", "Polygon"]
    items: list[ItemInput]


class ProcedureBase(ProcedureInput):
    """Base procedure schema."""

    procedure_system_type: SystemTypes | None = None
    procedure_active: bool = False
    procedure_uuid: UUID
    procedure_created_at: datetime
    created_by: str
    editable: bool | None = False
    items: list[ItemDB]

    @model_validator(mode="after")
    def test_system(self) -> "ProcedureBase":
        if self.procedure_system_type is None:
            if self.procedure_type in [
                "Sensor installation",
                "Sensor maintenance",
                "Sensor removal",
            ]:
                self.procedure_system_type = "Sensor"
            elif self.procedure_type in ["Field observation"]:
                self.procedure_system_type = "Observation"
            elif self.procedure_type in ["Field sample"]:
                self.procedure_system_type = "Sample"
        return self


class Procedure(ProcedureBase):
    """Full procedure schema."""

    procedure_id: int
    organisation_id_fk: int


class SystemBase(BaseModel):
    """Base system schema."""

    system_name: str
    system_description: str
    tags: list[str]
    shared: bool | None = False
    system_type: SystemTypes
    system_thumbnail: str
    measurement_type: MeasurementType | None = None

    @model_validator(mode="after")
    def test_system(self) -> "SystemBase":
        if self.system_type == "Sensor":
            if self.measurement_type is None:
                raise ValueError("Measurement type must be defined")
            if self.measurement_type not in ["Camera", "Bioacoustic"]:
                raise ValueError("measurement_type must be Camera or Bioacoustic")
        elif self.system_type == "Sample":
            if self.measurement_type is None:
                raise ValueError("measurement_type must be defined")
            if self.measurement_type not in ["eDNA"]:
                raise ValueError("measurement_type must be eDNA")
        elif self.system_type == "Observation":
            self.measurement_type = "Observation"
        return self


class System(SystemBase):
    """Full system schema."""

    system_id: int
    system_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    system_updated_at: datetime | None = Field(default_factory=datetime.now)
    system_created_at: datetime | None = Field(default_factory=datetime.now)
    organisation_id_fk: int | None = None
    created_by: str | None = None
    active: bool | None = False
    editable: bool
    procedures: list[Procedure]


class ProjectSystemDaughters(System):
    """Project system with deployment info."""

    project_system_id: int | None = None
    deployed_count: int | None = 0
    project_id_fk: int


class SystemsResponse(BaseModel):
    """Systems response with catalogue and project systems."""

    catalogue: list[System]
    project: list[ProjectSystemDaughters]


# ============================================================
# Field Record Schemas
# ============================================================


class ObservationProperties(BaseModel):
    """Observation properties."""

    item_uuid: UUID
    item_type: PhoneTypes
    observation_uuid: UUID | None = Field(default_factory=lambda: uuid4().hex)
    observation_created_at: datetime | None = Field(default_factory=datetime.now)
    data: list[str | float | int]


class NestedObservationRecord(FeatureModel):
    """Nested observation record."""

    properties: ObservationProperties


class FieldRecordInit(BaseModel):
    """Field record initialization."""

    feature_uuid: UUID
    project_system_id: int
    procedure_id: int
    procedure_start_timestamp: datetime
    procedure_end_timestamp: datetime
    created_by_method: Literal["drawn", "traced"]
    geometry: PolygonModel | PointModel | LineStringModel | None = None


class ObservationRecord(BaseModel):
    """Observation record."""

    observations: list[NestedObservationRecord]


class FieldRecord(ObservationRecord, FieldRecordInit):
    """Full field record."""

    pass


class DeviceBase(BaseModel):
    """Base device schema."""

    model_config = ConfigDict(populate_by_name=True)

    device_unique_id: str = Field(alias="device_id")
    phone_model: str
    phone_operating_system: str
    device_created_at: datetime | None = Field(default_factory=datetime.now)
    device_last_used: datetime | None = None

    @field_validator("device_last_used", mode="before")
    @classmethod
    def set_time(cls, v: datetime | None) -> datetime:
        return v or datetime.now()


class DeviceInstanceProperties(BaseModel):
    """Device instance properties."""

    battery_level: float | None = 100
    carrier: str
    build_number: str
    build_id: str


class DeviceSettings(DeviceBase, DeviceInstanceProperties):
    """Device settings."""

    pass


class DeviceUpload(BaseModel):
    """Device upload payload."""

    feature_payload: list[FieldRecord]
    device_settings: DeviceSettings


class UploadRecordResponse(BaseModel):
    """Upload record response."""

    timestamp: datetime | None = Field(default_factory=datetime.now)
    message: str
    object: Literal["feature", "observation", "item", "device", "media"]
    uuid: UUID | str
    response_type: Literal["success", "warning", "error"]


class GlobalFeaturePolygonProperties(BaseModel):
    """Global feature polygon properties."""

    feature_id: int
    feature_uuid: UUID
    feature_name: str
    feature_description: str | None = None
    feature_class: str
    feature_type: str
