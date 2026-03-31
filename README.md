# NatureCubePy

Python wrapper around the [Okala](https://okala.io) dashboard API services.

NatureCubePy mirrors the functionality of the [`okalaR`](https://github.com/Okala-Ltd/OkalaR)
R package, providing the same API wrapper capabilities as a Python library that can be
submitted to [PyPI](https://pypi.org).

Package management is handled by [uv](https://docs.astral.sh/uv/).

---

## Installation

### From PyPI (once published)

```bash
pip install naturecubepy
# or with uv:
uv add naturecubepy
```

### Development installation

```bash
git clone https://github.com/Okala-Ltd/OkalaR.git
cd OkalaR/NatureCubePy
uv sync
```

---

## Quick start

```python
import os
from naturecubepy import auth_headers, get_project, get_station_info, plot_stations

# Set your API key (or export OKALA_API_KEY in your shell)
os.environ["OKALA_API_KEY"] = "your_api_key_here"

# Authenticate
hdr = auth_headers("your_api_key_here")

# Check active project
get_project(hdr)
# Setting your active project as - My Ecology Project

# Get station metadata (video, audio, image, or eDNA)
stations = get_station_info(hdr, datatype="video")

# Plot stations on an interactive map
map_widget = plot_stations(stations)
map_widget.save("stations.html")
```

---

## API Reference

### Authentication

| Function | Description |
|---|---|
| `get_key()` | Read `OKALA_API_KEY` from environment |
| `auth_headers(api_key)` | Create auth context for production API |
| `auth_headers_dev(api_key)` | Create auth context for development API |

### Project & Stations

| Function | Description |
|---|---|
| `get_project(hdr)` | Display the active project name |
| `get_station_info(hdr, datatype)` | Return a `GeoDataFrame` of station metadata |
| `plot_stations(gdf)` | Return a `folium.Map` of station locations |

### Media Assets

| Function | Description |
|---|---|
| `get_media_assets(hdr, datatype, psr_id)` | Retrieve media for a project system record |
| `update_media_timestamps(hdr, media_records)` | Update timestamps directly |
| `push_new_timestamps(hdr, media_metadata, chunksize)` | Update timestamps in chunks |

### Labels

| Function | Description |
|---|---|
| `get_project_labels(hdr, labeltype)` | Get project labels (`"Bioacoustic"` or `"Camera"`) |
| `add_project_labels(hdr, labeltype, labels)` | Add labels to project |
| `get_iucn_labels(hdr, offset, limit, search_term)` | Search IUCN species database |
| `add_iucn_labels(hdr, labels, chunksize)` | Add IUCN labels in chunks |
| `push_new_labels(hdr, submission_records, chunksize)` | Push segment label updates |
| `set_segment_blank_status(hdr, blank_status, segment_record_ids)` | Mark/unmark segments as blank |

### eDNA

| Function | Description |
|---|---|
| `check_edna_labels(hdr, edna_data)` | Validate eDNA records against the database |
| `upload_edna_records(hdr, validated_data, project_system_record_id)` | Upload validated eDNA |

### Phone Observations

| Function | Description |
|---|---|
| `build_device_settings(...)` | Build device metadata for upload |
| `build_observation(...)` | Build a single GeoJSON observation |
| `build_feature_record(...)` | Build a feature (collection of observations) |
| `validate_observation_payload(...)` | Validate payload before upload |
| `upload_phone_observations(...)` | Upload observations (with optional media) |

---

## Phone Observations Example

```python
from datetime import datetime, timezone
from naturecubepy import (
    auth_headers,
    build_device_settings,
    build_observation,
    build_feature_record,
    upload_phone_observations,
)

hdr = auth_headers("your_api_key")

device = build_device_settings(
    device_id="device-abc123",
    phone_model="iPhone 14 Pro",
    phone_os="iOS 17.2",
    carrier="Vodafone",
    build_number="1.2.3",
    build_id="build-456",
)

obs = build_observation(
    item_uuid="f47ac10b-58cc-4372-a567-0e02b2c3d479",
    item_type="text",
    data=["Observed 3 red kites circling"],
    geometry={"type": "Point", "coordinates": [-1.5, 53.4]},
)

feature = build_feature_record(
    feature_uuid="feature-uuid-123",
    project_system_id=42,
    procedure_id=7,
    start_time=datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    created_by_method="drawn",
    geometry={"type": "Point", "coordinates": [-1.5, 53.4]},
    observations=[obs],
)

result = upload_phone_observations(
    hdr=hdr,
    project_id=42,
    feature_payload=[feature],
    device_settings=device,
)
print(result["summary"])
```

---

## Development

### Running tests

```bash
uv run pytest
```

### Building the package

```bash
uv build
```

### Publishing to PyPI

```bash
uv publish
```

---

## License

Apache License 2.0 — see [LICENSE](../LICENSE) for details.

## Support

For issues and questions, please open a GitHub issue or contact [adam@okala.io](mailto:adam@okala.io).
