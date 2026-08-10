# NatureCubePy

Python client for the [Okala](https://okala.io) / NatureCube API.

Use it to authenticate, pull biodiversity observations (camera trap, bioacoustic, eDNA), update labels and media, plot results, and export project figures and tables. It is the Python counterpart to [`NatureCubeR`](https://github.com/Okala-Ltd/NatureCubeR).

Requires **Python 3.12+**. Package management uses [uv](https://docs.astral.sh/uv/).

---

## What you can do

| Area | Examples |
|------|----------|
| **Connect** | Authenticate with an API key and inspect the active project |
| **Retrieve** | Stations, media, segments, and species observations by sensor |
| **Update** | Labels, timestamps, blank/publish status, eDNA uploads |
| **Analyze & plot** | Station maps, species summaries, accumulation curves, IUCN charts |
| **Export** | Cache project data and save standard figures and summary tables |

---

## Package layout

| Module | Role |
|--------|------|
| `naturecubepy.api` | HTTP API wrappers (auth, stations, observations, labels, eDNA) |
| `naturecubepy.viz` | Maps, timelines, species and eDNA plots |
| `naturecubepy.analysis` | Load project data; summary tables; `export_project_assets` |
| `naturecubepy.phone_observations` | Build and upload phone-based field observations |
| `naturecubepy.schema` | Pydantic models shared across the package |

Most day-to-day helpers are also re-exported from `naturecubepy` directly.

---

## Installation

```bash
pip install naturecubepy
# or
uv add naturecubepy
```

### Development install

```bash
git clone https://github.com/Okala-Ltd/NatureCubePy.git
cd NatureCubePy
uv sync
```

---

## Quick start

```python
from naturecubepy import (
    auth_headers,
    get_key,
    get_project,
    get_station_info,
    get_camera_trap_data,
)
from naturecubepy.viz import station_map, station_explorer

# Authenticate (reads OKALA_API_KEY, or pass the key string directly)
hdr = auth_headers(get_key())

get_project(hdr)  # prints the active project name

stations = get_station_info(hdr, measurement_type="all")
camtrap = get_camera_trap_data(hdr, include_iucn_status=True)

# Interactive map
station_explorer(stations)

# Static satellite map
fig = station_map(stations, measurement_type="all")
fig.savefig("stations.png", dpi=300, bbox_inches="tight")
```

For local API development, pass a custom root:

```python
hdr = auth_headers(get_key(), "http://127.0.0.1:8000/api/")
```

---

## Next steps

Step-by-step notebooks live in [`tutorials/`](tutorials/README.md):

1. [Authentication](tutorials/01_authentication.ipynb)
2. [Data retrieval](tutorials/02_data_retrieval.ipynb)
3. [Data upload](tutorials/03_data_upload.ipynb)
4. [Visualization](tutorials/04_visualization.ipynb)

Common follow-ons:

```python
# All sensors in one call, or export figures/tables for a project
from naturecubepy import get_species_observations, load_project_data, export_project_assets

observations = get_species_observations(hdr)          # camera + audio + eDNA
bundle = load_project_data(hdr)                       # ObservationBundle for analysis/plots
assets = export_project_assets(hdr, output_dir="project_output")
```

Plotting helpers are in `naturecubepy.viz` (`station_map`, `records_per_class`, `plot_top_species`, `plot_edna_records`, `iucn_bar_plot`, and more) — see the visualization tutorial for examples of each.

---

## Development

```bash
uv run pytest   # tests
uv build        # build wheel/sdist
uv publish      # publish to PyPI
```

---

## License & support

Apache License 2.0.

Questions or bugs: open a GitHub issue or email [adam@okala.io](mailto:adam@okala.io).
