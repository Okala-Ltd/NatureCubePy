# NatureCubePy Tutorials

Step-by-step guides for using NatureCubePy with the Okala API.

## Getting Started

| Tutorial | Format | Description |
|----------|--------|-------------|
| [01 - Authentication](01_authentication.ipynb) | Notebook | Set up API keys and connect to Okala |
| [02 - Data Retrieval](02_data_retrieval.ipynb) | Notebook | Retrieve stations, labels, and observation datasets |
| [03 - Data Upload](03_data_upload.ipynb) | Notebook | Update observations and labels |
| [04 - Visualization](04_visualization.ipynb) | Notebook | Plot stations, species, accumulation, and eDNA |

## Requirements

- Python 3.12+
- A valid Okala API key
- NatureCubePy installed

## Quick Start

```python
import os
from naturecubepy import auth_headers, get_project

# Set your API key
os.environ["OKALA_API_KEY"] = "your_key_here"

# Authenticate and verify
hdr = auth_headers(os.environ["OKALA_API_KEY"])
get_project(hdr)
```
