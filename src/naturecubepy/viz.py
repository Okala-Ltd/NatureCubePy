from __future__ import annotations

import math
import numpy as np

import pandas as pd
import geopandas as gpd
import warnings
import math
import httpx
from pathlib import Path
import tempfile
import subprocess
import shutil
import json
from io import BytesIO

import folium
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Rectangle
from pyproj import Transformer
from datetime import date
from PIL import Image
from shapely import wkt as shapely_wkt


# ---------------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------------

SENSOR_COLORS = {
    "Camera": "#AEF8C1",
    "Bioacoustic": "#F8F5E4",
    "eDNA": "#67ECB5",
}

# Core Okala palette.
CORE_PALETTE = [
    "#093939",
    "#AEF7C0",
    "#F8F5E4",
    "#45546A",
    "#67ECB5",
]

# Biodiversity plots palette.
BIODIVERSITY_PALETTE = [
    "#DDA0DD",
    "#304E4F",
    "#F5F5DC",
    "#76EEC6",
]

# Sampled from the Okala lockup; used for station markers so they carry the
# brand mint and stay legible over satellite imagery.
LOGO_MINT = "#AEF8C1"

# Map legend panel: Okala dark teal ground with cream type and mint accents.
PANEL_BG = "#093939"
PANEL_TEXT = "#F8F5E4"
PANEL_ACCENT = "#67ECB5"
PANEL_MUTED = "#A9C6BE"

CLASS_COLORS = {
    "Mammalia": "#304E4F",
    "Aves": "#DDA0DD",
    "Reptilia": "#F5F5DC",
    "Insecta": "#76EEC6",
}

IUCN_COLOR_MAP = {
        "Extinct": "#000000",
        "Extinct in the Wild": "#542344",
        "Critically Endangered": "#D81E05",
        "Endangered": "#FC7F3F",
        "Vulnerable": "#F9E814",
        "Near Threatened": "#CCE226",
        "Least Concern": "#60C659",
        "Data Deficient": "#D1D1C6",
        "Not Evaluated": "#FFFFFF",
        "Not Applicable": "#C1B5A5",
}


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def add_category_legend(
    ax,
    category_colors,
    *,
    categories=None,
    title="Measurement Type",
    loc="upper right",
    anchor=(1.0, 0.2)
):

    if categories is None:
        categories = category_colors.keys()

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=category_colors[category],
            markeredgecolor="black",
            markersize=8,
            label=str(category),
        )
        for category in categories
        if category in category_colors
    ]

    legend = ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        frameon=True,
        bbox_to_anchor=anchor,
    )

    ax.add_artist(legend)

    return legend


def add_size_legend(
    ax,
    values,
    size_func,
    *,
    title: str = "Size",
    loc: str = "lower right",
    anchor=(1.0, 0.35),
    color: str = "lightgrey",
    n_breaks: int = 4,
    nice_breaks: bool = True,
):
    """
    Add a marker size legend.

    Parameters
    ----------
    values : array-like
        Raw values (e.g. record_count).
    size_func : callable
        Function mapping value -> marker size.
    """

    values = np.asarray(values)

    if nice_breaks:
        vmax = np.nanmax(values)
        vmin = np.nanmin(values)

        if vmax == vmin:
            values = np.array([vmax])

        else:
            # choose rounding magnitude
            magnitude = 10 ** np.floor(np.log10(vmax))

            # round step to a sensible interval
            step = magnitude / 10

            # create evenly spaced breaks
            values = np.linspace(
                vmin,
                vmax,
                n_breaks,
            )

            # round to nearest step
            values = np.round(values / step) * step

            # remove duplicates
            values = np.unique(values).astype(int)

    handles = [
        plt.scatter(
            [],
            [],
            s=size_func(v),
            facecolor=color,
            edgecolor="black",
        )
        for v in values
    ]

    legend = ax.legend(
        handles,
        [f"{v:,.0f}" for v in values],
        title=title,
        loc=loc,
        frameon=True,
        bbox_to_anchor=anchor,
        scatterpoints=1,
        labelspacing=1.2,
        borderpad=0.8,
        handletextpad=1.0,
    )

    ax.add_artist(legend)

    return legend


def make_size_breaks(
    values,
    *,
    n=3,
):
    """
    Generate sensible legend breaks.

    Examples
    --------
    [5, 120, 850]
    """

    values = np.asarray(values)

    if len(values) == 0:
        return [1]

    return np.unique(
        np.percentile(
            values,
            np.linspace(0, 100, n),
        ).round()
    )


def _nice_tick_step(max_value: float, n_ticks_target: int = 6) -> int:
    """Return a 1-2-5*10^k tick step close to max_value / n_ticks_target."""
    if max_value <= 0:
        return 1
    raw = max_value / max(1, n_ticks_target)
    exponent = int(math.floor(math.log10(raw)))
    base = 10 ** exponent
    scaled = raw / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    # int() truncates for sub-1 bases (e.g. max_value=1 → step 0.2), so clamp.
    return max(1, int(round(nice * base)))


def _style_class_axis(
        ax,
        *,
        values: np.ndarray,
        y_label: str,
        panel_letter: str,
        font_family: str = "Arial",
        y_tick_step: int = 0,
        x_label_rotation: float = 45.0,
        y_baseline_pad: float = 0.03,
) -> None:
    """Apply axis formatting to a class bar chart axis."""
    max_val = float(np.nanmax(values)) if len(values) > 0 else 0.0
    if not np.isfinite(max_val):
        max_val = 0.0
    step = y_tick_step if y_tick_step > 0 else _nice_tick_step(max_val)
    if step <= 0:
        step = 1
    y_max = int(np.ceil(max_val / step) * step) if max_val > 0 else step
    # Pad as a fraction of the axis range so y=0 lands at the same height in
    # every panel, whatever each panel's data scale is.
    y_lower_pad = y_max * y_baseline_pad
    ax.set_ylim(-y_lower_pad, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, step))
    ax.spines["left"].set_bounds(0, y_max)

    ax.set_ylabel(y_label, fontfamily=font_family)
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", length=0, rotation=x_label_rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
        lbl.set_rotation_mode("anchor")
        lbl.set_fontfamily(font_family)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily(font_family)

    ax.text(
        -0.12, 1.06, panel_letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=14, fontweight="bold",
        fontfamily=font_family,
        clip_on=False,
    )


def _species_series(df: pd.DataFrame) -> pd.Series:
    if "species" in df.columns:
        return df["species"].fillna("").astype(str).str.strip()
    if "label" in df.columns:
        return df["label"].fillna("").astype(str).str.strip()
    return pd.Series(dtype="object")


def _taxonomic_class_series(df: pd.DataFrame) -> pd.Series:
    for col in ["class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
        if col in df.columns:
            out = df[col].fillna("Unknown").astype(str).str.strip()
            return out.mask(out == "", "Unknown")
    return pd.Series(["Unknown"] * len(df), index=df.index, dtype="object")


# ---------------------------------------------------------------------------
# Mapping Helpers
# ---------------------------------------------------------------------------
_WEB_MERCATOR_LIMIT = 20037508.342789244

def _mercator_to_tile(x: float, y: float, zoom: int) -> tuple[int, int]:
    n_tiles = 2**zoom
    tile_x = int(((x + _WEB_MERCATOR_LIMIT) / (2 * _WEB_MERCATOR_LIMIT)) * n_tiles)
    tile_y = int((( _WEB_MERCATOR_LIMIT - y) / (2 * _WEB_MERCATOR_LIMIT)) * n_tiles)
    tile_x = max(0, min(n_tiles - 1, tile_x))
    tile_y = max(0, min(n_tiles - 1, tile_y))
    return tile_x, tile_y


def _tile_bounds(tile_x: int, tile_y: int, zoom: int) -> tuple[float, float, float, float]:
    n_tiles = 2**zoom
    tile_span = (2 * _WEB_MERCATOR_LIMIT) / n_tiles
    minx = -_WEB_MERCATOR_LIMIT + tile_x * tile_span
    maxx = minx + tile_span
    maxy = _WEB_MERCATOR_LIMIT - tile_y * tile_span
    miny = maxy - tile_span
    return minx, maxx, miny, maxy


def _choose_satellite_zoom(x_span: float, y_span: float) -> int:
    span = max(x_span, y_span)
    if span <= 1500:
        return 16
    if span <= 3000:
        return 15
    if span <= 7000:
        return 14
    if span <= 15000:
        return 13
    if span <= 30000:
        return 12
    if span <= 60000:
        return 11
    return 10


def _fetch_satellite_tile(tile_x: int, tile_y: int, zoom: int) -> np.ndarray | None:
    url = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
        f"tile/{zoom}/{tile_y}/{tile_x}"
    )
    try:
        response = httpx.get(url, timeout=20.0)
        response.raise_for_status()
        return plt.imread(BytesIO(response.content), format="jpeg")
    except Exception:
        return None


def _add_satellite_basemap(ax, *, crs: object) -> bool:
    if str(crs).upper() not in {"EPSG:3857", "3857"}:
        return False

    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    x_span = maxx - minx
    y_span = maxy - miny
    if x_span <= 0 or y_span <= 0:
        return False

    zoom = _choose_satellite_zoom(x_span, y_span)
    min_tile_x, max_tile_y = _mercator_to_tile(minx, miny, zoom)
    max_tile_x, min_tile_y = _mercator_to_tile(maxx, maxy, zoom)

    added_any = False
    for tile_x in range(min(min_tile_x, max_tile_x), max(min_tile_x, max_tile_x) + 1):
        for tile_y in range(min(min_tile_y, max_tile_y), max(min_tile_y, max_tile_y) + 1):
            image = _fetch_satellite_tile(tile_x, tile_y, zoom)
            if image is None:
                continue
            tile_minx, tile_maxx, tile_miny, tile_maxy = _tile_bounds(tile_x, tile_y, zoom)
            ax.imshow(
                image,
                extent=(tile_minx, tile_maxx, tile_miny, tile_maxy),
                interpolation="bilinear",
                zorder=0,
            )
            added_any = True

    return added_any


def _add_scalebar(ax, length_m: float | None = None):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return

    if length_m is None:
        target = width * 0.2
        choices = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], dtype=float)
        length_m = float(choices[np.argmin(np.abs(choices - target))])

    start_x = x0 + width * 0.06
    start_y = y0 + height * 0.08
    end_x = start_x + length_m

    ax.plot([start_x, end_x], [start_y, start_y], color="#ffffff", linewidth=3, solid_capstyle="butt")
    ax.plot([start_x, start_x], [start_y - height * 0.008, start_y + height * 0.008], color="#ffffff", linewidth=2)
    ax.plot([end_x, end_x], [start_y - height * 0.008, start_y + height * 0.008], color="#ffffff", linewidth=2)

    if length_m >= 1000:
        label = f"{length_m / 1000:.0f} km"
    else:
        label = f"{int(length_m)} m"
    ax.text((start_x + end_x) / 2, start_y + height * 0.02, label, ha="center", va="bottom", fontsize=9, color="#ffffff")


def make_size_scaler(values, min_size=50, max_size=400, pad=0.1):

    values = pd.Series(values).astype(float)

    vmin = values.min()
    vmax = values.max()

    if vmax == vmin:
        return lambda x: (min_size + max_size) / 2

    # add padding so min/max do not hit exact limits
    value_range = vmax - vmin
    vmin = max(0, vmin - value_range * pad)
    vmax = vmax + value_range * pad

    sqrt_min = np.sqrt(vmin)
    sqrt_max = np.sqrt(vmax)

    def scale(x):
        return min_size + (
            (np.sqrt(x) - sqrt_min)
            / (sqrt_max - sqrt_min)
            * (max_size - min_size)
        )

    return scale


def load_stations(
    stations: pd.DataFrame | gpd.GeoDataFrame,
):
    df = stations.copy()

    # ---- Get coordinates ----
    if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
        if df.crs and not df.crs.is_geographic:
            points = df.to_crs(epsg=4326).geometry
        else:
            points = df.geometry

        df["lat"] = points.y
        df["lon"] = points.x

    else:
        lat_col = next(
            (c for c in ["lat", "latitude", "Latitude"] if c in df.columns),
            None,
        )
        lon_col = next(
            (c for c in ["lon", "lng", "longitude", "Longitude"] if c in df.columns),
            None,
        )

        if lat_col is not None and lon_col is not None:
            df["lat"] = df[lat_col]
            df["lon"] = df[lon_col]
        elif "geometry" in df.columns:
            geoms = []
            for value in df["geometry"]:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    geoms.append(None)
                    continue
                if hasattr(value, "x") and hasattr(value, "y"):
                    geoms.append(value)
                    continue
                try:
                    geoms.append(shapely_wkt.loads(str(value)))
                except Exception:
                    geoms.append(None)
            gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
            gdf = gdf[gdf.geometry.notna()].copy()
            df = gdf
            df["lat"] = df.geometry.y
            df["lon"] = df.geometry.x
        else:
            raise ValueError(
                "Input must contain either a geometry column or latitude/longitude columns"
            )

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid station coordinates found")

    return df


def load_project_boundary(
    project_boundary: str | Path | gpd.GeoDataFrame | None,
    *,
    target_crs: object,
) -> gpd.GeoDataFrame | None:
    if project_boundary is None:
        return None

    try:
        if isinstance(project_boundary, gpd.GeoDataFrame):
            boundary = project_boundary.copy()
        else:
            boundary = gpd.read_file(str(project_boundary))
    except Exception as exc:
        warnings.warn(f"Could not read project boundary; no boundary will be drawn. Original error: {exc}", stacklevel=2)
        return None

    if boundary.empty:
        warnings.warn("Provided project boundary is empty; no boundary will be drawn.", stacklevel=2)
        return None

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326", allow_override=True)

    try:
        return boundary.to_crs(target_crs)
    except Exception as exc:
        warnings.warn(f"Could not reproject project boundary; no boundary will be drawn. Original error: {exc}", stacklevel=2)
        return None 


# ---------------------------------------------------------------------------
# Cartography helpers (station_map report layout)
# ---------------------------------------------------------------------------

def _nice_degree_step(span_deg: float, target_ticks: int = 5) -> float:
    if span_deg <= 0:
        return 0.1
    raw = span_deg / max(1, target_ticks)
    exp = math.floor(math.log10(raw))
    base = 10 ** exp
    scaled = raw / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * base)


def _format_lon(lon: float) -> str:
    hemi = "E" if lon >= 0 else "W"
    return f"{abs(lon):.2f}°{hemi}"


def _format_lat(lat: float) -> str:
    hemi = "N" if lat >= 0 else "S"
    return f"{abs(lat):.2f}°{hemi}"


def _add_lonlat_ticks(ax, *, font_family: str = "DejaVu Sans", color: str = "#111111") -> None:
    """Draw lon/lat tick labels around a Web Mercator map frame."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    to_lonlat = Transformer.from_crs(3857, 4326, always_xy=True)
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)

    lon0, lat0 = to_lonlat.transform(x0, y0)
    lon1, lat1 = to_lonlat.transform(x1, y1)
    lon_min, lon_max = sorted([lon0, lon1])
    lat_min, lat_max = sorted([lat0, lat1])

    lon_step = _nice_degree_step(lon_max - lon_min)
    lat_step = _nice_degree_step(lat_max - lat_min)

    def _tick_start(vmin: float, step: float) -> float:
        return math.ceil(vmin / step) * step

    lon_ticks = []
    lon = _tick_start(lon_min, lon_step)
    while lon <= lon_max + 1e-9:
        lon_ticks.append(lon)
        lon += lon_step

    lat_ticks = []
    lat = _tick_start(lat_min, lat_step)
    while lat <= lat_max + 1e-9:
        lat_ticks.append(lat)
        lat += lat_step

    tick_kw = dict(color=color, linewidth=0.8, clip_on=False, zorder=6)
    label_kw = dict(fontsize=7, color=color, fontfamily=font_family, clip_on=False)

    # Bottom / top longitude ticks
    for lon in lon_ticks:
        x, _ = to_merc.transform(lon, (lat_min + lat_max) / 2)
        if not (x0 <= x <= x1):
            continue
        dy = (y1 - y0) * 0.012
        ax.plot([x, x], [y0, y0 + dy], **tick_kw)
        ax.plot([x, x], [y1 - dy, y1], **tick_kw)
        ax.text(x, y0 - dy * 1.8, _format_lon(lon), ha="center", va="top", **label_kw)
        ax.text(x, y1 + dy * 1.8, _format_lon(lon), ha="center", va="bottom", **label_kw)

    # Left / right latitude ticks
    for lat in lat_ticks:
        _, y = to_merc.transform((lon_min + lon_max) / 2, lat)
        if not (y0 <= y <= y1):
            continue
        dx = (x1 - x0) * 0.012
        ax.plot([x0, x0 + dx], [y, y], **tick_kw)
        ax.plot([x1 - dx, x1], [y, y], **tick_kw)
        ax.text(x0 - dx * 1.8, y, _format_lat(lat), ha="right", va="center", **label_kw)
        ax.text(x1 + dx * 1.8, y, _format_lat(lat), ha="left", va="center", **label_kw)


def _axes_frac_per_inch(ax) -> tuple[float, float]:
    """Axes-fraction units per inch for an axes spanning 0-1 in both directions."""
    fig_w, fig_h = ax.figure.get_size_inches()
    pos = ax.get_position()
    width_in = max(pos.width * fig_w, 1e-6)
    height_in = max(pos.height * fig_h, 1e-6)
    return 1.0 / width_in, 1.0 / height_in


def _nice_scalebar_length(target_m: float) -> float:
    """Largest 1/2/5 x 10^k value that still fits the available bar width."""
    if target_m <= 0:
        return 100.0
    exponent = math.floor(math.log10(target_m))
    for mantissa in (5.0, 2.0, 1.0):
        candidate = mantissa * 10.0**exponent
        if candidate <= target_m:
            return candidate
    return 10.0**exponent


def _draw_north_arrow(
    ax,
    *,
    x: float,
    y: float,
    height_in: float = 0.34,
    color: str = "#111111",
    background: str = "#ffffff",
) -> float:
    """Two-tone north arrow drawn in axes-fraction coordinates.

    ``x``/``y`` are the arrow centre and base; returns the y position just
    below the arrow so callers can stack further items underneath.
    """
    fx, fy = _axes_frac_per_inch(ax)
    arrow_h = height_in * fy
    arrow_w = height_in * 0.42 * fx

    tip = y + arrow_h
    waist = y + arrow_h * 0.28
    common = dict(closed=True, edgecolor=color, linewidth=0.6, zorder=4, clip_on=False)
    ax.add_patch(Polygon([(x, tip), (x - arrow_w, y), (x, waist)], facecolor=color, **common))
    ax.add_patch(Polygon([(x, tip), (x + arrow_w, y), (x, waist)], facecolor=background, **common))
    ax.text(
        x,
        tip + 0.02 * fy,
        "N",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=color,
        zorder=4,
        clip_on=False,
    )
    return y


def _draw_scalebar(
    ax,
    map_ax,
    *,
    x: float,
    y: float,
    max_width: float = 0.78,
    color: str = "#111111",
    background: str = "#ffffff",
) -> tuple[float, float]:
    """Segmented scale bar drawn in axes-fraction coordinates of a sidebar axes.

    The bar length is derived from the Web Mercator extent of ``map_ax``,
    corrected to ground distance at the centre latitude of the map. Returns the
    drawn bar's ``(width, height)`` in axes fractions so callers can align other
    elements to it.
    """
    x0, x1 = map_ax.get_xlim()
    y0, y1 = map_ax.get_ylim()
    data_w = x1 - x0
    data_h = y1 - y0
    if data_w <= 0 or data_h <= 0:
        return 0.0, 0.0

    fig_w, fig_h = map_ax.figure.get_size_inches()
    map_pos = map_ax.get_position()
    # With aspect="equal" the drawn map is letterboxed inside its subplot box,
    # so the tighter of the two dimensions sets the projected metres per inch.
    merc_per_in = max(
        data_w / max(map_pos.width * fig_w, 1e-6),
        data_h / max(map_pos.height * fig_h, 1e-6),
    )

    _, lat_centre = Transformer.from_crs(3857, 4326, always_xy=True).transform(
        (x0 + x1) / 2, (y0 + y1) / 2
    )
    ground_per_in = merc_per_in * math.cos(math.radians(lat_centre))
    if not np.isfinite(ground_per_in) or ground_per_in <= 0:
        return 0.0, 0.0

    fx, fy = _axes_frac_per_inch(ax)
    bar_in = _nice_scalebar_length(ground_per_in * (max_width / fx)) / ground_per_in
    bar_w = bar_in * fx
    length_m = bar_in * ground_per_in
    bar_h = 0.075 * fy

    n_segs = 4
    seg_w = bar_w / n_segs
    for i in range(n_segs):
        ax.add_patch(
            Rectangle(
                (x + i * seg_w, y),
                seg_w,
                bar_h,
                facecolor=color if i % 2 == 0 else background,
                edgecolor=color,
                linewidth=0.7,
                zorder=4,
                clip_on=False,
            )
        )

    unit_km = length_m >= 1000
    for frac in (0.0, 0.5, 1.0):
        value = length_m * frac / 1000 if unit_km else length_m * frac
        label = f"{value:g}"
        if frac == 1.0:
            label += " km" if unit_km else " m"
        ax.text(
            x + bar_w * frac,
            y + bar_h * 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
            zorder=4,
            clip_on=False,
        )
    return bar_w, bar_h


def _default_logo_path() -> Path | None:
    """Locate the bundled Okala lockup, falling back to a repo-level assets folder."""
    candidates = [Path(__file__).parent / "assets" / "okala_logo.png"]
    for parent in Path(__file__).resolve().parents[:4]:
        candidates.extend(sorted((parent / "assets").glob("*okala*ogo*.png")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_okala_logo(logo_path: str | Path | None) -> np.ndarray | None:
    """Load Okala lockup as RGBA (dark logo for light/white sidebars)."""
    if logo_path is None:
        logo_path = _default_logo_path()
    if logo_path is None:
        return None
    path = Path(logo_path)
    if not path.exists():
        warnings.warn(f"Okala logo not found at {path}", stacklevel=2)
        return None
    try:
        img = Image.open(path).convert("RGBA")
        return np.asarray(img)
    except Exception as exc:
        warnings.warn(f"Could not load Okala logo: {exc}", stacklevel=2)
        return None


def _draw_station_map_sidebar(
    ax,
    *,
    types_present: list[str],
    color_map: dict[str, object],
    has_boundary: bool,
    logo_rgba: np.ndarray | None,
    title: str = "Legend",
    map_ax=None,
    panel_top: float = 1.0,
    panel_bottom: float = 0.0,
) -> None:
    """Legend panel on the Okala dark ground, aligned to the map frame.

    ``panel_top``/``panel_bottom`` are the map frame edges in this axes'
    fraction coordinates; the panel grows downwards if its content needs more
    room than the map frame height.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fx, fy = _axes_frac_per_inch(ax)

    pad = 0.20 * fy
    left = 0.08
    row_h = 0.24 * fy
    logo_w = 0.72

    logo_h = 0.0
    if logo_rgba is not None:
        img_h, img_w = logo_rgba.shape[:2]
        logo_h = (logo_w / fx) * (img_h / img_w) * fy

    meta_lines = [
        "Coordinate system: EPSG:3857",
        "Basemap: Esri World Imagery",
        "Data: Okala / NatureCube",
        f"Date: {date.today().strftime('%b %Y')}",
    ]
    meta_h = len(meta_lines) * 0.19 * fy

    scale_h = 0.075 * fy * 1.5 + 0.14 * fy + 0.50 * fy
    entries_h = (
        0.34 * fy
        + (row_h if has_boundary else 0.0)
        + row_h * len(types_present)
        + 0.30 * fy
        + row_h * 3
    )
    needed = pad + logo_h + 0.28 * fy + entries_h + 0.34 * fy + scale_h + 0.40 * fy + meta_h + pad

    top = min(1.0, max(panel_top, needed))
    bottom = max(0.0, min(panel_bottom, top - needed))
    ax.add_patch(
        Rectangle(
            (0.0, bottom),
            1.0,
            top - bottom,
            facecolor=PANEL_BG,
            edgecolor="none",
            zorder=0,
            clip_on=False,
        )
    )

    def rule(y: float) -> None:
        ax.plot([left, 1 - left], [y, y], color=PANEL_TEXT, alpha=0.25, linewidth=0.8, zorder=1)

    y = top - pad

    # Mint lockup sits at the head of the panel, on the dark ground.
    if logo_rgba is not None:
        ax.imshow(
            logo_rgba,
            extent=((1 - logo_w) / 2, (1 + logo_w) / 2, y - logo_h, y),
            aspect="auto",
            zorder=2,
            interpolation="bilinear",
        )
        y -= logo_h + 0.14 * fy
        rule(y)
        y -= 0.14 * fy

    ax.text(left, y, title, fontsize=13, fontweight="bold", color=PANEL_TEXT, va="top", fontfamily="DejaVu Sans")
    y -= 0.44 * fy

    chip_w = 0.23 * fx
    label_x = left + chip_w + 0.05

    if has_boundary:
        ax.plot([left, left + chip_w], [y, y], color=PANEL_TEXT, linewidth=2.4, solid_capstyle="butt", zorder=3)
        ax.text(label_x, y, "Project boundary", fontsize=9, color=PANEL_TEXT, va="center")
        y -= row_h

    for sensor in types_present:
        color = color_map.get(sensor, LOGO_MINT)
        ax.scatter([left + chip_w / 2], [y], s=62, c=[color], edgecolors="#1f2933", linewidths=0.8, zorder=3)
        ax.text(label_x, y, str(sensor), fontsize=9, color=PANEL_TEXT, va="center")
        y -= row_h

    y -= 0.30 * fy
    ax.text(left, y, "Record count", fontsize=9, fontweight="bold", color=PANEL_ACCENT, va="center")
    y -= row_h
    for size, label in [(20, "Low"), (55, "Medium"), (110, "High")]:
        ax.scatter([left + chip_w / 2], [y], s=size, c=[LOGO_MINT], edgecolors="#1f2933", linewidths=0.7, zorder=3)
        ax.text(label_x, y, label, fontsize=8, color=PANEL_TEXT, va="center")
        y -= row_h

    # Scale bar first, so the north arrow can be centred on the drawn bar. The
    # pair is centred in whatever space is left above the source metadata.
    meta_rule_y = bottom + pad + meta_h + 0.12 * fy
    if map_ax is not None:
        slack = (y - 0.20 * fy - meta_rule_y) - scale_h
        bar_y = meta_rule_y + max(slack / 2, 0.10 * fy)
        bar_w, bar_h = _draw_scalebar(
            ax,
            map_ax,
            x=left,
            y=bar_y,
            max_width=1 - 2 * left,
            color=PANEL_TEXT,
            background=PANEL_BG,
        )
        if bar_w > 0:
            _draw_north_arrow(
                ax,
                x=left + bar_w / 2,
                y=bar_y + bar_h * 1.5 + 0.14 * fy,
                color=PANEL_TEXT,
                background=PANEL_BG,
            )

    # Metadata pinned to the foot of the panel
    meta_y = bottom + pad
    rule(meta_rule_y)
    for line in reversed(meta_lines):
        ax.text(left, meta_y, line, fontsize=7, color=PANEL_MUTED, va="bottom")
        meta_y += 0.19 * fy


def station_map(
    stations: pd.DataFrame | gpd.GeoDataFrame,
    measurement_type: str = "all",
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    output_path: str | Path | None = None,
    logo_path: str | Path | None = None,
):
    """Cartography-style sampling map for reports.

    Layout matches Okala map products: satellite map with lon/lat edge ticks
    and a legend panel on the Okala dark ground carrying the mint lockup,
    sensor keys, north arrow, segmented scale bar, and source metadata.

    ``logo_path`` defaults to the Okala lockup bundled with the package; pass a
    path to override it.
    """
    def _no_stations_figure():
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=PANEL_BG)
        ax.set_facecolor(PANEL_BG)
        ax.text(
            0.5,
            0.5,
            "No sampling locations available for selected sensor",
            ha="center",
            va="center",
            color=PANEL_TEXT,
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig

    try:
        gdf = load_stations(stations)
    except ValueError:
        return _no_stations_figure()

    if measurement_type != "all":
        if "measurement_type" not in gdf.columns:
            raise ValueError("No measurement_type column available")
        gdf = gdf[gdf["measurement_type"] == measurement_type].copy()

    if gdf.empty:
        return _no_stations_figure()

    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(
            gdf,
            geometry=gpd.points_from_xy(gdf["lon"], gdf["lat"]),
            crs="EPSG:4326",
        )
    elif gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_plot = gdf.to_crs(epsg=3857)
    if "record_count" not in gdf_plot.columns:
        gdf_plot["record_count"] = 1
    gdf_plot["record_count"] = pd.to_numeric(gdf_plot["record_count"], errors="coerce").fillna(1)

    scale_size = make_size_scaler(gdf_plot["record_count"])
    gdf_plot["plot_size"] = gdf_plot["record_count"].apply(scale_size)

    types = sorted(gdf_plot["measurement_type"].fillna("Unknown").unique().tolist()) if "measurement_type" in gdf_plot.columns else ["Station"]
    # Use the light Okala brand tints so sensor types remain distinct while
    # retaining enough contrast over satellite imagery.
    color_map = {
        t: SENSOR_COLORS.get(_canonical_measurement_type(t), LOGO_MINT)
        for t in types
    }

    boundary_gdf = load_project_boundary(project_boundary, target_crs=gdf_plot.crs)
    if boundary_gdf is not None:
        boundary_gdf = boundary_gdf[boundary_gdf.geometry.notna()].copy()
        if boundary_gdf.empty:
            boundary_gdf = None

    fig = plt.figure(figsize=(11.5, 8.0), facecolor=PANEL_BG)
    # wspace leaves room for the map's right-hand latitude labels, which are
    # drawn outside the map frame and would otherwise run into the legend.
    gs = GridSpec(1, 2, width_ratios=[3.05, 1.0], wspace=0.12, left=0.06, right=0.98, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])

    minx, miny, maxx, maxy = gdf_plot.total_bounds
    if boundary_gdf is not None:
        bminx, bminy, bmaxx, bmaxy = boundary_gdf.total_bounds
        if np.isfinite([bminx, bminy, bmaxx, bmaxy]).all():
            minx = min(minx, bminx)
            miny = min(miny, bminy)
            maxx = max(maxx, bmaxx)
            maxy = max(maxy, bmaxy)

    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    pad_x = span_x * 0.08
    pad_y = span_y * 0.08
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if not _add_satellite_basemap(ax, crs=gdf_plot.crs):
        warnings.warn("Could not load satellite basemap; plotting without tiles.", stacklevel=2)

    if boundary_gdf is not None:
        # Dark halo under a thick white line so the boundary reads over both
        # bright and dark satellite imagery.
        boundary_gdf.boundary.plot(ax=ax, color="#111111", linewidth=5.0, alpha=0.45, zorder=3)
        boundary_gdf.boundary.plot(ax=ax, color="#ffffff", linewidth=2.8, zorder=4)

    group_col = gdf_plot["measurement_type"].fillna("Unknown") if "measurement_type" in gdf_plot.columns else pd.Series(["Station"] * len(gdf_plot), index=gdf_plot.index)
    for sensor, group in gdf_plot.groupby(group_col):
        ax.scatter(
            group.geometry.x,
            group.geometry.y,
            s=group["plot_size"],
            c=[color_map.get(sensor, "#999999")],
            edgecolors="#1f2933",
            linewidths=0.8,
            alpha=0.9,
            zorder=5,
            label=str(sensor),
        )

    # Ticks and frame carry the panel's cream, since the labels sit outside the
    # frame on the dark ground.
    _add_lonlat_ticks(ax, color=PANEL_TEXT)

    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(PANEL_TEXT)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Equal aspect shrinks the map axes to fit its data, so the frame's real
    # position is only known after a draw. Resolve it before laying out the
    # panel so the panel edges line up with the map frame.
    fig.canvas.draw()
    map_pos = ax.get_position()
    leg_pos = ax_leg.get_position()

    logo_rgba = _load_okala_logo(logo_path)
    _draw_station_map_sidebar(
        ax_leg,
        types_present=types,
        color_map=color_map,
        has_boundary=boundary_gdf is not None,
        logo_rgba=logo_rgba,
        title="Legend",
        map_ax=ax,
        panel_top=(map_pos.y1 - leg_pos.y0) / leg_pos.height,
        panel_bottom=(map_pos.y0 - leg_pos.y0) / leg_pos.height,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15, facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Mapping Functions (explorers / timelines)
# ---------------------------------------------------------------------------
def station_explorer(
    stations: pd.DataFrame | gpd.GeoDataFrame,
    measurement_type: str | None = "all",
) -> folium.Map:
    """Plot station locations on an interactive Folium map.

    Circle markers are sized proportionally to the number of media records
    at each station.

    Parameters
    ----------
    stations:
        A DataFrame or GeoDataFrame containing station information.
    measurement_type:
        One of 'all' (default), 'Camera'

    Returns
    -------
    folium.Map
        An interactive map widget.

    Examples
    --------
    >>> m = station_explorer(stations)  # doctest: +SKIP
    """
    df = load_stations(stations)

    # ---- Sensor filtering ----
    if measurement_type != "all":
        if "measurement_type" not in df.columns:
            raise ValueError("No measurement_type column available")

        df = df[df["measurement_type"] == measurement_type].copy()

    # ---- Map centre ----
    m = folium.Map(
        location=[df["lat"].mean(), df["lon"].mean()],
        zoom_start=10,
        tiles="Esri WorldImagery",
    )

    # ---- Marker scaling ----
    counts = df.get("record_count", pd.Series(1, index=df.index)).fillna(1)

    min_count = counts.min()
    max_count = counts.max()
    count_range = max_count - min_count

    def rescale(value, low=5, high=15):
        if count_range == 0:
            return (low + high) / 2

        return low + (value - min_count) / count_range * (high - low)

    # ---- Add markers ----
    for _, row in df.iterrows():

        record_count = row.get("record_count", 1)

        popup_html = f"""
        Device ID: {row.get('device_id', '')}<br>
        Measurement type: {row.get('measurement_type', '')}<br>
        Data type: {row.get('data_type', '')}<br>
        Start time: {row.get('project_system_record_start_timestamp', '')}<br>
        End time: {row.get('project_system_record_end_timestamp', '')}<br>
        No. media files: {record_count}<br>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=rescale(record_count),
            tooltip=str(row.get("device_id", "")),
            popup=folium.Popup(popup_html, max_width=300),
            color="red",
            fill=True,
            fill_opacity=0.6,
            opacity=0.2,
        ).add_to(m)

    return m


def camera_activity_timeline(
    df: pd.DataFrame,
    *,
    id_col: str | None = None,
    start_cols: list[str] | None = None,
    end_cols: list[str] | None = None,
    figsize=(8.5, 5),
    save_path: str | Path | None = None,
    font_family="Arial",
    marker_size=36,
    day_lines=True,
    font_size=12
):
    """
    Plot camera deployment activity timeline.

    Parameters
    ----------
    df:
        Camera deployment dataframe.

    id_col:
        Camera identifier column.

    start_cols:
        Candidate start timestamp columns.

    end_cols:
        Candidate end timestamp columns.

    marker_size:
        Marker area for the start/end points.

    day_lines:
        Draw a faint vertical line on every day covered by a deployment.

    Returns
    -------
    fig, timeline_df
    """

    start_cols = start_cols or [
        "installation_timestamp",
        "deployment_timestamp",
        "project_system_record_start_timestamp",
    ]

    end_cols = end_cols or [
        "removal_timestamp",
        "project_system_record_end_timestamp",
    ]


    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No camera records available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig, pd.DataFrame()


    work = df.copy()


    id_col = id_col or next(
        (
            c for c in [
                "device_id",
                "station_name",
                "feature_id",
            ]
            if c in work.columns
        ),
        None,
    )

    if id_col is None:
        raise KeyError("No camera identifier column found")


    start_cols = [
        c for c in start_cols
        if c in work.columns
    ]

    end_cols = [
        c for c in end_cols
        if c in work.columns
    ]

    if not start_cols or not end_cols:
        raise KeyError("Missing timestamp columns")


    def coalesce_dates(cols):
        out = pd.Series(
            pd.NaT,
            index=work.index,
        )

        for col in cols:
            out = out.fillna(
                pd.to_datetime(
                    work[col],
                    errors="coerce",
                    utc=True,
                )
                .dt.tz_localize(None)
            )

        return out


    work["start"] = coalesce_dates(start_cols)
    work["end"] = coalesce_dates(end_cols)


    work = work.dropna(
        subset=[id_col]
    )

    work["start"] = work["start"].fillna(work["end"])
    work["end"] = work["end"].fillna(work["start"])

    work = work.dropna(
        subset=["start", "end"]
    )


    work[["start", "end"]] = pd.DataFrame(
        {
            "start": work[["start", "end"]].min(axis=1),
            "end": work[["start", "end"]].max(axis=1),
        },
        index=work.index,
    )


    timeline = (
        work.groupby(id_col)
        .agg(
            start=("start", "min"),
            end=("end", "max"),
        )
        .reset_index()
    )


    timeline["year"] = (
        timeline["start"]
        .dt.year
        .astype(int)
    )


    timeline["year_start"] = pd.to_datetime(
        timeline["year"].astype(str) + "-01-01"
    )

    timeline["year_end"] = pd.to_datetime(
        timeline["year"].astype(str) + "-12-31"
    )


    timeline["start_plot"] = pd.to_datetime(
        "2000-"
        + timeline["start"]
        .dt.strftime("%m-%d")
    )

    timeline["end_plot"] = pd.to_datetime(
        "2000-"
        + timeline["end"]
        .dt.strftime("%m-%d")
    )


    years = sorted(
        timeline["year"].unique()
    )


    # ---- stack rows ----
    y = np.zeros(len(timeline))

    centers = {}
    cursor = 0
    gap = 0


    for year in years:

        idx = timeline.index[
            timeline["year"] == year
        ]

        values = cursor + np.arange(len(idx))

        y[idx] = values
        centers[year] = values.mean()

        cursor = values.max() + gap + 1


    # Height follows the number of stacked rows so bars stay a sensible
    # thickness once they are drawn edge to edge.
    total_slots = max(int(cursor), 1)

    fig_height = float(
        np.clip(
            total_slots * 0.18 + 1.5,
            3.0,
            22.0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height),
        facecolor="white",
    )


    # Light palette colours carry the year bands and dark ones the bars, so
    # every fill is an exact hex with no transparency.
    band_cycle = [
        CORE_PALETTE[2],  # cream
        CORE_PALETTE[1],  # pale mint
        CORE_PALETTE[4],  # mint
    ]

    bar_cycle = [
        CORE_PALETTE[0],  # deep teal
        CORE_PALETTE[3],  # slate
    ]

    band_colors = {
        year: band_cycle[i % len(band_cycle)]
        for i, year in enumerate(years)
    }

    bar_colors = {
        year: bar_cycle[i % len(bar_cycle)]
        for i, year in enumerate(years)
    }


    for year in years:

        idx = timeline.index[
            timeline["year"] == year
        ]

        ax.axhspan(
            y[idx].min() - 0.5,
            y[idx].max() + 0.5,
            color=band_colors[year],
            zorder=0,
        )


    if day_lines:

        tracked_days = sorted({
            day
            for start, end in zip(
                timeline["start_plot"],
                timeline["end_plot"],
            )
            for day in pd.date_range(start, end, freq="D")
        })

        # White reads as a light line against all three band colours.
        for day in tracked_days:
            ax.axvline(
                day,
                color="white",
                linewidth=0.35,
                zorder=1,
            )


    starts = mdates.date2num(timeline["start_plot"])
    ends = mdates.date2num(timeline["end_plot"])

    # date2num is in days, so a same-day deployment still gets a visible bar.
    widths = np.maximum(ends - starts, 1.0)


    for year in years:

        mask = (timeline["year"] == year).to_numpy()

        ax.barh(
            y[mask],
            widths[mask],
            left=starts[mask],
            height=0.8,
            color=bar_colors[year],
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )


    # Contrasting hues with a dark outline so both stay readable wherever they
    # land, on a dark bar or on a light year band.
    ax.scatter(
        timeline["start_plot"],
        y,
        s=marker_size,
        color="#76EEC6",
        edgecolor="#093939",
        linewidth=1.0,
        zorder=3,
        label="Start",
    )

    ax.scatter(
        timeline["end_plot"],
        y,
        s=marker_size,
        color="#DDA0DD",
        edgecolor="#093939",
        linewidth=1.0,
        zorder=3,
        label="End",
    )


    for year in years:

        ax.text(
            -0.01,
            centers[year],
            str(year),
            transform=ax.get_yaxis_transform(),
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            fontfamily=font_family,
            zorder=4,
        )


    # Pad the year so a marker sitting on 1 Jan or 31 Dec is not clipped in
    # half by the edge of the axes.
    axis_pad = pd.Timedelta(days=5)

    ax.set_xlim(
        pd.Timestamp("2000-01-01") - axis_pad,
        pd.Timestamp("2000-12-31") + axis_pad,
    )

    # Match the year bands exactly so no white strip shows above the x axis.
    if len(y):
        ax.set_ylim(
            y.min() - 0.5,
            y.max() + 0.5,
        )


    # Fixed ticks rather than a locator, so the padded limits do not add a
    # thirteenth month at either end.
    ax.set_xticks(
        pd.date_range(
            "2000-01-01",
            "2000-12-01",
            freq="MS",
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b")
    )


    ax.set_yticks([])
    ax.set_xlabel("Month",
                  fontsize=font_size,
                  fontfamily=font_family)


    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontsize(font_size)
        tick.set_fontfamily(font_family)


    ax.grid(
        axis="x",
        linestyle="--",
        color=CORE_PALETTE[3],
        linewidth=0.7,
    )

    ax.grid(False, axis="y")
    ax.set_axisbelow(True)


    for side in [
        "top",
        "right",
        "left",
    ]:
        ax.spines[side].set_visible(False)


    # Anchored just outside the top right corner so it can never sit on top of
    # a deployment that runs late in the year.
    legend = ax.legend(
        fontsize=font_size,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=CORE_PALETTE[3],
        framealpha=1.0,
    )

    legend.set_zorder(5)


    plt.tight_layout()


    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


    return fig, timeline


# ---------------------------------------------------------------------------
# Species Records
# ---------------------------------------------------------------------------
def _class_record_species_counts(df: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame({
        "class": _taxonomic_class_series(df),
        "species": _species_series(df),
    })

    records = (
        work.groupby("class")
        .size()
        .rename("number_of_records")
    )

    species = (
        work.loc[work["species"].ne(""), ["class", "species"]]
        .drop_duplicates()
        .groupby("class")
        .size()
        .rename("number_of_species")
    )

    return (
        pd.concat([records, species], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values("number_of_records", ascending=False)
        .reset_index(drop=True)
    )


def records_per_class(
    df: pd.DataFrame,
    *,
    class_levels: list[str],
    class_colors: dict[str, object] = CLASS_COLORS,
    panel: str = "both",
    font_family: str = "Arial",
    figsize: tuple[float, float] = (10.0, 3.8),
    x_label_rotation: float = 45.0,
    dpi: float = 200.0,
):
    """
    Plot records and/or species counts per taxonomic class.

    Parameters
    ----------
    panel:
        "both"    : records + species panels
        "records" : only number of records
        "species" : only number of species
    dpi:
        Figure resolution. Above the matplotlib default so labels stay sharp
        when the figure is shown inline or saved as a raster.
    """

    if panel not in {"both", "records", "species"}:
        raise ValueError(
            "panel must be 'both', 'records', or 'species'"
        )

    metrics = (
        _class_record_species_counts(df)
        .set_index("class")
        .reindex(class_levels, fill_value=0)
        .reset_index()
    )

    metrics = (
        metrics
        .query(
            "number_of_records > 0 or number_of_species > 0"
        )
        .reset_index(drop=True)
    )

    if metrics.empty:
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            facecolor="white",
        )

        ax.text(
            0.5,
            0.5,
            "No class data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)

        return fig


    panels = {
        "records": (
            "number_of_records",
            "Number of records",
            "A",
        ),
        "species": (
            "number_of_species",
            "Number of species",
            "B",
        ),
    }


    selected = (
        list(panels)
        if panel == "both"
        else [panel]
    )

    fig, axes = plt.subplots(
        ncols=len(selected),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        facecolor="white",
    )

    axes = axes.flatten()

    colors = [
        class_colors.get(c, "#999999")
        for c in metrics["class"]
    ]


    for ax, key in zip(axes, selected):

        column, ylabel, letter = panels[key]

        ax.bar(
            metrics["class"],
            metrics[column],
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
            clip_on=False,
        )

        _style_class_axis(
            ax,
            values=metrics[column].to_numpy(dtype=float),
            y_label=ylabel,
            panel_letter=letter,
            font_family=font_family,
            x_label_rotation=x_label_rotation,
        )


    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.22,
        top=0.94,
        wspace=0.32,
    )

    return fig


def top_species_by_class(
    df: pd.DataFrame,
    class_name: str,
    *,
    class_col="class",
    species_col="species",
    top_n=10,
):
    if df.empty:
        return pd.DataFrame(columns=["species", "records"])

    subset = df.copy()

    if class_col not in subset.columns:
        # Fall back to any taxonomic class column if present.
        for candidate in ["class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
            if candidate in subset.columns:
                class_col = candidate
                break
        else:
            return pd.DataFrame(columns=["species", "records"])

    if species_col not in subset.columns:
        for candidate in ["species", "common_name", "label"]:
            if candidate in subset.columns:
                species_col = candidate
                break
        else:
            return pd.DataFrame(columns=["species", "records"])

    subset[class_col] = (
        subset[class_col]
        .astype(str)
        .str.strip()
    )

    species = (
        subset.loc[
            subset[class_col] == class_name,
            species_col,
        ]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    return (
        species.value_counts()
        .head(top_n)
        .sort_values()
        .rename_axis("species")
        .reset_index(name="records")
    )


def plot_top_species(
    df: pd.DataFrame,
    *,
    classes: dict[str, str] = CLASS_COLORS,
    top_n: int = 10,
    figsize=(7, 6),
    x_label_rotation: float = 0,
    font_family: str = "Arial",
    panel_letters: bool = True,
):
    """
    Plot top species for supplied classes.

    Parameters
    ----------
    classes:
        Mapping of class name -> color

        Example:
        {
            "Mammalia": "blue",
            "Aves": "green",
        }
    panel_letters:
        Label each class panel with a bold letter (A, B, C, ...).
    """

    panels = []

    for class_name, color in classes.items():
        data = top_species_by_class(
            df,
            class_name,
            top_n=top_n,
        )

        if not data.empty:
            panels.append(
                (
                    class_name,
                    data,
                    color,
                )
            )


    if not panels:
        fig, ax = plt.subplots(figsize=figsize)

        ax.text(
            0.5,
            0.5,
            "No species data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.axis("off")
        return fig


    fig, axes = plt.subplots(
        ncols=len(panels),
        figsize=(figsize[0] * len(panels), figsize[1]),
        squeeze=False,
    )

    axes = axes.flatten()


    for index, (ax, (class_name, data, color)) in enumerate(
        zip(
            axes,
            panels,
        )
    ):

        positions = np.arange(len(data))

        ax.barh(
            positions,
            data["records"],
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
        )

        # One tick per bar, so every species is named.
        ax.set_yticks(positions)
        ax.set_yticklabels(data["species"])
        ax.set_ylim(-0.7, len(positions) - 0.3)

        max_records = float(data["records"].max())
        step = _nice_tick_step(max_records)
        x_max = int(np.ceil(max_records / step) * step) if max_records > 0 else step

        ax.set_xlim(0, x_max)
        ax.set_xticks(np.arange(0, x_max + 1, step))
        ax.spines["bottom"].set_bounds(0, x_max)

        ax.set_xlabel(
            "Number of records",
            fontfamily=font_family,
            fontsize=12,
        )

        ax.set_ylabel(
            class_name,
            fontfamily=font_family,
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )

        ax.tick_params(
            axis="y",
            length=0,
            labelsize=12,
        )

        ax.tick_params(
            axis="x",
            rotation=x_label_rotation,
            labelsize=11,
        )

        for label in ax.get_xticklabels():
            label.set_fontfamily(font_family)

        # Scientific names are conventionally italicised.
        for label in ax.get_yticklabels():
            label.set_fontfamily(font_family)
            label.set_fontstyle("italic")

        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)

        ax.set_axisbelow(True)

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.4,
        )

        if panel_letters:
            # Above the panel, taking its x from the class name (y-axis label)
            # so the letter lines up with it however wide the species names are.
            ax.annotate(
                chr(ord("A") + index),
                xy=(0.5, 1.02),
                xycoords=(ax.yaxis.label, "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                fontfamily=font_family,
                annotation_clip=False,
            )


    plt.tight_layout()

    return fig


def plot_top_species_by_sensor(
    camera_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    *,
    top_n: int = 10,
):
    """
    Return top species plots for each sensor.
    """

    classes = {
        "Mammalia": CLASS_COLORS["Mammalia"],
        "Aves": CLASS_COLORS["Aves"],
    }

    return {
        "camera": plot_top_species(
            camera_df,
            classes=classes,
            top_n=top_n,
        ),
        "bioacoustic": plot_top_species(
            bio_df,
            classes=classes,
            top_n=top_n,
        ),
    }


def iucn_bar_plot(
    df: pd.DataFrame,
    *,
    class_col="class",
    species_col="species",
    iucn_col="iucn_status",
    rank_col=None,
    exclude_statuses=None,
    status_order=None,
    class_order=None,
    color_map=IUCN_COLOR_MAP,
    legend_title="IUCN Status",
    font_family="Arial",
    figsize=(6.2, 3.8),
    return_summary=False,
):

    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No eDNA records available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig


    work = df[
        [class_col, species_col, iucn_col]
        + ([rank_col] if rank_col else [])
    ].copy()


    work.columns = (
        ["class", "species", "iucn"]
        + (["rank"] if rank_col else [])
    )


    # clean
    work = work.dropna(subset=["species", "iucn"])

    work["class"] = (
        work["class"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    work["species"] = work["species"].astype(str).str.strip()
    work["iucn"] = work["iucn"].astype(str).str.strip()


    if rank_col:
        work = work[
            work["rank"].str.lower()
            == "species"
        ]


    if exclude_statuses:
        work = work[
            ~work["iucn"].isin(exclude_statuses)
        ]


    if work.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No species-level taxa available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig


    # standardise IUCN labels
    work["iucn"] = work["iucn"].replace({
        "Least concern": "Least Concern",
        "Near threatened": "Near Threatened",
        "Data deficient": "Data Deficient",
        "Critically endangered": "Critically Endangered",
        "Not evaluated": "Not Evaluated",
        "NE": "Not Evaluated",
    })


    summary = (
        work.drop_duplicates(
            ["class", "species", "iucn"]
        )
        .groupby(["class", "iucn"])
        .size()
        .unstack(fill_value=0)
    )


    if class_order:
        summary = summary.reindex(
            class_order,
            fill_value=0,
        )


    if status_order:
        summary = summary[
            [
                x for x in status_order
                if x in summary.columns
            ]
        ]


    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor="white",
    )


    summary.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[
            color_map.get(
                c,
                "#bdbdbd",
            )
            for c in summary.columns
        ],
        edgecolor="#333333",
        linewidth=0.5,
    )


    ax.set_ylabel(
        "Number of unique species",
        fontfamily=font_family,
    )

    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


    ax.tick_params(
        axis="x",
        rotation=45,
        length=0,
    )

    legend = ax.legend(
        title=legend_title,
        frameon=False,
    )

    if legend:
        legend.get_title().set_fontfamily(
            font_family
        )


    fig.tight_layout()

    if return_summary:
        return fig, summary.reset_index()

    return fig

# ---------------------------------------------------------------------------
# Accumulation / eDNA stacks / report figure export
# ---------------------------------------------------------------------------

PLOT_COLORS = list(CLASS_COLORS.values())

def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in [
        "timestamp",
        "observation_timestamp",
        "segment_start_timestamp",
        "segment_timestamp",
        "media_file_timestamp",
        "project_system_record_start_timestamp",
    ]:
        if col in df.columns:
            return col
    return None

def _apply_minimal_axes_style(ax, *, grid_axis: str | None = "y"):
    if grid_axis is not None:
        ax.grid(axis=grid_axis, alpha=0.25)
        ax.set_axisbelow(True)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

def _filter_by_class(df: pd.DataFrame, class_name: str) -> pd.DataFrame:
    class_series = _taxonomic_class_series(df).astype(str).str.strip().str.lower()
    target = class_name.strip().lower()
    aliases = {
        "mammalia": ["mammalia", "mammal"],
        "aves": ["aves", "bird", "avian"],
    }
    tokens = aliases.get(target, [target])
    mask = class_series.apply(lambda value: any(token in value for token in tokens))
    return df[mask].copy()

def _plot_species_accumulation_for_groups(
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    groups: tuple[tuple[str, str], tuple[str, str]] = (("Mammalia", "Mammal"), ("Aves", "Bird")),
    *,
    effort_label: str = "Sampling days",
):
    # Keep one shared effort timeline across classes so panel A/B x-axes are comparable.
    shared_stations = _clip_station_windows_to_observation_range(stations_df, observations_df)

    panels: list[tuple[str, pd.DataFrame, str]] = []
    for idx, (class_name, display_name) in enumerate(groups):
        class_df = _filter_by_class(observations_df, class_name)
        if class_df.empty:
            continue
        curve = _incidence_species_accumulation_curve(
            class_df,
            shared_stations,
            clip_stations_to_observations=False,
        )
        if curve.empty:
            continue
        panels.append((chr(ord("A") + idx), curve, display_name))

    if not panels:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No mammal or bird accumulation data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_minimal_axes_style(ax, grid_axis=None)
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(ncols=len(panels), figsize=(7 * len(panels), 5), sharey=False)
    axes_list = np.atleast_1d(axes)
    show_letters = len(panels) > 1
    shared_x_max = max(float(curve["x"].max()) for _, curve, _ in panels)

    for ax, (label, curve, display_name) in zip(axes_list, panels):
        color = PLOT_COLORS[0] if display_name == "Mammal" else PLOT_COLORS[1]
        x_vals = curve["x"].to_numpy(dtype=float)
        mean_vals = curve["mean"].to_numpy(dtype=float)

        ax.fill_between(
            x_vals,
            curve["lower"].to_numpy(dtype=float),
            curve["upper"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(x_vals, mean_vals, color=color, linewidth=2)
        # Reference point: richness observed at the full sampling effort.
        ax.plot(x_vals[-1], mean_vals[-1], marker="o", markersize=5, color=color, zorder=3)
        if show_letters:
            ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
        ax.set_xlabel(effort_label)
        ax.set_ylabel("Species richness")
        ax.set_title(display_name)
        ax.set_xlim(0.0, max(1.0, shared_x_max))

        upper = float(curve["upper"].max()) if not curve.empty else 0.0
        ax.set_ylim(0, upper * 1.05 if upper > 0 else 1)
        _apply_minimal_axes_style(ax, grid_axis="both")

    plt.tight_layout()
    return fig

def plot_species_accumulation_mammal_bird_by_sensor(
    camera_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    include_unobserved_stations: bool = False,
) -> dict[str, plt.Figure]:
    """Return A/B mammal-vs-bird accumulation figures for camera and bioacoustic."""
    camera_stations = _filter_stations_for_sensor(
        stations_df,
        camera_df,
        include_unobserved_stations=include_unobserved_stations,
    )
    bio_stations = _filter_stations_for_sensor(
        stations_df,
        bio_df,
        include_unobserved_stations=include_unobserved_stations,
    )
    return {
        "camera": _plot_species_accumulation_for_groups(
            camera_df,
            camera_stations,
            effort_label="Camera trap days",
        ),
        "bioacoustic": _plot_species_accumulation_for_groups(
            bio_df,
            bio_stations,
            effort_label="Bioacoustic days",
        ),
    }

def _edna_class_rank_summary(
    edna_df: pd.DataFrame,
    *,
    class_col: str | None = None,
    rank_col: str | None = None,
    taxon_col: str | None = None,
    count_col: str | None = None,
) -> pd.DataFrame:
    """Build class/rank/count summary from either raw or pre-aggregated eDNA data."""
    if edna_df.empty:
        return pd.DataFrame(columns=["resolved_class", "resolved_rank", "unique_taxa"])

    work = edna_df.copy()

    # Case 1: caller already provides aggregated class/rank/count table.
    if class_col and rank_col and count_col and {class_col, rank_col, count_col}.issubset(work.columns):
        out = work[[class_col, rank_col, count_col]].copy()
        out.columns = ["resolved_class", "resolved_rank", "unique_taxa"]
        out["resolved_class"] = out["resolved_class"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        out["resolved_rank"] = out["resolved_rank"].fillna("Order").astype(str).str.strip().str.title()
        out["resolved_rank"] = out["resolved_rank"].replace({"Higher": "Order", "Order_Or_Higher": "Order"})
        out["unique_taxa"] = pd.to_numeric(out["unique_taxa"], errors="coerce").fillna(0).astype(int)
        return out.groupby(["resolved_class", "resolved_rank"], as_index=False)["unique_taxa"].sum()

    # Case 2: infer summary from raw records.
    class_source = class_col if class_col and class_col in work.columns else None
    if class_source is None:
        for candidate in ["Class", "class", "class_", "taxonomic_class", "class_name", "taxon_class"]:
            if candidate in work.columns:
                class_source = candidate
                break

    if class_source is None:
        resolved_class = pd.Series(["Unknown"] * len(work), index=work.index, dtype="object")
    else:
        resolved_class = work[class_source].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    rank_source = rank_col if rank_col and rank_col in work.columns else None
    if rank_source is None:
        for candidate in ["taxon_rank", "updated_taxon_rank_final", "updated_taxon_rank"]:
            if candidate in work.columns:
                rank_source = candidate
                break

    if rank_source is not None:
        resolved_rank = work[rank_source].fillna("Order").astype(str).str.strip().str.title()
    else:
        resolved_rank = pd.Series(["Order"] * len(work), index=work.index, dtype="object")

    resolved_rank = resolved_rank.replace({"Higher": "Order", "Order_Or_Higher": "Order"})

    if rank_source is None:
        # Derive lowest identification rank when explicit rank column is not present.
        for rank_name, cols in [
            ("Species", ["taxon_name", "Species", "species", "label"]),
            ("Genus", ["Genus", "genus"]),
            ("Family", ["Family", "family"]),
            ("Order", ["Order", "order"]),
        ]:
            existing = [c for c in cols if c in work.columns]
            if not existing:
                continue
            values = pd.Series([""] * len(work), index=work.index, dtype="object")
            for c in existing:
                vals = work[c].fillna("").astype(str).str.strip()
                values = values.mask(values == "", vals)
            if rank_name == "Species":
                mask = values != ""
                resolved_rank.loc[mask] = "Species"
            elif rank_name == "Genus":
                mask = (resolved_rank == "Order") & (values != "")
                resolved_rank.loc[mask] = "Genus"
            elif rank_name == "Family":
                mask = (resolved_rank == "Order") & (values != "")
                resolved_rank.loc[mask] = "Family"

    if taxon_col and taxon_col in work.columns:
        resolved_taxon = work[taxon_col].fillna("").astype(str).str.strip()
    else:
        resolved_taxon = pd.Series([""] * len(work), index=work.index, dtype="object")
        for candidate in ["taxon_name", "Species", "species", "label", "Genus", "genus", "Family", "family", "Order", "order"]:
            if candidate not in work.columns:
                continue
            vals = work[candidate].fillna("").astype(str).str.strip()
            resolved_taxon = resolved_taxon.mask(resolved_taxon == "", vals)
        resolved_taxon = resolved_taxon.mask(resolved_taxon == "", resolved_class)

    out = pd.DataFrame(
        {
            "resolved_class": resolved_class,
            "resolved_rank": resolved_rank,
            "resolved_taxon": resolved_taxon,
        }
    )
    out = out[(out["resolved_class"] != "") & (out["resolved_taxon"] != "")]

    uniq = out.drop_duplicates(["resolved_class", "resolved_rank", "resolved_taxon"])
    summary = (
        uniq.groupby(["resolved_class", "resolved_rank"], as_index=False)
        .size()
        .rename(columns={"size": "unique_taxa"})
    )
    return summary

def plot_edna_records(
    edna_df: pd.DataFrame,
    *,
    class_col: str | None = None,
    rank_col: str | None = None,
    taxon_col: str | None = None,
    count_col: str | None = None,
    rank_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    legend_title: str = "Identified to:",
    font_family: str = "Arial",
    figsize: tuple[float, float] = (5.2, 3.2),
    y_tick_step: int = 20,
    x_label_rotation: float = 45.0,
    return_summary: bool = False,
):
    """Plot compact stacked eDNA class-by-lowest-identification figure.

    The function accepts either:
    1. Raw eDNA records with taxonomic columns; or
    2. Pre-aggregated class/rank/count table (via class_col/rank_col/count_col).

    When ``return_summary=True``, returns a class-indexed summary table that
    includes rank counts and per-class percentages for each rank.
    """
    summary = _edna_class_rank_summary(
        edna_df,
        class_col=class_col,
        rank_col=rank_col,
        taxon_col=taxon_col,
        count_col=count_col,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    if summary.empty:
        ax.text(0.5, 0.5, "No eDNA records available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        if return_summary:
            return fig, pd.DataFrame()
        return fig

    rank_order_use = rank_order if rank_order is not None else ["Order", "Family", "Genus", "Species"]
    colors_use = color_map if color_map is not None else {
        "Order": "#304E4F",
        "Family": "#F5F5DC",
        "Genus": "#DDA0DD",
        "Species": "#76EEC6",
    }

    summary["resolved_rank"] = summary["resolved_rank"].astype(str).str.title().replace({"Higher": "Order"})
    available_ranks = summary["resolved_rank"].unique().tolist()
    ordered_ranks = [r for r in rank_order_use if r in available_ranks] + [
        r for r in available_ranks if r not in rank_order_use
    ]

    pivot = (
        summary.pivot_table(
            index="resolved_class",
            columns="resolved_rank",
            values="unique_taxa",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=ordered_ranks, fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    bottom = np.zeros(len(pivot), dtype=float)
    for rank in pivot.columns:
        values = pivot[rank].to_numpy(dtype=float)
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=colors_use.get(rank, "#bdbdbd"),
            edgecolor="#333333",
            linewidth=0.6,
            label=rank,
            clip_on=False,
        )
        bottom += values

    max_total = float(pivot.sum(axis=1).max()) if not pivot.empty else 0.0
    y_max = int(np.ceil(max_total / float(y_tick_step)) * y_tick_step) if max_total > 0 else y_tick_step
    # Small lower padding preserves the bottom bar edge without showing an x-axis baseline.
    y_lower_pad = max(0.8, float(y_tick_step) * 0.04)
    ax.set_ylim(-y_lower_pad, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, y_tick_step))
    ax.spines["left"].set_bounds(0, y_max)

    ax.set_ylabel("Number of unique taxa", fontfamily=font_family)
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, rotation=x_label_rotation, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
        label.set_fontfamily(font_family)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_family)

    legend = ax.legend(
        title=legend_title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        borderaxespad=0.0,
        prop={"family": font_family, "size": 8.5},
    )
    if legend is not None and legend.get_title() is not None:
        legend.get_title().set_fontfamily(font_family)
        legend.get_title().set_fontsize(9)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.96)

    if return_summary:
        summary_out = pivot.copy()
        totals = summary_out.sum(axis=1).astype(float)
        summary_out["Total"] = totals.astype(int)
        for rank in ordered_ranks:
            pct_col = f"{rank} (%)"
            summary_out[pct_col] = np.where(
                totals > 0,
                (summary_out[rank].astype(float) / totals) * 100.0,
                0.0,
            ).round(1)
        return fig, summary_out
    return fig

def plot_edna_unique_taxa_stacked(edna_df: pd.DataFrame):
    """Backward-compatible wrapper for legacy eDNA stacked figure calls."""
    return plot_edna_records(edna_df)

def _build_all_device_days(stations_df: pd.DataFrame) -> pd.DataFrame:
    if stations_df.empty or "device_id" not in stations_df.columns:
        return pd.DataFrame(columns=["device_id", "date"])

    start_candidates = [
        "project_system_record_start_timestamp",
        "installation_timestamp",
        "deployment_timestamp",
        "start_timestamp",
        "start_time",
    ]
    end_candidates = [
        "project_system_record_end_timestamp",
        "removal_timestamp",
        "retrieval_timestamp",
        "end_timestamp",
        "end_time",
    ]

    start_col = next((c for c in start_candidates if c in stations_df.columns), None)
    end_col = next((c for c in end_candidates if c in stations_df.columns), None)
    if start_col is None or end_col is None:
        return pd.DataFrame(columns=["device_id", "date"])

    def _as_day(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

    work = stations_df[["device_id", start_col, end_col]].copy()
    work[start_col] = _as_day(work[start_col])
    work[end_col] = _as_day(work[end_col])

    rows: list[pd.DataFrame] = []
    for _, row in work.iterrows():
        start = row[start_col]
        end = row[end_col]
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        dates = pd.date_range(start, end, freq="D")
        rows.append(pd.DataFrame({"device_id": row["device_id"], "date": dates}))

    if not rows:
        return pd.DataFrame(columns=["device_id", "date"])

    return pd.concat(rows, ignore_index=True).drop_duplicates(["device_id", "date"])

def _clip_station_windows_to_observation_range(
    stations_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clip station deployment windows to observed date ranges per device.

    This prevents inflated effort when a station end date is open-ended
    (for example, far-future placeholders) while observations are only present
    for a shorter period.
    """
    if stations_df.empty:
        return stations_df

    ts_col = _timestamp_column(observations_df)
    if ts_col is None or "device_id" not in observations_df.columns:
        return stations_df

    start_candidates = [
        "project_system_record_start_timestamp",
        "installation_timestamp",
        "deployment_timestamp",
        "start_timestamp",
        "start_time",
    ]
    end_candidates = [
        "project_system_record_end_timestamp",
        "removal_timestamp",
        "retrieval_timestamp",
        "end_timestamp",
        "end_time",
    ]

    start_col = next((c for c in start_candidates if c in stations_df.columns), None)
    end_col = next((c for c in end_candidates if c in stations_df.columns), None)
    if start_col is None or end_col is None:
        return stations_df

    obs = observations_df[["device_id", ts_col]].copy()
    obs["device_id"] = obs["device_id"].astype(str).str.strip()
    obs["_obs_date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs = obs.dropna(subset=["_obs_date"])
    if obs.empty:
        return stations_df

    obs_range = (
        obs.groupby("device_id", as_index=False)
        .agg(_obs_min=("_obs_date", "min"), _obs_max=("_obs_date", "max"))
    )

    out = stations_df.copy()
    out["device_id"] = out["device_id"].astype(str).str.strip()
    out = out.merge(obs_range, on="device_id", how="left")

    start = pd.to_datetime(out[start_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    end = pd.to_datetime(out[end_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

    obs_min = pd.to_datetime(out["_obs_min"], errors="coerce")
    obs_max = pd.to_datetime(out["_obs_max"], errors="coerce")

    clipped_start = start.copy()
    clipped_end = end.copy()

    mask_min = obs_min.notna() & clipped_start.notna()
    clipped_start.loc[mask_min] = clipped_start.loc[mask_min].where(
        clipped_start.loc[mask_min] >= obs_min.loc[mask_min],
        obs_min.loc[mask_min],
    )

    mask_max = obs_max.notna() & clipped_end.notna()
    clipped_end.loc[mask_max] = clipped_end.loc[mask_max].where(
        clipped_end.loc[mask_max] <= obs_max.loc[mask_max],
        obs_max.loc[mask_max],
    )

    out[start_col] = clipped_start
    out[end_col] = clipped_end

    valid = out[start_col].notna() & out[end_col].notna() & (out[end_col] >= out[start_col])
    out = out[valid].copy()
    return out.drop(columns=["_obs_min", "_obs_max"], errors="ignore")

def _incidence_frequency_vector(observations_df: pd.DataFrame, stations_df: pd.DataFrame) -> list[int]:
    """Build iNEXT-style incidence frequency vector: [n_sampling_units, species frequencies...]."""
    all_days = _build_all_device_days(stations_df)
    n_days = len(all_days)
    if n_days == 0:
        return [0]

    ts_col = _timestamp_column(observations_df)
    species_col = "common_name" if "common_name" in observations_df.columns else "species"
    if ts_col is None or "device_id" not in observations_df.columns or species_col not in observations_df.columns:
        return [n_days]

    obs = observations_df[["device_id", ts_col, species_col]].copy()
    obs["date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs["species"] = obs[species_col].fillna("").astype(str).str.strip()
    obs = obs.dropna(subset=["date"])
    obs = obs[obs["species"] != ""]
    if obs.empty:
        return [n_days]

    obs_days = obs[["device_id", "date", "species"]].drop_duplicates()
    merged = all_days.merge(obs_days, on=["device_id", "date"], how="inner")
    if merged.empty:
        return [n_days]

    freq = (
        merged.groupby("species", as_index=False)
        .size()
        .rename(columns={"size": "fq"})
        .sort_values("species")
    )
    values = [int(n_days)] + [int(v) for v in freq["fq"].tolist() if int(v) > 0]
    return values if values else [n_days]

def _inext_curve_via_r(freq_vector: list[int], endpoint: int) -> pd.DataFrame | None:
    """Run R iNEXT to get q=0 size-based curve and confidence bounds."""
    if endpoint <= 0:
        return pd.DataFrame(columns=["x", "mean", "lower", "upper"])
    if shutil.which("Rscript") is None:
        return None

    script = """
    suppressPackageStartupMessages({
      library(iNEXT)
      library(jsonlite)
    })
    freq <- c(__FREQ__)
    out <- iNEXT(freq, endpoint=__ENDPOINT__, datatype='incidence_freq', se=TRUE, conf=0.95)
    est <- out$iNextEst$size_based
    est <- est[est$q == 0, c('t', 'qD', 'qD.LCL', 'qD.UCL')]
    names(est) <- c('x', 'mean', 'lower', 'upper')
    cat(toJSON(est, dataframe='rows', auto_unbox=TRUE))
    """.replace("__FREQ__", ",".join(str(v) for v in freq_vector)).replace("__ENDPOINT__", str(int(endpoint)))

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(script)
        r_path = fh.name

    try:
        proc = subprocess.run(
            ["Rscript", r_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        payload = proc.stdout.strip()
        if not payload:
            return None
        data = json.loads(payload)
        out = pd.DataFrame(data)
        if out.empty:
            return pd.DataFrame(columns=["x", "mean", "lower", "upper"])
        for col in ["x", "mean", "lower", "upper"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["x", "mean", "lower", "upper"]).sort_values("x").reset_index(drop=True)
        return out
    except Exception:
        return None
    finally:
        try:
            Path(r_path).unlink(missing_ok=True)
        except Exception:
            pass

def _filter_stations_for_sensor(
    stations_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    *,
    include_unobserved_stations: bool = False,
) -> pd.DataFrame:
    if stations_df.empty:
        return stations_df

    out = stations_df.copy()

    if "measurement_type" in out.columns and "measurement_type" in obs_df.columns:
        obs_types = set(obs_df["measurement_type"].dropna().astype(str).str.strip().str.lower().unique())
        if obs_types:
            station_types = out["measurement_type"].astype(str).str.strip().str.lower()
            out = out[station_types.isin(obs_types)].copy()

    # Optionally limit station effort to devices that occur in observations.
    # Keeping unobserved stations includes zero-detection effort in accumulation
    # curves, which is appropriate for effort accounting in some workflows.
    if (not include_unobserved_stations) and "device_id" in out.columns and "device_id" in obs_df.columns:
        obs_devices = set(obs_df["device_id"].dropna().astype(str).str.strip())
        if obs_devices:
            out = out[out["device_id"].astype(str).str.strip().isin(obs_devices)].copy()

    return out

def _effort_grid(n_units: int, n_points: int = 160) -> np.ndarray:
    """Effort values to evaluate a rarefaction curve on, always including 0 and n_units."""
    if n_units <= n_points:
        return np.arange(0, n_units + 1)
    grid = np.linspace(0.0, float(n_units), num=n_points).round().astype(int)
    return np.unique(np.concatenate(([0], grid, [n_units])))


def _expected_richness(incidence_counts: np.ndarray, n_units: int, grid: np.ndarray) -> np.ndarray:
    """Sample-based rarefaction: expected species richness for t of n_units samples.

    Uses the analytical form (Chao et al. 2014): with ``Y_i`` sampling units
    containing species ``i``, the chance of missing that species in ``t`` units
    is ``C(T - Y_i, t) / C(T, t)``, so expected richness is the sum of detection
    probabilities. Evaluated through log-factorials to stay stable for large
    effort.
    """
    counts = np.asarray(incidence_counts, dtype=int)
    counts = counts[counts > 0]
    if n_units <= 0 or counts.size == 0:
        return np.zeros(len(grid), dtype=float)

    log_factorial = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n_units + 1, dtype=float)))))
    absent = (n_units - counts)[:, None]
    t = np.asarray(grid, dtype=int)[None, :]

    # Species i can only be missed entirely while t <= T - Y_i.
    reachable = t <= absent
    log_miss = np.where(
        reachable,
        log_factorial[np.where(reachable, absent, 0)]
        - log_factorial[np.where(reachable, absent - t, 0)]
        - log_factorial[n_units]
        + log_factorial[n_units - t],
        -np.inf,
    )
    return (1.0 - np.exp(log_miss)).sum(axis=0)


def _bootstrap_richness_se(
    incidence_counts: np.ndarray,
    n_units: int,
    grid: np.ndarray,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Standard error of the rarefaction curve, via the iNEXT bootstrap scheme.

    Resampling the observed sampling units cannot produce more species than were
    seen, which pins the upper bound to the estimate at full effort. Instead this
    follows Chao et al. (2014): observed species get coverage-corrected detection
    probabilities, the undetected species implied by the Chao2 estimate share the
    leftover probability mass, and each replicate redraws incidences from that
    community.
    """
    counts = np.asarray(incidence_counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0 or n_units <= 0 or n_bootstrap <= 1:
        return np.zeros(len(grid), dtype=float)

    units = float(n_units)
    q1 = float(np.count_nonzero(counts == 1))
    q2 = float(np.count_nonzero(counts == 2))
    if q2 > 0:
        f0 = (units - 1) / units * q1 * q1 / (2 * q2)
    else:
        f0 = (units - 1) / units * q1 * (q1 - 1) / 2
    f0 = max(f0, 0.0)

    detection = counts / units
    undetected_share = np.power(1.0 - detection, units)
    unseen_mass = q1 / units * (units * f0 / (units * f0 + q1)) if q1 > 0 and f0 > 0 else 0.0
    spread = float(np.sum(detection * undetected_share))
    correction = unseen_mass / spread if spread > 0 else 0.0

    probs = np.clip(detection * (1.0 - correction * undetected_share), 1e-12, 1.0)
    n_unseen = int(np.ceil(f0))
    if n_unseen > 0 and unseen_mass > 0:
        probs = np.concatenate([probs, np.full(n_unseen, np.clip(unseen_mass / n_unseen, 1e-12, 1.0))])

    rng = np.random.default_rng(seed)
    replicates = np.empty((n_bootstrap, len(grid)), dtype=float)
    for i in range(n_bootstrap):
        replicates[i] = _expected_richness(rng.binomial(n_units, probs), n_units, grid)
    return replicates.std(axis=0, ddof=1)


def _incidence_species_accumulation_curve(
    observations_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    random_seed: int = 42,
    use_r_inext: bool = False,
    clip_stations_to_observations: bool = True,
) -> pd.DataFrame:
    """Compute incidence-based accumulation by unique device-day effort with CI."""
    if clip_stations_to_observations:
        stations_df = _clip_station_windows_to_observation_range(stations_df, observations_df)

    # Optional R iNEXT path (disabled by default to keep plotting Python-only).
    if use_r_inext:
        freq_vector = _incidence_frequency_vector(observations_df, stations_df)
        endpoint = int(freq_vector[0]) if freq_vector else 0
        inext_curve = _inext_curve_via_r(freq_vector, endpoint=endpoint)
        if inext_curve is not None:
            return inext_curve

    all_days = _build_all_device_days(stations_df)
    if all_days.empty:
        return pd.DataFrame(columns=["x", "mean", "lower", "upper"])

    samples = all_days.drop_duplicates(["device_id", "date"]).reset_index(drop=True)
    n_units = len(samples)

    def _flat_curve() -> pd.DataFrame:
        grid = _effort_grid(n_units)
        zeros = np.zeros(len(grid), dtype=float)
        return pd.DataFrame({"x": grid, "mean": zeros, "lower": zeros, "upper": zeros})

    ts_col = _timestamp_column(observations_df)
    species_col = "common_name" if "common_name" in observations_df.columns else "species"
    if ts_col is None or "device_id" not in observations_df.columns or species_col not in observations_df.columns:
        return _flat_curve()

    obs = observations_df[["device_id", ts_col, species_col]].copy()
    obs["date"] = pd.to_datetime(obs[ts_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    obs["species"] = obs[species_col].fillna("").astype(str).str.strip()
    obs = obs.dropna(subset=["date"])
    obs = obs[obs["species"] != ""]
    if obs.empty or n_units == 0:
        return _flat_curve()

    # Species-by-sampling-unit incidence matrix over the effort timeline.
    obs_unique = obs[["device_id", "date", "species"]].drop_duplicates()
    samples = samples.reset_index(names="_unit")
    merged = obs_unique.merge(samples, on=["device_id", "date"], how="inner")
    if merged.empty:
        return _flat_curve()

    incidence = (
        pd.crosstab(merged["species"], merged["_unit"])
        .reindex(columns=range(n_units), fill_value=0)
        .to_numpy()
        > 0
    )

    grid = _effort_grid(n_units)
    counts = incidence.sum(axis=1)
    mean = _expected_richness(counts, n_units, grid)
    se = _bootstrap_richness_se(counts, n_units, grid, n_bootstrap=n_bootstrap, seed=random_seed)

    return pd.DataFrame(
        {
            "x": grid,
            "mean": mean,
            "lower": np.clip(mean - 1.96 * se, 0.0, None),
            "upper": mean + 1.96 * se,
        }
    )

def _canonical_measurement_type(value: object) -> str:
    mt = str(value).strip().lower().replace("_", " ")
    if mt in {"camera", "camera trap", "image"}:
        return "Camera"
    if mt in {"bioacoustic", "audio", "bio acoustic"}:
        return "Bioacoustic"
    if mt in {"edna", "e dna"}:
        return "eDNA"
    if mt == "all":
        return "all"
    return str(value).strip()


def _filter_stations_by_sensor(stations_df: pd.DataFrame, measurement_type: str) -> pd.DataFrame:
    if measurement_type == "all" or stations_df.empty or "measurement_type" not in stations_df.columns:
        return stations_df
    wanted = _canonical_measurement_type(measurement_type)
    types = stations_df["measurement_type"].map(_canonical_measurement_type)
    return stations_df[types == wanted].copy()


def _ordered_class_levels(camera_df: pd.DataFrame, bio_df: pd.DataFrame) -> list[str]:
    classes = pd.concat(
        [_taxonomic_class_series(camera_df), _taxonomic_class_series(bio_df)],
        ignore_index=True,
    )
    classes = classes.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    unique_classes = classes.unique().tolist()

    def _rank(name: str) -> tuple[int, str]:
        lower = str(name).strip().lower()
        if any(token in lower for token in ["mammalia", "mammal"]):
            return (0, lower)
        if any(token in lower for token in ["aves", "bird", "avian"]):
            return (1, lower)
        return (2, lower)

    return sorted(unique_classes, key=_rank)


def _empty_figure(message: str, *, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    return fig


def save_all_figures(
    bundle,
    output_dir: str | Path,
    *,
    top_n: int = 10,
    dpi: int = 300,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    sensor_types: list[str] | tuple[str, ...] | None = None,
    logo_path: str | Path | None = None,
    filename_prefix: str | None = None,
) -> dict[str, str]:
    """Generate and save project figures using the plotters in this module.

    Only emits figures for sensors present in ``sensor_types`` (or
    ``bundle.sensor_types`` when omitted).

    When ``filename_prefix`` is set, files are named
    ``{prefix}_{key}.png``; dict keys remain the logical ``key``.
    """
    selected = set(sensor_types or getattr(bundle, "sensor_types", ("camera", "bioacoustic", "edna")))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    prefix = f"{filename_prefix}_" if filename_prefix else ""

    def _path(key: str) -> Path:
        return out / f"{prefix}{key}.png"

    map_specs = [("all_sampling_locations", "all")]
    if "camera" in selected:
        map_specs.append(("camera_sampling_locations", "Camera"))
    if "bioacoustic" in selected:
        map_specs.append(("bioacoustic_sampling_locations", "Bioacoustic"))
    if "edna" in selected:
        map_specs.append(("edna_sampling_locations", "eDNA"))

    for key, sensor in map_specs:
        stations = (
            bundle.stations
            if sensor == "all"
            else _filter_stations_by_sensor(bundle.stations, sensor)
        )
        try:
            fig = station_map(
                stations,
                measurement_type="all",
                project_boundary=project_boundary,
                logo_path=logo_path,
            )
        except Exception:
            fig = _empty_figure(f"No sampling locations available for {sensor}")
        path = _path(key)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        saved[key] = str(path)

    timeline_specs = []
    if "camera" in selected:
        timeline_specs.append(("timeline_camera_trap_activity", "Camera"))
    if "bioacoustic" in selected:
        timeline_specs.append(("timeline_bioacoustic_activity", "Bioacoustic"))

    for key, sensor in timeline_specs:
        subset = _filter_stations_by_sensor(bundle.stations, sensor)
        try:
            fig, _ = camera_activity_timeline(subset)
        except Exception:
            fig = _empty_figure(f"No {sensor} timeline data available", figsize=(12, 5))
        path = _path(key)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved[key] = str(path)

    class_levels = _ordered_class_levels(bundle.camera, bundle.bioacoustic)
    class_specs = []
    if "camera" in selected:
        class_specs.append(("species_per_class_camera", bundle.camera))
    if "bioacoustic" in selected:
        class_specs.append(("species_per_class_bioacoustic", bundle.bioacoustic))
    for key, frame in class_specs:
        fig = records_per_class(frame, class_levels=class_levels or ["Unknown"], class_colors=CLASS_COLORS)
        path = _path(key)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved[key] = str(path)

    if "camera" in selected or "bioacoustic" in selected:
        accumulation_figs = plot_species_accumulation_mammal_bird_by_sensor(
            bundle.camera if "camera" in selected else pd.DataFrame(),
            bundle.bioacoustic if "bioacoustic" in selected else pd.DataFrame(),
            bundle.stations,
        )
        if "camera" in selected:
            path = _path("species_accumulation_mammal_bird_camera")
            accumulation_figs["camera"].savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(accumulation_figs["camera"])
            saved["species_accumulation_mammal_bird_camera"] = str(path)
        if "bioacoustic" in selected:
            path = _path("species_accumulation_mammal_bird_bioacoustic")
            accumulation_figs["bioacoustic"].savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(accumulation_figs["bioacoustic"])
            saved["species_accumulation_mammal_bird_bioacoustic"] = str(path)

        top_figs = plot_top_species_by_sensor(
            bundle.camera if "camera" in selected else pd.DataFrame(),
            bundle.bioacoustic if "bioacoustic" in selected else pd.DataFrame(),
            top_n=top_n,
        )
        if "camera" in selected:
            path = _path("top_mammal_bird_species_camera")
            top_figs["camera"].savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(top_figs["camera"])
            saved["top_mammal_bird_species_camera"] = str(path)
        if "bioacoustic" in selected:
            path = _path("top_mammal_bird_species_bioacoustic")
            top_figs["bioacoustic"].savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(top_figs["bioacoustic"])
            saved["top_mammal_bird_species_bioacoustic"] = str(path)

    if "edna" in selected:
        if bundle.edna is not None and not bundle.edna.empty:
            edna_df = bundle.edna
        elif "measurement_type" in bundle.all_species.columns:
            edna_df = bundle.all_species[
                bundle.all_species["measurement_type"].map(_canonical_measurement_type) == "eDNA"
            ]
        else:
            edna_df = pd.DataFrame()
        edna_fig = plot_edna_unique_taxa_stacked(edna_df)
        edna_path = _path("edna_unique_taxa_stacked")
        edna_fig.savefig(edna_path, dpi=dpi, bbox_inches="tight")
        plt.close(edna_fig)
        saved["edna_unique_taxa_stacked"] = str(edna_path)

    return saved
