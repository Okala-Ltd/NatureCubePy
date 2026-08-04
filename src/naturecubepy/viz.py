import math
import numpy as np

import pandas as pd
import geopandas as gpd
import warnings
import math
import httpx
from pathlib import Path
from io import BytesIO

import folium
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------------

SENSOR_COLORS = {
    "Camera": "#1f77b4",
    "Bioacoustic": "#ff7f0e",
    "eDNA": "#2ca02c",
}

CLASS_COLORS = {
    "Mammalia": "#3E5859",
    "Aves": "#D9ACDE",
    "Reptilia": "#F6F6E2",
    "Insecta": "#9DEECF",
}

IUCN_COLOR_MAP = {
        "Critically Endangered": "#7f0000",
        "Endangered": "#d7301f",
        "Vulnerable": "#fc8d59",
        "Near Threatened": "#fdcc8a",
        "Least Concern": "#c7e9b4",
        "Data Deficient": "#9ecae1",
        "Not Evaluated": "#d9d9d9",
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


def _style_class_axis(
        ax,
        *,
        values: np.ndarray,
        y_label: str,
        panel_letter: str,
        font_family: str = "Arial",
        y_tick_step: int = 0,
        x_label_rotation: float = 45.0,
) -> None:
    """Apply axis formatting to a class bar chart axis."""
    def _nice_tick_step(max_value: float, n_ticks_target: int = 6) -> int:
        """Return a 1-2-5*10^k y-axis step close to max_value / n_ticks_target."""
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
        return int(nice * base)

    max_val = float(np.max(values)) if len(values) > 0 else 0.0
    step = y_tick_step if y_tick_step > 0 else _nice_tick_step(max_val)
    y_max = int(np.ceil(max_val / step) * step) if max_val > 0 else step
    y_lower_pad = max(0.8, step * 0.04)
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

        if lat_col is None or lon_col is None:
            raise ValueError(
                "Input must contain either a geometry column or latitude/longitude columns"
            )

        df["lat"] = df[lat_col]
        df["lon"] = df[lon_col]

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
# Mapping Functions
# ---------------------------------------------------------------------------
def station_map(
    stations: pd.DataFrame | gpd.GeoDataFrame,
    measurement_type: str = "all",
    *,
    project_boundary: str | Path | gpd.GeoDataFrame | None = None,
    output_path: str | Path | None = None
):
    """Plot sampling locations as a static PNG-ready figure over satellite imagery.

    The project boundary overlay is optional and only drawn when provided.
    """
    gdf = load_stations(stations)

    # ---- Sensor filtering ----
    if measurement_type != "all":
        if "measurement_type" not in gdf.columns:
            raise ValueError("No measurement_type column available")

        gdf = gdf[gdf["measurement_type"] == measurement_type].copy()


    if gdf.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No sampling locations available for selected sensor", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax.spines[side].set_visible(False)
        return fig

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)  # assume WGS84 lat/lon

    gdf_plot = gdf.to_crs(epsg=3857)


    scale_size = make_size_scaler(gdf_plot["record_count"])
    gdf_plot["plot_size"] = gdf_plot["record_count"].apply(scale_size)

    types = sorted(
        gdf_plot["measurement_type"]
        .fillna("Unknown")
        .unique()
    )

    palette = list(TABLEAU_COLORS.values())

    color_map = {
        t: palette[i % len(palette)]
        for i, t in enumerate(types)
    }

    boundary_gdf = load_project_boundary(project_boundary, target_crs=gdf_plot.crs)
    if boundary_gdf is not None:
        boundary_gdf = boundary_gdf[boundary_gdf.geometry.notna()].copy()
        if boundary_gdf.empty:
            boundary_gdf = None

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set a padded extent so satellite tiles and points are visible even for tight clusters.
    minx, miny, maxx, maxy = gdf_plot.total_bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    pad_x = span_x * 0.20
    pad_y = span_y * 0.20
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if not _add_satellite_basemap(ax, crs=gdf_plot.crs):
        warnings.warn("Could not load satellite basemap; plotting without tiles.", stacklevel=2)

    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color="#1f2933", linewidth=2, zorder=3)

    for sensor, group in gdf_plot.groupby(
    gdf_plot["measurement_type"].fillna("Unknown")
    ):

        ax.scatter(
            group.geometry.x,
            group.geometry.y,
            s=group["plot_size"],
            c=color_map[sensor],
            edgecolors="#1f2933",
            linewidths=1,
            alpha=0.85,
            zorder=4,
            label=sensor,
        )

    ## LEGENDS ##
    _add_scalebar(ax)

    add_size_legend(
        ax,
        values=gdf_plot["record_count"],
        size_func=scale_size,
        title="Record Count",
    )
    add_category_legend(
        ax,
        SENSOR_COLORS,
        title="Measurement Type",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    return fig


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
    figsize=(10, 5),
    save_path: str | Path | None = None,
    font_family="Arial",
    linewidth=4,
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
    gap = 3


    for year in years:

        idx = timeline.index[
            timeline["year"] == year
        ]

        values = cursor + np.arange(len(idx))

        y[idx] = values
        centers[year] = values.mean()

        cursor = values.max() + gap + 1


    fig_height = max(
        8,
        len(timeline) * 0.03,
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height),
        facecolor="white",
    )


    colors = cm.Set3(
        np.linspace(0, 1, len(years))
    )

    year_colors = dict(
        zip(years, colors)
    )


    for year in years:

        idx = timeline.index[
            timeline["year"] == year
        ]

        ax.axhspan(
            y[idx].min() - 0.5,
            y[idx].max() + 0.5,
            color=year_colors[year],
            alpha=0.25,
        )


    for year in years:

        mask = timeline["year"] == year

        ax.hlines(
            y=y[mask],
            xmin=timeline.loc[mask, "start_plot"],
            xmax=timeline.loc[mask, "end_plot"],
            color=year_colors[year],
            linewidth=4,
        )


    ax.scatter(
        timeline["start_plot"],
        y,
        s=22,
        color="#1f4e79",
        edgecolor="white",
        label="Start",
    )

    ax.scatter(
        timeline["end_plot"],
        y,
        s=22,
        color="#c0392b",
        edgecolor="white",
        label="End",
    )


    for year in years:

        ax.text(
            pd.Timestamp("2000-01-10"),
            centers[year],
            str(year),
            fontsize=11,
            fontweight="bold",
        )


    ax.set_xlim(
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2000-12-31"),
    )


    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
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
        alpha=0.5,
    )

    ax.grid(False, axis="y")


    for side in [
        "top",
        "right",
        "left",
    ]:
        ax.spines[side].set_visible(False)


    ax.legend(
        frameon=False,
        fontsize=font_size,
    )


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
):
    """
    Plot records and/or species counts per taxonomic class.

    Parameters
    ----------
    panel:
        "both"    : records + species panels
        "records" : only number of records
        "species" : only number of species
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
    subset = df.copy()

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


    for ax, (class_name, data, color) in zip(
        axes,
        panels,
    ):

        ax.barh(
            data["species"],
            data["records"],
            color=color,
            edgecolor="#333333",
            linewidth=0.6,
        )

        ax.set(
            xlabel="Number of records",
            ylabel=f"{class_name} species",
        )

        ax.tick_params(
            axis="y",
            labelsize=9,
        )

        ax.tick_params(
            axis="x",
            labelsize=9,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.4,
        )

        _style_class_axis(
            ax,
            values=data["records"].to_numpy(dtype=float),
            y_label=f"{class_name} species",
            panel_letter=None,
            font_family="Arial",
            x_label_rotation=x_label_rotation,
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
        "Mammalia": CLASS_COLORS[0],
        "Aves": CLASS_COLORS[1],
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


