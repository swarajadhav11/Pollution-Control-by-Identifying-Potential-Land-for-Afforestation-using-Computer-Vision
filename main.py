"""
=============================================================================
PROJECT: Pollution Control by Identifying Potential Land for Afforestation
         using Computer Vision
=============================================================================
Author  : Academic Project
Purpose : Detect green/empty areas from satellite imagery using OpenCV
          and estimate how many trees can be planted to reduce pollution.
=============================================================================
"""

import os
import math
import time
import requests
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend — works in VS Code
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Default location: Aarey Colony, Mumbai — a well-known afforestation debate site
DEFAULT_LAT  = 19.1515
DEFAULT_LON  = 72.8650
ZOOM         = 16          # Zoom level  (15–17 works well for city blocks)
GRID_SIZE    = 3           # Download a GRID_SIZE × GRID_SIZE tile grid
TILE_SIZE    = 256          # Standard OSM/ESRI tile pixel size
TREE_SPACING = 5            # Assumed spacing between trees in metres
OUTPUT_DIR   = "outputs"    # All saved images go here

# Free ESRI World Imagery tile URL (no API key needed)
TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Request headers to mimic a browser (avoids occasional 403 blocks)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COORDINATE → TILE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """
    Convert geographic coordinates to OSM/ESRI tile (x, y) numbers.
    Uses the standard Web Mercator / Slippy-map formula.
    """
    lat_rad = math.radians(lat)
    n       = 2 ** zoom
    x_tile  = int((lon + 180.0) / 360.0 * n)
    y_tile  = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)

    # Clamp to valid tile range so we never request out-of-bounds tiles
    max_tile = n - 1
    x_tile   = max(0, min(x_tile, max_tile))
    y_tile   = max(0, min(y_tile, max_tile))
    return x_tile, y_tile


def tile_to_metres_per_pixel(lat: float, zoom: int) -> float:
    """
    Return the ground resolution (metres per pixel) at a given latitude / zoom.
    Formula: 156543.03 * cos(lat) / 2^zoom
    """
    return 156_543.03 * math.cos(math.radians(lat)) / (2 ** zoom)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DOWNLOAD A SINGLE TILE
# ─────────────────────────────────────────────────────────────────────────────

def download_tile(x: int, y: int, zoom: int, retries: int = 3) -> Image.Image | None:
    """
    Fetch one map tile from the ESRI World Imagery server.
    Returns a PIL Image, or None on failure.
    """
    url = TILE_URL.format(z=zoom, x=x, y=y)
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                print(f"    ⚠  Tile ({x},{y}) attempt {attempt}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"    ⚠  Tile ({x},{y}) attempt {attempt}: {e}")
        time.sleep(1)   # brief pause before retry
    return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DOWNLOAD GRID & STITCH
# ─────────────────────────────────────────────────────────────────────────────

def download_and_stitch(lat: float, lon: float, zoom: int, grid: int) -> np.ndarray:
    """
    Download a grid × grid block of tiles centred on (lat, lon) and
    stitch them into one large NumPy image array (BGR for OpenCV).
    """
    cx, cy = lat_lon_to_tile(lat, lon, zoom)

    # Tile coordinates for the grid (centred on the target tile)
    half   = grid // 2
    x_tiles = list(range(cx - half, cx - half + grid))
    y_tiles = list(range(cy - half, cy - half + grid))

    print(f"\n📡 Downloading {grid}×{grid} = {grid*grid} satellite tiles …")
    rows = []
    for yi, y in enumerate(y_tiles):
        row_images = []
        for xi, x in enumerate(x_tiles):
            print(f"   Tile ({xi+1},{yi+1}) / ({grid},{grid})  →  x={x}, y={y}, z={zoom}")
            tile = download_tile(x, y, zoom)
            if tile is None:
                # Fill with black if the tile failed
                tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
            row_images.append(np.array(tile))
        rows.append(np.hstack(row_images))

    stitched_rgb = np.vstack(rows)
    stitched_bgr = cv2.cvtColor(stitched_rgb, cv2.COLOR_RGB2BGR)
    print(f"✅ Stitched image size: {stitched_bgr.shape[1]}×{stitched_bgr.shape[0]} px")
    return stitched_bgr

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VEGETATION DETECTION (HSV masking)
# ─────────────────────────────────────────────────────────────────────────────

def detect_vegetation(bgr_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert image to HSV colour space and create a binary mask for
    green / vegetated pixels using two HSV ranges (light and dark green).

    Returns
    -------
    mask         : binary mask (255 = vegetation, 0 = non-vegetation)
    highlighted  : original image with vegetation tinted bright green
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # ── HSV Green Ranges ──────────────────────────────────────────────────────
    # Hue channel:  green sits roughly between 35–85 in OpenCV (0-179 scale)
    # We use two ranges to capture both light and dark vegetation shades.

    # Range 1: typical lush green
    lower_green1 = np.array([35,  40,  40])
    upper_green1 = np.array([85, 255, 255])

    # Range 2: dry / olive / yellowish green (common in satellite imagery)
    lower_green2 = np.array([25,  30,  30])
    upper_green2 = np.array([35, 255, 255])

    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup: remove tiny specks, fill small holes
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Create a visual overlay: tint detected pixels bright green
    highlighted = bgr_image.copy()
    highlighted[mask == 255] = [0, 200, 0]   # BGR bright green

    return mask, highlighted

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CALCULATE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_stats(
    mask: np.ndarray,
    lat: float,
    zoom: int,
    tree_spacing_m: float
) -> dict:
    """
    From the vegetation mask compute:
      • green coverage ratio  (fraction of image that is green)
      • total image area in km²
      • green area in km²
      • estimated number of plantable trees
    """
    total_pixels = mask.size
    green_pixels = int(np.count_nonzero(mask))

    green_ratio  = green_pixels / total_pixels if total_pixels > 0 else 0.0

    # Ground resolution: metres per pixel at this zoom & latitude
    mpp          = tile_to_metres_per_pixel(lat, zoom)

    # Each pixel covers mpp² square metres
    total_area_m2 = total_pixels * (mpp ** 2)
    green_area_m2 = green_pixels * (mpp ** 2)

    # Approximate trees plantable in green/empty spaces
    # Assumes one tree per tree_spacing² square metres on average
    tree_area_m2  = tree_spacing_m ** 2
    est_trees     = int(green_area_m2 / tree_area_m2)

    return {
        "total_pixels"  : total_pixels,
        "green_pixels"  : green_pixels,
        "green_ratio"   : green_ratio,
        "mpp"           : mpp,
        "total_area_km2": total_area_m2 / 1_000_000,
        "green_area_km2": green_area_m2 / 1_000_000,
        "est_trees"     : est_trees,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def save_image(image_bgr: np.ndarray, filename: str):
    """Save a BGR OpenCV image to the outputs directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, image_bgr)
    print(f"💾 Saved: {path}")


def save_comparison(
    original_bgr : np.ndarray,
    highlighted_bgr: np.ndarray,
    mask         : np.ndarray,
    stats        : dict,
    location_name: str,
):
    """
    Create a 3-panel matplotlib figure:
      1. Original satellite image
      2. Vegetation highlighted
      3. Binary mask + stats text
    Save as comparison.png.
    """
    original_rgb    = cv2.cvtColor(original_bgr,    cv2.COLOR_BGR2RGB)
    highlighted_rgb = cv2.cvtColor(highlighted_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    titles = ["[1] Original Satellite Image",
              "[2] Detected Vegetation (Green Areas)",
              "[3] Binary Vegetation Mask"]
    images = [original_rgb, highlighted_rgb, mask]
    cmaps  = [None, None, "Greens"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")

    # Stats annotation on the third panel
    stats_text = (
        f"Location : {location_name}\n"
        f"Green Ratio   : {stats['green_ratio']*100:.1f}%\n"
        f"Green Area    : {stats['green_area_km2']:.4f} km²\n"
        f"Total Area    : {stats['total_area_km2']:.4f} km²\n"
        f"Est. Trees    : {stats['est_trees']:,}\n"
        f"Spacing Used  : {TREE_SPACING} m"
    )
    axes[2].text(
        0.02, 0.02, stats_text,
        transform=axes[2].transAxes,
        color="lime", fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
        family="monospace"
    )

    # Legend patches
    green_patch = mpatches.Patch(color="lime",  label="Vegetation")
    black_patch = mpatches.Patch(color="black", label="Non-vegetation")
    axes[2].legend(
        handles=[green_patch, black_patch],
        loc="upper right", fontsize=8,
        facecolor="#333355", edgecolor="white", labelcolor="white"
    )

    plt.suptitle(
        "Pollution Control  |  Afforestation Land Identification via Computer Vision",
        color="white", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"💾 Saved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(stats: dict, lat: float, lon: float, location_name: str):
    """Pretty-print the analysis results to the terminal."""
    bar = "═" * 60
    print(f"\n{bar}")
    print("  🌍  AFFORESTATION ANALYSIS RESULTS")
    print(bar)
    print(f"  Location        : {location_name}")
    print(f"  Coordinates     : {lat:.4f}°N, {lon:.4f}°E")
    print(f"  Ground Res.     : {stats['mpp']:.2f} m / pixel")
    print(f"  Total Pixels    : {stats['total_pixels']:,}")
    print(f"  Green Pixels    : {stats['green_pixels']:,}")
    print(f"  Green Coverage  : {stats['green_ratio']*100:.2f}%")
    print(f"  Total Area      : {stats['total_area_km2']:.4f} km²")
    print(f"  Green Area      : {stats['green_area_km2']:.4f} km²")
    print(f"  Tree Spacing    : {TREE_SPACING} m  (assumed)")
    print(f"  Est. Trees      : {stats['est_trees']:,}")
    print(bar)
    print("  Output files saved in ./outputs/")
    print(f"{bar}\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    lat          : float = DEFAULT_LAT,
    lon          : float = DEFAULT_LON,
    zoom         : int   = ZOOM,
    grid         : int   = GRID_SIZE,
    location_name: str   = "Aarey Colony, Mumbai, India"
):
    """
    End-to-end pipeline:
      1. Download satellite tiles
      2. Stitch into one image
      3. Detect vegetation
      4. Calculate statistics
      5. Save outputs
      6. Print results
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  POLLUTION CONTROL — AFFORESTATION LAND IDENTIFIER")
    print("=" * 60)
    print(f"  Target   : {location_name}")
    print(f"  Lat/Lon  : {lat}, {lon}")
    print(f"  Zoom     : {zoom}   Grid: {grid}×{grid}")
    print("=" * 60)

    # ── 1. Download & stitch ─────────────────────────────────────────────────
    satellite_bgr = download_and_stitch(lat, lon, zoom, grid)

    # ── 2. Save raw satellite image ──────────────────────────────────────────
    print("\n🖼  Saving satellite image …")
    save_image(satellite_bgr, "final_map.png")

    # ── 3. Vegetation detection ───────────────────────────────────────────────
    print("\n🔍 Detecting vegetation via HSV colour masking …")
    mask, highlighted_bgr = detect_vegetation(satellite_bgr)

    # Convert mask to 3-channel BGR for saving
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    save_image(highlighted_bgr, "green_detected.png")

    # ── 4. Calculate statistics ───────────────────────────────────────────────
    print("\n📊 Calculating coverage statistics …")
    stats = calculate_stats(mask, lat, zoom, TREE_SPACING)

    # ── 5. Comparison figure ─────────────────────────────────────────────────
    print("\n📈 Generating comparison figure …")
    save_comparison(satellite_bgr, highlighted_bgr, mask, stats, location_name)

    # ── 6. Print results ─────────────────────────────────────────────────────
    print_results(stats, lat, lon, location_name)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Change these values to analyse any location ──────────────────────────
    LATITUDE      = 19.1515          # Aarey Colony, Mumbai
    LONGITUDE     = 72.8650
    LOCATION_NAME = "Aarey Colony, Mumbai, India"

    # Uncomment for other locations:
    # LATITUDE, LONGITUDE, LOCATION_NAME = 28.6139, 77.2090, "Central Delhi, India"
    # LATITUDE, LONGITUDE, LOCATION_NAME = 12.9716, 77.5946, "Bengaluru, India"
    # LATITUDE, LONGITUDE, LOCATION_NAME = 51.5074, -0.1278, "London, UK"

    run_pipeline(
        lat           = LATITUDE,
        lon           = LONGITUDE,
        zoom          = ZOOM,
        grid          = GRID_SIZE,
        location_name = LOCATION_NAME,
    )