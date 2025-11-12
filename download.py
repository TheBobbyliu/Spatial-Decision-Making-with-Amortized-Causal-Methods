import os
os.environ['SH_CLIENT_ID']='906e4c79-dc4c-45d7-8b75-a27076ef8429'
os.environ['SH_CLIENT_SECRET']='a2MLWnqx1LlLEsehxYbD8by1pK4TKme4'
os.environ['SH_INSTANCE_ID']='d941453d-a41f-4e5f-bc8a-f96dadc0e457'
import os, numpy as np, matplotlib.pyplot as plt
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection, MimeType, SentinelHubRequest,
    bbox_to_dimensions, MosaickingOrder
)

# ---- 0) Credentials sanity check ----
def get_cfg():
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv('SH_CLIENT_ID')
    cfg.sh_client_secret = os.getenv('SH_CLIENT_SECRET')
    # instance_id optional for new OAuth flows; set if you have one:
    if os.getenv('SH_INSTANCE_ID'):
        cfg.instance_id = os.getenv('SH_INSTANCE_ID')
    print("ID ok?", bool(cfg.sh_client_id), "SECRET ok?", bool(cfg.sh_client_secret))
    return cfg

cfg = get_cfg()

# ---- 1) Define AOI (lon, lat) and a safe bbox ----
lat, lon = 34.2117433, -101.7172438  # your point near Wasco, CA
# lat, lon = -1.18669817,  0.42572741
deg = 0.01  # ~1 km+ each side at this latitude; increase if needed
bbox = BBox(((lon - deg, lat - deg), (lon + deg, lat + deg)), crs=CRS.WGS84)

# Use a robust pixel size rather than super-fine dims for debugging
size = bbox_to_dimensions(bbox, resolution=10)  # 10 m/pixel
print("Requested image size:", size)

# ---- 2) Evalscript: RGB + dataMask (float) ----
EVALSCRIPT_RGB = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B03","B02","SCL","dataMask"],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.B04, s.B03, s.B02, s.SCL, s.dataMask];
}
"""

# ---- 3) Request with a wide time window and least-cloud mosaicking ----
req = SentinelHubRequest(
    evalscript=EVALSCRIPT_RGB,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=("2024-04-01", "2024-07-31"),
            mosaicking_order=MosaickingOrder.LEAST_CC
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=size,
    data_folder="downloads",
    config=cfg
)

print("Fetching data (L2A, least cloud cover)...")
arr = req.get_data()[0]  # shape: H x W x 5  (R,G,B,SCL,dataMask)
rgb = arr[..., :3]
scl = arr[..., 3]
mask = arr[..., 4] > 0.5  # valid pixels

print("Valid pixels:", mask.sum(), "of", mask.size)
print("RGB min/max (raw):", float(rgb.min()), float(rgb.max()))

# ---- 4) If no valid pixels, fallback to L1C and/or widen time window ----
if mask.sum() == 0:
    print("No valid L2A pixels; retrying with L1C and a wider window...")
    req_L1C = SentinelHubRequest(
        evalscript=EVALSCRIPT_RGB.replace('"B04","B03","B02","SCL","dataMask"',
                                          '"B04","B03","B02","dataMask"'),  # SCL not in L1C
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2024-01-01","2024-12-31"),
            mosaicking_order=MosaickingOrder.LEAST_CC
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, data_folder="downloads", config=cfg
    )
    arr = req_L1C.get_data()[0]
    rgb = arr[..., :3]
    mask = arr[..., -1] > 0.5
    print("Valid pixels (L1C):", mask.sum(), "of", mask.size)

# ---- 5) Visualize with safe percentile stretch on valid pixels ----
def show(img, m, title):
    if m.sum() == 0:
        plt.figure(); plt.title(title + " (NO VALID PIXELS)"); plt.axis('off'); plt.show(); return
    x = img.copy()
    x[~m] = np.nan
    # percentile stretch ignoring NaNs
    lo, hi = np.nanpercentile(x, [2, 98])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    plt.figure(); plt.imshow(x); plt.title(title); plt.axis('off'); plt.show()

show(rgb, mask, "Sentinel-2 RGB (stretched, valid pixels only)")

# ---- 6) Optional quick SCL diagnostic (if L2A) ----
if arr.shape[-1] == 5:  # we have SCL
    plt.figure(); plt.imshow(np.where(mask, scl, np.nan)); plt.title("SCL (L2A)"); plt.axis('off'); plt.show()
