import numpy as np
import rasterio
import matplotlib.pyplot as plt
def preprocess(img):
    x = img.copy()
    # percentile stretch ignoring NaNs
    lo, hi = np.nanpercentile(x, [2, 98])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    x *= 255
    x = np.int64(x)
    x = np.moveaxis(x, 0, -1)
    return x

taskname = 'cdcsvi_nohsdp_poverty_disc'
src = rasterio.open("{}/{}.tif".format(taskname, 0)).read()

dst = preprocess(src)
# (height, width, depth:3)
# 0-255
plt.imshow(dst)