import numpy as np
from tqdm import tqdm
from download_func import download_scene

coordinates = np.load("coord.npy")
taskname = "cdcsvi_nohsdp_poverty_disc_large"
i = 0
for coord in tqdm(coordinates):
    download_scene(coord[0], coord[1], 0.05, 0.05, "{}/{}".format(taskname, i))
    i += 1
