import h5py
import numpy as np
from tqdm import tqdm

with h5py.File("./data/RAISE_LPBF_train.hdf5") as h5f:
    candidate_layers = list(h5f["C030"].keys())[100:-8]
    selected_layers = np.random.choice(
        candidate_layers, len(candidate_layers) // 10, replace=False
    )
    with h5py.File(
        f"./data/RAISE_LPBF_train_sampled.hdf5", "w"
    ) as h5f_sampled:
        for layer in tqdm(
            selected_layers, desc="Sampling layers", total=len(selected_layers)
        ):
            # dump in a new HDF5 file
            h5f_sampled.create_group(f"C030/{layer}")
            for k in h5f["C030"][layer].keys():
                h5f_sampled[f"C030/{layer}"].create_dataset(
                    k, data=h5f[f"C030/{layer}/{k}"][:]
                )
