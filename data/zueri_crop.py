#%%
import pandas as pd
import h5py
#%%
# Specify the path to the HDF5 file you want to read
file_path = r'..\raw_data\ZueriCrop\ZueriCrop.hdf5'
# %%
df = h5py.File(file_path, "r", libver='latest', swmr=True)
# %%
df.keys()
# %%
data = df["data"]
# %%
data
# %%
for i in data:
    print(i)
# %%
