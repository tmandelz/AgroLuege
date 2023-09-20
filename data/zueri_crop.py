#%%
import pandas as pd
import h5py
#%%
# Specify the path to the HDF5 file you want to read
file_path = 'D:\Temp\TorchGEO\ZueriCrop.hdf5'

# Use pandas.read_hdf() to read the HDF5 file into a DataFrame
df = pd.read_hdf(file_path)

# Now, 'df' contains your data as a DataFrame

# %%
df
# %%
df = h5py.File(file_path, "r", libver='latest', swmr=True)
# %%
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
