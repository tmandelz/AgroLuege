#%%
import h5py
from torchgeo.datasets import ZueriCrop

# %%
def get_ZueriCrop(file_path):
    ZueriCrop_data = ZueriCrop(root=file_path,download=True)
    return ZueriCrop_data

# %%
file_path = r'..\raw_data\ZueriCrop\ZueriCrop.hdf5'
get_ZueriCrop(file_path)
# %%
df = h5py.File(file_path, "r", libver='latest', swmr=True)
# %%
df.keys()
# %%
data = df["data"]
# %%
for i in data[0:10]:
    print(i)
# %%
