

#%%
from dataset import Dataset
traindataset= Dataset(r"raw_data\ZueriCrop\ZueriCrop.hdf5", 0., 'train', False, 1, "labels.csv", num_channel=4, apply_cloud_masking=False,small_train_set_mode=True)
# %%
import torch.utils.data

traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)

# %%
for iteration, data in enumerate(traindataloader):
    print(len(data))
    print(data[0].shape)
    print(data[0])
    print(data[1].shape)
    print(data[1])
    print(data[2].shape)
    print(data[2])
    print(data[3].shape)
    print(data[3])

# %%
