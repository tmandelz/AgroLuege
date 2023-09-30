

#%%
from dataset import Dataset
import torch.utils.data
import matplotlib.pyplot as plt

#%%
traindataset= Dataset(r"raw_data\ZueriCrop\ZueriCrop.hdf5", 0., 'train', False, 1, "labels.csv", num_channel=4, apply_cloud_masking=False,small_train_set_mode=True)
traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=0)

# %%
for iteration, data in enumerate(next(iter(traindataloader))):
    data_i = data
    # print(data[0].shape)
    # print(data[0])
    # print(data[1].shape)
    # print(data[1])
    # print(data[2].shape)
    # print(data[2])
    # print(data[3].shape)
    # print(data[3])

# %%
# Create a heatmap plot
plt.imshow(data_i[0], cmap='viridis', interpolation='nearest')
plt.colorbar()  # Add a color bar on the right to indicate values
plt.title('Heatmap of a 24x24 Field')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

# %%
