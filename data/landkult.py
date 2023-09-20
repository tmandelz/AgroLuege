#%%
import geopandas as gpd

# %%
x = gpd.read_file("D:/Temp/TorchGEO/LANDKULT/data/LANDKULT_NUTZFL.shp",rows=10000)
# %%
x.columns
#%%
x["LNF_CODE"].unique()
# %%
x.plot()
# %%
x.to_csv("bern_subset_landkult.csv")
# %%
