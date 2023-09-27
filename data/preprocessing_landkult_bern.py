# %%
import geopandas as gpd

# %%
x = gpd.read_file("../raw_data/LANDKULT/data/LANDKULT_NUTZFL.shp")
# %%
x.columns
# %%
x["LNF_CODE"].unique()
# %%
x.plot()
# %%
x.to_csv("bern_landkult.csv")
