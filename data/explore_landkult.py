#%%
import geopandas as gpd
import pandas as pd
from shapely import wkt

#%%
df_landkult = pd.read_csv('bern_subset_landkult.csv')
df_landkult.head()

#%%
"""https://stackoverflow.com/questions/61122875/geopandas-how-to-read-a-csv-and-convert-to-a-geopandas-dataframe-with-polygons"""
df_landkult['geometry'] = df_landkult['geometry'].apply(wkt.loads)

#%%
crs = {'init': 'epsg:2056'}
gdf = gpd.GeoDataFrame(df_landkult, crs=crs, geometry='geometry')


#%%
gdf.head()

#%%
gdf.plot(column='LNF_CODE')
# %%
gdf.LNF_CODE.unique()