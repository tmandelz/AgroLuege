#%%
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np

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
#%%
a = gdf.LNF_CODE.unique()

#%%
labels_zueri = pd.read_csv('labels_zuericrop.csv')
labels_zueri_array = labels_zueri['LNF_code'].values
np.isin(a, labels_zueri_array)

# %% [markdown]
"""
# Kulturcode-Labels
Die folgenden Labels kommen nicht in Züricrop vor: 
* 725: Permakultur
* 921: Hochstamm-Feldobstbäume
* 922: Nussbäume
* 923: Kastanienbäume 
* 924: Einheimische standortgerechte Einzelbäume und Alleen
* 927: Andere Bäume

## Bäume
Die 920er Labels sind überlagernde Flächen- und Punktelemente. Das bedeutet, dass diese Nutzungsflächen entweder numerisch oder geometrisch erfasst werden. Die geometrische Nutzung wird als Polygon erfasst. Die numerischen Datenerfassung erfolgt in Kombination mit einem geometrischen Bezug, indem sie als Sachdaten der entsprechden Bewirtschaftungseinheit angehängt werden.

Aus: Minimale Geodatenmodelle, Landwirtschaftliche Bewirtschaftung, Bezugsjahr 2022, Seite 48-51 Bundesamt für Landwirtschaft

"""


# %%
