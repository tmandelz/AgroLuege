#%%
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np

#%%
df_landkult = pd.read_csv('bern_subset_landkult.csv')
df_landkult.head()

#%%
labels_zueri = pd.read_csv('labels_zuericrop.csv')
labels_zueri.head()

#%%
"""https://stackoverflow.com/questions/61122875/geopandas-how-to-read-a-csv-and-convert-to-a-geopandas-dataframe-with-polygons"""
df_landkult['geometry'] = df_landkult['geometry'].apply(wkt.loads)
crs = {'init': 'epsg:2056'}
gdf = gpd.GeoDataFrame(df_landkult, crs=crs, geometry='geometry')

#%%
gdf.head()

#%%
gdf.plot(column='LNF_CODE')

#%% [markdown]
"""
Der Kanton bern hat eine Fläche von 5'960 km2, die Summe der Agrarflächen, die gemäss Dokumentation im m2 angegeben ist, beträgt:
"""
#%%
gdf.Shape_Area.sum()

#%%
mask = df_landkult['LNF_CODE'].isin(labels_zueri['LNF_code'])
no_labels = df_landkult[~mask].LNF_CODE.unique()

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

## Permakultur
Die Permakultur gilt als Spezialkultur. Sie ist Beitragsberechtigt. Wieso diese nicht berücksichtigt wurde im Züricrop, ist reine Hypothese.
Wir gehen davon aus, dass dies mit der sehr unterschiedlich ausgeprägten Erstellung von Permakulturflächen zusammenhängt.

Aus: Minimale Geodatenmodelle, Landwirtschaftliche Bewirtschaftung, Bezugsjahr 2022, Seite 48-51 Bundesamt für Landwirtschaft

"""

#%% 
# Die Fläche, welche keine für uns verwertbaren Kulturcode-Labels aufweist, beträgt [km2]:
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(no_labels)).sum()/1000000

# %%
# Dies entspricht folgendem Flächenprozentsatz des Datensatzes vom Kanton Bern [%]: 
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(no_labels)).sum()/gdf.Shape_Area.sum()*100

# %%
gdf.where(gdf['LNF_CODE'].isin(no_labels)).plot(column='LNF_CODE')
# %%
