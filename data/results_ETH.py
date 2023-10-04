# %%
import pandas as pd
import geopandas as gpd
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt


# %%
results = pd.read_csv('results_ETH.txt', delim_whitespace=True, names=['raw_ENG', 'Dataset_perc', 'Accuracy'])
results.head()

# %%
results['Dataset_perc'] = results['Dataset_perc'].replace('\(', '', regex=True)
results['Dataset_perc'] = results['Dataset_perc'].replace('\<', '', regex=True)
results['Dataset_perc'] = results['Dataset_perc'].replace('%\)', '', regex=True)

# %%
results['Dataset_perc'] = results['Dataset_perc'].astype('float')

# %%
results.plot.scatter(x='Accuracy', y='Dataset_perc', title='Accuracy and portion of dataset')

# %%
results.plot.scatter(x='Accuracy', y='Dataset_perc', ylim=(0,1), xlim=(0,100),title='Accuracy and portion of dataset')

# %%
results.plot.hist(column='Accuracy', bins=50, xlim=(0,60), title='Accuracy of the ETH-Modell')

# %%
results.plot.hist(column='Dataset_perc', bins=50, xlim=(0,12), title='Portion of the Züricrop-Dataset')

# %% [markdown]
"""
Generate lists for different benchmarks to plot and evaluate
"""
# %%
labels= pd.read_csv('labels_zuericrop.csv')

# %%
labels = labels[['LNF_code', '4th_tier_ENG']]
labels.columns = ['LNF_code', 'raw_ENG']
labels = labels.dropna(subset = ['raw_ENG'])

# %%
results_code = pd.merge(results, labels, how='left', on='raw_ENG')
results_code.LNF_code.isna().sum()

# %%
results_code.LNF_code.iloc[44] = 569
results_code.LNF_code.iloc[58] = 555
results_code.LNF_code.iloc[25] = 522
results_code.LNF_code.iloc[38] = 536
# %%
acc60_per02= results_code[(results_code['Accuracy'] < 60) & (results_code['Dataset_perc'] < 0.2)].LNF_code.to_list()
acc60_per01= results_code[(results_code['Accuracy'] < 60) & (results_code['Dataset_perc'] < 0.1)].LNF_code.to_list()
acc40_per01= results_code[(results_code['Accuracy'] < 40) & (results_code['Dataset_perc'] < 0.1)].LNF_code.to_list()
acc20_per01= results_code[(results_code['Accuracy'] < 20) & (results_code['Dataset_perc'] < 0.1)].LNF_code.to_list()

# %%
df_landkult = pd.read_csv('bern_landkult.csv')
df_landkult['geometry'] = df_landkult['geometry'].apply(wkt.loads)
crs = {'init': 'epsg:2056'}
gdf = gpd.GeoDataFrame(df_landkult, crs=crs, geometry='geometry')

# %%
fig, ax = plt.subplots(figsize=(10, 10))
gdf.where(gdf['LNF_CODE'].isin(acc60_per02)).plot(column='LNF_CODE', figsize=(50, 40), ax=ax, legend=True)

ax.set_xlim(2.55e6, 2.65e6)
ax.set_ylim(1.16e6, 1.24e6)
ax.set_title('Area of Labels with acc<60% & Data_perc<0.02%')

print(gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc60_per02)).sum()/1000000)
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc60_per02)).sum()/gdf.Shape_Area.sum()*100

# %%
print(gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc60_per01)).sum()/1000000)
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc60_per01)).sum()/gdf.Shape_Area.sum()*100
# %%
print(gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc40_per01)).sum()/1000000)
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc40_per01)).sum()/gdf.Shape_Area.sum()*100
# %%
print(gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc20_per01)).sum()/1000000)
gdf.Shape_Area.where(gdf['LNF_CODE'].isin(acc20_per01)).sum()/gdf.Shape_Area.sum()*100
# %%
fig, ax = plt.subplots(figsize=(10, 10))
gdf.where(gdf['LNF_CODE'].isin(acc20_per01)).plot(column='LNF_CODE', figsize=(50, 40), ax=ax, legend=True)

ax.set_xlim(2.55e6, 2.65e6)
ax.set_ylim(1.16e6, 1.24e6)
# %%
