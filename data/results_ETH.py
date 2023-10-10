# %%
import pandas as pd
import geopandas as gpd
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
import numpy as np



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
gdf = gdf.to_crs(4326)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
gdf.where(gdf['LNF_CODE'].isin(acc60_per02)).plot(column='LNF_CODE', figsize=(50, 40), ax=ax, legend=True)

#ax.set_xlim(2.55e6, 2.65e6)
#ax.set_ylim(1.16e6, 1.24e6)
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
gdf60_20 = gdf[gdf['LNF_CODE'].isin(acc60_per02)]
gdf60_20['LNF_CODE'] = gdf60_20.LNF_CODE.astype('category').copy()

# %%
gdf60_20.LNF_CODE.unique()
# %%
gdf['LNF_CODE'] = gdf.LNF_CODE.astype('category')
gdf.plot(column='LNF_CODE', legend=True)
# %%
fig, ax = plt.subplots(figsize=(10, 10))
gdf60_20.plot(column='LNF_CODE', figsize=(50, 40), ax=ax, legend=True)

ax.set_xlim(2.60e6, 2.61e6)
ax.set_ylim(1.16e6, 1.17e6)

# %%
def sum_quadrant(geo_df, step):

    list_x = list(np.arange(7.0, 8.0 +step, step))
    list_y = list(np.arange(46.4, 47.4 +step, step))
      
    
    results = []
    
    for i in range(len(list_x)-1):        
        for j in range(len(list_y)-1):   
                     
            sum_value =  geo_df.cx[list_x[i]:list_x[i+1],list_y[j]:list_y[j+1]].Shape_Area.sum()/1000
            result_entry = {'i': list_x[i], 'j': list_y[j], 'sum': sum_value}
            results.append(result_entry)
    results = pd.DataFrame(results)
    return results
    
            
# %%
plot60_20 = sum_quadrant(gdf60_20, 0.024)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x='i', y='j', s='sum', data=plot60_20, alpha=0.5)
ax.grid(True)
ax.set_title('sum of Area(<60% accuracy % < 0.02% of Dataset ZüriCrop)')
plt.show()
# %%
gdf20_10 = gdf[gdf['LNF_CODE'].isin(acc20_per01)]
gdf20_10['LNF_CODE'] = gdf20_10.LNF_CODE.astype('category').copy()
plot20_10 = sum_quadrant(gdf20_10, 0.024)
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(x='i', y='j', s='sum', data=plot20_10, alpha=0.5)
ax.grid(True)
ax.set_title('sum of Area(<20% accuracy % < 0.01% of Dataset ZüriCrop)')
plt.show()
# %%

# %%
