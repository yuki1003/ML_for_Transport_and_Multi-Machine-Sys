#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sb
import geopandas as gpd
import matplotlib.colors as colors
import warnings


# In[13]:


#import data
pq_taxi = pq.read_table("yellow_tripdata_processed.parquet")
df_taxi = pq_taxi.to_pandas()


# In[21]:


plt.style.use('ggplot')
warnings.filterwarnings("ignore")


# In[7]:


df_taxi.head()


# In[16]:


sf = gpd.read_file('taxi_zones/taxi_zones.shp')
zone = pd.read_csv("taxi+_zone_lookup.csv")
sf['geometry'] = sf['geometry'].to_crs('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


# In[26]:


# Compute pickup and dropoff amount of each zone
pu_zone = df_taxi.groupby(['PULocationID'], as_index=False).size()
do_zone = df_taxi.groupby(['DOLocationID'], as_index=False).size()
pu_zone = gpd.GeoDataFrame(pd.merge(pu_zone, sf, left_on='PULocationID', right_on='LocationID')).drop('LocationID', axis=1)
do_zone = gpd.GeoDataFrame(pd.merge(do_zone, sf, left_on='DOLocationID', right_on='LocationID')).drop('LocationID', axis=1)
pu_zone = pu_zone.to_crs({'init': 'epsg:3857'})
do_zone = do_zone.to_crs({'init': 'epsg:3857'})


# In[11]:


# plot geospatial visualisation to compare pickup and dropoff amount from each zone
fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.09, right=0.8, wspace=0.02, hspace=0.02)
# create a normalized colorbar
vmin, vmax = pu_zone['size'].min(), pu_zone['size'].max()
axs[0] = pu_zone.plot(column='size', linewidth=0.09, edgecolor='k', figsize=(10, 10), 
    norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='Blues', legend=False, ax=axs[0]) 
ctx.add_basemap(axs[0])
vmin, vmax = do_zone['size'].min(), do_zone['size'].max()
axs[1] = do_zone.plot(column='size', linewidth=0.1, edgecolor='k', figsize=(10, 10), 
    norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='Blues', legend=False, ax=axs[1])
ctx.add_basemap(axs[1])
axs[0].set_title('Pickup')
axs[1].set_title('Dropoff')
axs[0].set_axis_off()
axs[1].set_axis_off()
# draw the color bar
patch_col = axs[0].collections[0]
cb = fig.colorbar(patch_col, ax=axs, shrink=0.72, orientation="vertical", pad=0.005)
cb.ax.set_ylabel('Trip Amount')
plt.show()


# In[ ]:




