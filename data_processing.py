"""
Authors:
Group:
Description: This file will only be used for data processing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

#%% Importing data


pq_trips = pq.read_table("data/yellow_tripdata_2022-03.parquet")
df_trips = pq_trips.to_pandas()

# Exploring the data
print(df_trips.head())
print("Headers are: \n",df_trips.columns)


#%%  sorting by time of day



#%% 



