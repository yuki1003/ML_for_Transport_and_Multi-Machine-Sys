"""
Authors:
Group:
Description: This file will only be used for data processing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
import pyarrow.parquet as pq

#%% Importing data


pq_trips = pq.read_table("data/yellow_tripdata_2022-03.parquet")
df_trips = pq_trips.to_pandas()

# Exploring the data
print(df_trips.head())
print("Headers are: \n",df_trips.columns)


#%% removing unused columns
#vendorID, surcharge, tolls amount, 


#removing rows pandas.DataFrame.dropna



#%% convert date time



#%% plotting pick up and drop off locations
# sort by drop off zone?




#%% sorting by time of day



