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


df_trips_head= df_trips.head()

# Exploring the data
print(df_trips.head())
print("Headers are: \n",df_trips.columns)


#%% removing unused columns
#vendorID, surcharge, tolls amount, 


# Remove columns (MTA tax, improvement_surcharge, VendorID, Store_and_fwd_flag, payment_type, Congestion_Surcharge)
df_trips.drop(columns=['VendorID', 'store_and_fwd_flag',
                       'payment_type','extra',
                       'mta_tax', 'tolls_amount', 'improvement_surcharge',
                       'congestion_surcharge', 'airport_fee'], inplace=True)
print(df_trips.shape[0])

# Removing empty rows pandas.DataFrame.dropna
df_trips.dropna(axis=0, how='any', inplace=True)
print(df_trips.shape[0])

#pickup and drop-off date within month


#location zone ID should be in range [1,263]
df_trips = df_trips[(df_trips['PULocationID'] >= 1) &
                    (df_trips['PULocationID'] <= 263) & 
                    (df_trips['DOLocationID'] >= 1) &
                    (df_trips['DOLocationID'] <= 263)]

#passenger count should be range [1,6]
df_trips = df_trips[(df_trips['passenger_count'] >= 1) |
                    (df_trips['passenger_count'] <= 6)]

#trip distance not equal to 0 and higher that 100 miles
df_trips = df_trips[(df_trips['trip_distance'] >=0) |
                    (df_trips['passenger_count'] <= 100)]
print(df_trips.shape[0])

#fare amount should be at least $2,50

##tip percentage maximum? 50%?

#trip duration minimum 1 minute? max 3 hours?

# Convert all strings to floats

# Convert time schemes

# add column for trip duration: Requires calculation

# Groupby time of day (rush hour), weekday vs. weekend vs. holiday

# identify pick-up and dropoff borough (use taxizonelookup.excel sheet)

# tip percentage

#%% convert date time



#%% plotting pick up and drop off locations
# sort by drop off zone?




#%% sorting by time of day



