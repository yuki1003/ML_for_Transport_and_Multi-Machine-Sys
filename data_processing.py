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
import seaborn as sb


#%% Importing data

pq_trips = pq.read_table("data/yellow_tripdata_2022-03.parquet")
df_trips = pq_trips.to_pandas()

df_borough = pd.read_csv('data/taxi+_zone_lookup.csv')


#%% Removing unwanted and unrealistic data

# Remove unused columns (MTA tax, improvement_surcharge, VendorID, Store_and_fwd_flag, payment_type, Congestion_Surcharge)
df_trips.drop(columns=['VendorID', 'store_and_fwd_flag', 'payment_type', 'extra', 'mta_tax', 'tolls_amount', 
                       'improvement_surcharge', 'congestion_surcharge', 'airport_fee'], inplace=True)

# Removing empty rows pandas.DataFrame.dropna
#df_trips.dropna(axis=0, how='any', inplace=True)  #is this correct? better to remove zeros in individual columns, 
#tipping amount may be zero

#pickup and drop-off date within month
#object type: datetime64[ns]

#location zone ID should be in range [1,263], or do we need to exclude zones that are not in the yellow zone?
df_trips = df_trips[(df_trips['PULocationID'] >= 1) & (df_trips['PULocationID'] <= 263) & 
                    (df_trips['DOLocationID'] >= 1) & (df_trips['DOLocationID'] <= 263)]

#passenger count should be range [1,6]
df_trips = df_trips[(df_trips['passenger_count'] >= 1) & (df_trips['passenger_count'] <= 6)]

#trip distance not equal to 0 and higher that 100 miles
df_trips = df_trips[(df_trips['trip_distance'] >= 0) & (df_trips['trip_distance'] <= 100)]

#fare amount should be at least $2,00
df_trips = df_trips[(df_trips['fare_amount'] >= 2.00) & (df_trips['fare_amount'] <= 300.00)]

#tip amount should be greater than or equal to 0 , see tip percentage to include a maximum
df_trips = df_trips[(df_trips['tip_amount'] >= 0)]

#calculate tip percentage
df_trips['tip_percent'] = (df_trips.tip_amount / df_trips.total_amount) * 100
df_trips = df_trips[(df_trips['tip_percent'] >= 0) & (df_trips['tip_percent'] < 50)]


#%% Identify pick-up and dropoff borough (use taxizonelookup.excel sheet)
def zone_conversion(id):
    i = id-1
    return df_borough.Zone[i]

df_trips['PUL_zone'] = df_trips['PULocationID'].apply(zone_conversion)
df_trips['DOL_zone'] = df_trips['DOLocationID'].apply(zone_conversion)


#%% Exploring the data
df_head = df_trips.head(n=20)
df_describe = df_trips.describe()
df_type = df_trips.dtypes

#print(df_head)
print("Columns are: \n",df_trips.columns)
#print(df_describe)
#print(df_type)
#print(df_trips.shape[0])



#%% Converting times and calculating trip duration

#trip duration minimum 1 minute? max 3 hours? [DAPHNE]

# Convert time schemes [DAPHNE]

# add column for trip duration: Requires calculation [DAPHNE]

# Groupby time of day (rush hour), weekday vs. weekend vs. holiday [DAPHNE]




#%% Plotting pick up and drop off locations
# sort by drop off zone?



#%% sorting by time of day
