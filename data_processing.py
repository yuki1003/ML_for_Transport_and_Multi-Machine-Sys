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

#only include RatecodeID 1 and 2, standard rate and JFK (get rid of discounted trips, etc)
df_trips = df_trips[(df_trips['RatecodeID'] == 1) | (df_trips['RatecodeID'] == 2)]

#location zone ID should be in range [1,263], or do we need to exclude zones that are not in the yellow zone?
df_trips = df_trips[(df_trips['PULocationID'] >= 1) & (df_trips['PULocationID'] <= 263) & 
                    (df_trips['DOLocationID'] >= 1) & (df_trips['DOLocationID'] <= 263)]

#passenger count should be range [1,6]
df_trips = df_trips[(df_trips['passenger_count'] >= 1) & (df_trips['passenger_count'] <= 6)]

#trip distance not equal to 0 and higher that 100 miles
df_trips = df_trips[(df_trips['trip_distance'] > 0) & (df_trips['trip_distance'] <= 100)]

#fare amount should be at least $2,50
df_trips = df_trips[(df_trips['fare_amount'] >= 2.50) & (df_trips['fare_amount'] <= 300.00)]

#tip amount should be greater than or equal to 0 , see tip percentage to include a maximum
df_trips = df_trips[(df_trips['tip_amount'] >= 0)]

#calculate tip percentage, set tip precentage in range [0,50] percent
df_trips['tip_percent'] = (df_trips.tip_amount / df_trips.total_amount) * 100
df_trips = df_trips[(df_trips['tip_percent'] >= 0) & (df_trips['tip_percent'] < 50)]


#%% Converting times and calculating trip duration

#converting columns to datetime
df_trips['tpep_pickup_datetime']=pd.to_datetime(df_trips['tpep_pickup_datetime'])
df_trips['tpep_dropoff_datetime']=pd.to_datetime(df_trips['tpep_dropoff_datetime'])

#seperate pickup and dropoff days
df_trips['pickup_date'] = df_trips['tpep_pickup_datetime'].dt.date.tolist()
df_trips['dropoff_date'] = df_trips['tpep_dropoff_datetime'].dt.date.tolist()

#CHECK dates are all within the month march 03
#df_trips = df_trips[(df_trips['pickup_date'])[7] == 3]

#adding columns with only pick up and drop off time, no date
df_trips['pickup_time']=df_trips['tpep_pickup_datetime'].dt.time
df_trips['dropoff_time']=df_trips['tpep_dropoff_datetime'].dt.time

#calculate trip duration in seconds and minutes
df_trips['trip_duration_seconds'] = (df_trips['tpep_dropoff_datetime']-df_trips['tpep_pickup_datetime']).astype('timedelta64[s]')
df_trips['trip_duration_minutes'] = (df_trips['tpep_dropoff_datetime']-df_trips['tpep_pickup_datetime']).astype('timedelta64[m]')

#trip duration in range [1min,2hour] min, processed in seconds
df_trips = df_trips[(df_trips['trip_duration_seconds'] >= 60) &
                    (df_trips['trip_duration_seconds'] <= 7200)] 

#split weekdays and weekend days for anaylsis
def day_week(x):
    if x in range(0,4):
        return 'Weekday'
    else:
        return 'Weekend'

df_trips['pickup_day_no'] = df_trips['tpep_dropoff_datetime'].dt.weekday
df_trips['day_of_week'] = (df_trips['tpep_pickup_datetime'].dt.weekday).apply(day_week)


#%% Add pick-up and dropoff borough name using taxizonelookup.excel sheet)
def zone_conversion(id):
    i = id-1
    return df_borough.Zone[i]

df_trips['pickup_zone'] = df_trips['PULocationID'].apply(zone_conversion)
df_trips['dropoff_zone'] = df_trips['DOLocationID'].apply(zone_conversion)


#%% Exploring and checking the data
df_head = df_trips.head(n=1000)
df_describe = df_trips.describe()
df_type = df_trips.dtypes
df_zeros = df_trips.isnull().sum()  
'''
for column_name in df_trips.columns:
    column = df_trips[column_name]
    count = (column == 0).sum()
    print('Count of zeros in', column_name, 'is :', count)

for column_name in df_trips.columns:
    column = df_trips[column_name]
    count = (column == 0).sum()
    print('Count of zeros in', column_name, 'is :', count)
'''

#print(df_head)
print("Columns are: \n",df_trips.columns)
#print("Amount of zeros per column: \n",df_zeros)
#print(df_trips.shape[0])


#%% Load new dataframe as new csv/parquet file to be used in data analysis and ML methods
outputFileName = 'yellow_tripdata_processed'
df_trips.to_parquet("data/{}.parquet".format(outputFileName)) #UNCOMMENT this to load file
#df_trips.to_csv("data/{}.csv".format(outputFileName))

