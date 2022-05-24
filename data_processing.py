"""
Authors:
Group:
Description: This file will only be used for data processing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

pq_trips = pq.read_table("yellow_tripdata_2022-03.parquet")
df_trips = pq_trips.to_pandas()

# Exploring the data
print(df_trips.head())
print("Headers are: \n",df_trips.columns)