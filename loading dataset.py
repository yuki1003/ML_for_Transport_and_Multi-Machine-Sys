# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:29:14 2022

@author: cypho
"""
test
import pyarrow.parquet as pq
import pandas as pd

trips = pq.read_table('yellow_tripdata_2022-03.parquet')
trips = trips.to_pandas()

