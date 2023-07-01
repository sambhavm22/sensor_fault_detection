# connect mongoDB dataset to python

import pandas as pd
from pymongo.mongo_client import MongoClient
from sklearn.model_selection import train_test_split

# Connect to MongoDB
client = MongoClient('mongodb+srv://sambhavm22:sambhav@cluster0.oekbqjn.mongodb.net/')

# create database name and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME="waferfault"

# read the data as a dataframe
df=pd.read_csv(r"C:\Study1\ALL_PROJECTS\wafer_fault_detection_pw\sensor-fault-detection\notebooks\wafer_23012020_041211.csv")
df=df.drop("Unnamed: 0",axis=1)

# Convert the data into json
json_record=list(json.loads(df.T.to_json()).values())

#now dump the data into the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)