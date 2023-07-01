import os, sys
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.components.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pymongo import MongoClient
from sensor_fault_detection.constant import *

@dataclass
class DataIngestionConfig:
    
    raw_data_path:str = os.path.join("artifact/data_ingestion", "raw_data.csv")
    train_data_path = os.path.join("artifact/data_ingestion", "train_data.csv")
    test_data_path = os.path.join("artifact/data_ingestion", "test_data.csv")
    
class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def export_collection_as_data_frame(self, collection_name, db_name):
        try:
            
            mongo_client = MongoClient(MONGO_DB_URL)
            db = mongo_client[db_name]
            collection = db[collection_name]
            df = pd.DataFrame(list(collection.find()))
            
            if "_id"in df.columns.tolist():
                df = df.drop(columns=["_id"], axis=1)
            
            df = df.replace("na", np.nan, inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def export_data_into_feature_store_file_path(self)->pd.DataFrame:
        
        #This function reads data from mongodb and saves it into artifacts. 
        try:
            logging.info(f"exporting data from mongoDB")
            raw_file_path = self.data_ingestion_config.raw_data_path
            os.makedirs(raw_file_path, exist_ok=True)
            
            sensor_data = self.export_collection_as_data_frame(
                collection_name= MONGO_COLLECTION_NAME,
                db_name = MONGO_DATABASE_NAME
                )
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self):

        logging.info("Initiating data ingestion")
        
        try:
            logging.info("data reading using pandas library from local system")
            sensor_data = pd.read_csv(os.path.join("notebook/wafer_23012020_041211.csv"))
            logging.info("data reading completed")      
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True) #make directory
            sensor_data.to_csv(self.data_ingestion_config.raw_data_path, index=False) #save to .csv file
            
            logging.info("splitting data into train and test")
            train_data, test_data = train_test_split(sensor_data, test_size=0.2, random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logging.info("data ingestion completed")
            
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)   
    