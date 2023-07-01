`import os, sys
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.components.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pymongo import MongoClient
from sensor_fault_detection.constant import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sensor_fault_detection.components.data_ingestion import DataIngestion, DataIngestionConfig
from sensor_fault_detection.utils import MainUtils


@dataclass
class DataTransformationConfig:
    """
    Data Transformation Config
    """
    preprocessor_file_path = os.path.join("artifact/data_transformation", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    @staticmethod    
    def get_data(raw_data_path:str)->pd.DataFrame:
        """
        Method Name :   get_data
        Description :   This method reads all the validated raw data from the feature_store_file_path and returns a pandas DataFrame containing the merged data. 
        
        Output      :   a pandas DataFrame containing the merged data 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            logging.info("Reading raw data from feature store")
            data = pd.read_csv(raw_data_path)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            
        except Exception as e:
            raise CustomException(e, sys)    
        
    def get_data_transformer_obj(self):
        
        try:
            # define the steps for the preprocessor pipeline
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                imputer_step,
                scaler_step
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)      
        
    def initiate_data_transformation(self):
        """
            Method Name :   initiate_data_transformation
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )

        try:
            dataframe = self.get_data(raw_data_path=self.data_ingestion_config.raw_data_path)
            X = dataframe.drop(columns=TARGET_COLUMN, axis=1)
            y = dataframe[TARGET_COLUMN]
            
            X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            preprocessor = self.get_data_transformer_obj()
            
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            
            processor_path = self.data_transformation_config.preprocessor_file_path
            os.makedirs(os.path.join(processor_path), exist_ok=True)
            self.save_object(file_path = processor_path, obj = preprocessor)
            
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]
            
            return train_arr, test_arr, processor_path
            
            
        except Exception as e:
            raise CustomException(e, sys)
            
`