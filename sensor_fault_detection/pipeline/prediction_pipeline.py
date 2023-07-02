import os, sys
import pickle
from blinker.base import P
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
import shutil

from sensor_fault_detection.components.logger import logging
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.utils import MainUtils

from sensor_fault_detection.components.data_ingestion import DataIngestion
from sensor_fault_detection.components.data_transformation import DataTransformation
from sensor_fault_detection.components.model_trainer import ModelTrainer
from sensor_fault_detection.constant import *

from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    model_file_path:str = os.path.join("artifact/data_transformation", "model.pkl")
    preprocessor_file_path:str = os.path.join("artifact/data_transformation", "preprocessor.pkl")
    prediction_file_name:str = os.path.join(prediction_output_dirname, prediction_file_name)
    
    
class PredictionPipeline:
    def __init__(self, request: request):
        
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()
        
        
    def save_input_file(self)->str:
        
        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input data frame
            On Failure  :   Write an exception log and then raise an exception
            
        """
        
        try:
            pred_file_input_dir = "artifact/predictions"
            os.makedirs(pred_file_input_dir, exist_ok=True)
            
            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            
            input_csv_file.save(pred_file_path)
            
            return pred_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, features):
        
        try:
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_file_path)

            transformed_x = preprocessor.transform(features)

            preds = model.predict(transformed_x)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)  
        
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        """
            Method Name :   get_predicted_data frame
            Description :   this method returns the data frame with a new column containing predictions

            
            Output      :   predicted data frame
            On Failure  :   Write an exception log and then raise an exception
        """       
        
        try:
            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe

            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad', 1:'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_name, index= False)
            logging.info("predictions completed. ")
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_file()
            self.get_predicted_dataframe(input_csv_path)
            
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)    