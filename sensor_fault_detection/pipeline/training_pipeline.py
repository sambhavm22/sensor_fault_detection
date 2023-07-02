import os, sys
from sensor_fault_detection.components.logger import logging
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.utils import MainUtils

from sensor_fault_detection.components.data_ingestion import DataIngestion
from sensor_fault_detection.components.data_transformation import DataTransformation
from sensor_fault_detection.components.model_trainer import ModelTrainer

class TrainingPipeline:
    
    def start_data_ingestion(self):
        try:
            
            data_ingestion = DataIngestion()
            raw_file_path = data_ingestion.initiate_data_ingestion()
            return raw_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
                
    def start_data_transformation(self):
        try:
            
            data_transformation = DataTransformation(raw_file_path=raw_file_path) 
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr, preprocessor_path
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_modeL_training(train_arr, test_arr)
            return model_score
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_pipeline(self):
        try:
            raw_file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(raw_file_path)
            model_score = self.start_model_training(train_arr, test_arr)
            
            print("training completed. Trained model score", model_score )

        except Exception as e:
            raise CustomException(e, sys)