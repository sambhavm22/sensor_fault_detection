import os, sys
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.components.logger import logging
import pickle
import yaml
import pandas as pd 
import numpy as np
from sensor_fault_detection.constant import *
from typing import Dict, Tuple
import boto3


class MainUtils:
    def __init__(self)-> None:
        pass
    
    def read_yaml_file(self, filename:str)->dict:
        try:
            with open(filename, 'rb') as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config

        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def save_object(file_path:str, obj:object):
    
        try:
            logging.info("Entered the load_object method of MainUtils class")
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
            logging.info("Exited the save_object method of MainUtils class")   
            
        except Exception as e:
            raise CustomException(e, sys)  
    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)

            logging.info("Exited the load_object method of MainUtils class")

            return obj

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod     
    def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return pickle.load(file_obj)
        except Exception as e:
            logging.info('Exception Occured in load_object function utils')
            raise CustomException(e,sys)
                
                