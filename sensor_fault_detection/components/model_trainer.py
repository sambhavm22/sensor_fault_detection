import os, sys
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.components.logger import logging
import pandas as pd
import numpy as np
from sensor_fault_detection.utils import MainUtils
from sensor_fault_detection.constant import *
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifact/model_trainer", "model.pkl")
    expected_accuracy=0.45
    model_config_file_path= os.path.join('config','model.yaml')
    
class ModelTrainer:
    def __init__(self):
        self.model_path = ModelTrainerConfig()
        self.utils = MainUtils()
        
        self.models = {
                        'XGBClassifier': XGBClassifier(),
                        'GradientBoostingClassifier' : GradientBoostingClassifier(),
                        'SVC' : SVC(),
                        'RandomForestClassifier': RandomForestClassifier()
                        }
    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            report = {} #The report dictionary is initialized to store the evaluation results for each model.
            
            for i in range(len(list(models))):
                model = list(models.values())[i] # a loop that iterates over the models. It retrieves each model object from the models dictionary using the list(models.values())[i]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report[list(models.keys())[i]] = accuracy
                
            return  report 
        
        except Exception as e:
                    raise CustomException(e, sys)
        
    def get_best_model(self, x_train:np.array, y_train: np.array, x_test:np.array, y_test: np.array):
        
        try:
            model_report: dict = evaluate_models(
                x_train =  x_train, 
                y_train = y_train, 
                x_test =  x_test, 
                y_test = y_test, 
                models = self.models)
                
            print(model_report)

            best_model_score = max(sorted(model_report.values())) # The best model score is determined by finding the maximum value from the sorted list of values in the model_report dictionary using max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] #To get the name of the best model from the model_report dictionary, the code uses list(model_report.keys())[list(model_report.values()).index(best_model_score)]. It retrieves the key (model name) corresponding to the index of the maximum score in the values list.

            best_model_object = self.models[best_model_name] #The best model object is obtained from the self.models dictionary by using the best model name (best_model_name) as the key: best_model_object = self.models[best_model_name]


            return best_model_name, best_model_object, best_model_score
                
        except Exception as e:
                raise CustomException(e, sys)  
            
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object:
        
        try:

            model_param_grid = self.utils.read_yaml_file(self.model_path.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]


            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )
            
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)
            

            return finetuned_model
        
        except Exception as e:
            raise CustomException(e,sys)          
    
    def initiate_modeL_training(self):
        try:
            logging.info(f"splitting training and testing input and target feature")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1], 
                test_array[:,:-1], 
                test_array[:,-1]
            )
            
            logging.info(f"extracting model config file path")
            
            model_report: dict = self.evaluate_models(X=X_train, y = y_train, models=self.models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = self.models[best_model_name]
            
            best_model = self.finetune_best_model(
                best_model_object = best_model,
                                best_model_name = best_model_name,
                                X_train = X_train,
                                y_train = y_train
            )
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"accuracy score: {accuracy} and best model name {best_model_name}")
            
            if accuracy< 0.5:
                raise Exception("No best model found with the accuracy score greater than the threshold 0.5")
            
            logging.info(f"Best found model on both training and testing dataset")
            
            logging.info(
                f"saving model at path: {self.model_path.trained_model_path}"
            )
                
            os.makedirs(os.path.dirname(self.model_path.trained_model_path), exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_path.trained_model_path,
                obj=best_model
            )    
            
            return self.model_path.trained_model_path
            
        except Exception as e:
            raise CustomException(e, sys)