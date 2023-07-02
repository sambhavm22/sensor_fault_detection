import sys
from flask import Flask, render_template, request, jsonify, send_file
from sensor_fault_detection.components.exception import CustomException
from sensor_fault_detection.utils import MainUtils
from sensor_fault_detection.components.logger import logging

from sensor_fault_detection.pipeline import training_pipeline, prediction_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "welcome to yhe application"

@app.route("/train")
def train_route():
    try:
        train_pipeline = TraininingPipeline()
        train_pipeline.run_pipeline()
        
        return "training completed"
    
    except Exception as e:
        raise CustomException(e sys)

@app.route('/predict', methods=['POST', 'GET'])    
def upload():
    try:
        if request.method == 'POST':
            # it is a object of prediction pipeline
            prediction_pipeline = PredictionPipeline(request)
            
            #now we are running this run pipeline method
            prediction_file_detail = prediction_pipeline.run_pipeline()

            logging.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)


        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e sys)
    
#execution will start from here
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)    