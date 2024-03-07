import sys
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')
import logging
import os
import pandas as pd
from Steps.DataValidation import DataValidation
from Steps.DataIngestion import data_ingestion
from Steps.DataPreprocessing import split_data
from Steps.TrainModel import trained_model
from Steps.evaluation import model_accuracy
from Steps.Prediction import app
from zenml import step
from zenml import pipeline




folderpath=''

# step by step 
# 1 ingest data
# 2 preprocess data
# 3 transform data
# 4 split data  
# 6 model selection & training
# 7 Flask API Prediction
# 8 Deployment
@pipeline()
def run_training_pipeline():
    DataValidation(folderpath)
    relative_path = "./Valid_Data"
    df=data_ingestion(relative_path)
    x_train, x_test, y_train, y_test = split_data(df)
    trained_model(x_train, x_test, y_train, y_test)
    model_accuracy(x_train, x_test, y_train, y_test)
    app.run()


if __name__=='__main__':
    training = trainingpipeline()
    training.run()