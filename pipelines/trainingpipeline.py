import sys
# Add the parent directory of 'Steps' package to the Python path
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')
import pandas as pd
import os
from zenml import pipeline
from Steps.DataValidation import DataValidation  # Module to validate data
from Steps.DataIngestion2 import data_ingestion  # Module to ingest data
from Steps.DataPreprocessing import split_data  # Module to split data
from Steps.TrainModel import trained_model  # Module to train the model
from Steps.evaluation import model_accuracy  # Module for model evaluation # Module for model Prediction flask

@pipeline(enable_cache=True)
def trainingpipeline():
    folder_path = './Data for ml'
    DataValidation(folder_path)
    relative_path = "./Valid_Data"
    df=data_ingestion(relative_path)
    x_train, x_test, y_train, y_test = split_data(df)
    trained_model(x_train, x_test, y_train, y_test)
    model_accuracy(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    trainingpipeline()
