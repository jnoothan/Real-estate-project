
from zenml import pipeline
from config.config import data_for_ml, valid_data
from Steps.DataValidation import DataValidation  # Module to validate data
from Steps.DataIngestion import data_ingestion  # Module to ingest data
from Steps.DataPreprocessing import split_data  # Module to split data
from Steps.TrainModel import trained_model  # Module to train the model
from Steps.evaluation import model_accuracy  # Module for model evaluation # Module for model Prediction flask

@pipeline(enable_cache=False)
def trainingpipeline():
    folder_path = data_for_ml
    DataValidation(folder_path)
    relative_path = valid_data
    df=data_ingestion(relative_path)
    x_train, x_test, y_train, y_test = split_data(df)
    trained_model(x_train, x_test, y_train, y_test)
    model_accuracy(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    trainingpipeline()
