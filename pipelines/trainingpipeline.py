from zenml import pipeline
from Steps.DataValidation import Datavalidation # Module to validate data
from Steps.DataIngestion import data_ingestion  # Module to ingest data
from Steps.DataPreprocessing import split_data  # Module to split data
from Steps.TrainModel import trained_model  # Module to train the model
from Steps.evaluation import model_accuracy  # Module for model evaluation

@pipeline()
def trainingpipeline():
    folder_path = '/Data for ml'
    Datavalidation(folder_path)  # Validate data
    df = data_ingestion(folder_path)  # Ingest data
    x_train, x_test, y_train, y_test = split_data(df)  # Split data
    trained_model(x_train, x_test, y_train, y_test)  # Train the model
    score, mse, r2, rmse = model_accuracy(x_train, x_test, y_train, y_test)  # Evaluate model accuracy

if __name__ == '__main__':
    trainingpipeline()
