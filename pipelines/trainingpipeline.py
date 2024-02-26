from zenml import pipeline
from Steps.DataValidation import DataValidation # returns csv files in ingested data folder
from Steps.DataIngestion import data_ingestion # returns df ingests data as df into ingested data folder or invalid data folder
from Steps.DataPreprocessing import DataPreprocess # returns x_train,X_test,y_train,y_test

@pipeline()
def trainingpipeline():
    folder_path='/Data for ml'
    DataValidation(folder_path)
    df=data_ingestion(Ingested Data)
    x_train,x_test,y_train,y_test=DataPreprocess(df)

