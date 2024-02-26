import os
import logging
import pandas as pd
from Steps.DataValidation import
from zenml import step
from zenml import pipeline
from Steps.DataValidation import DataValidation

folderpath=''

# step by step 
# 1 ingest data
# 2 preprocess data
# 3 transform data
# 4 split data  
# 6 model selection & training
# 7 Flask API Prediction
# 8 Deployment

if __name__=='__main__':
    DataValidation(folderpath)
    