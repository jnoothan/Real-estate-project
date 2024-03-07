import pandas as pd
import sys
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from zenml import step
import logging
import os
import pickle

# Set up logging
logging.basicConfig(filename='data_preprocessing.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocess:
    def __init__(self, df)->pd.DataFrame:
        self.df = df

    def data_transformation(self) -> pd.DataFrame:
        try:
            # Convert float64 columns to int32
            float_cols = ['bed', 'bath', 'house_size', 'price']
            self.df[float_cols] = self.df[float_cols].astype('int32')
            return self.df
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            return None

    def feature_selection(self) -> pd.DataFrame:
        try:
            # Drop unnecessary columns
            drop_columns = ["status"]
            self.df.drop(columns=drop_columns, inplace=True)
            return self.df
        except Exception as e:
            logging.error(f"Error in feature selection: {e}")
            return None

    def label_encoding(self) :
        try:
            # Label encode categorical columns
            encode_columns = ["city", "state"]
            for col in encode_columns:
                encoder = LabelEncoder()
                self.df[col] = encoder.fit_transform(self.df[col])
                file_path = os.path.join('model', col + '.pkl')
                with open(file_path, "wb") as f:
                    pickle.dump(encoder, f)
            return self.df
        except Exception as e:
            logging.error(f"Error in label encoding: {e}")
            return None

    def data_splitting(self) -> (pd.DataFrame, pd.Series):
        try:
            x = self.df.drop("price", axis=1)
            y = self.df["price"]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=1
            )
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            return None, None, None, None

@step
def split_data(df) -> (pd.DataFrame,  pd.DataFrame,pd.Series, pd.Series):
    preprocess = DataPreprocess(df)
    preprocess.data_transformation()
    preprocess.feature_selection()
    preprocess.label_encoding()
    type(preprocess.data_splitting())
    return preprocess.data_splitting()

if __name__ == '__main__':
    try:
        df = pd.read_csv(r'F:\Data Science\ML Projects 22-09-2023\Real estate project\Ingested Data\Ingested_data.csv')
        x_train, x_test, y_train, y_test = split_data(df)
        print(x_train, x_test, y_train, y_test)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
