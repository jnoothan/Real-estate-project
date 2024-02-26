import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from zenml import step

# preprocess data - handle missing values
class DataPreprocess:
    def __init__(self, df):
        self.df = df

    def data_transformation(self):
        columns = ["bed", "bath", "house_size", "price"]
        for col in columns:
            if self.df[col].dtype == "float64":
                self.df[col] = self.df[col].astype("int32")

    def feature_selection(self):
        drop_columns = ["status"]
        self.df = self.df.drop(columns=drop_columns)

    def one_hot_encoding(self):
        encode_columns = ["city", "state"]
        encoder = OneHotEncoder()
        for col in encode_columns:
            encoded_data = encoder.fit_transform(self.df[col].values.reshape(-1, 1)).toarray()
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
            self.df = pd.concat([self.df, encoded_df], axis=1)
            self.df.drop(columns=[col], inplace=True)

    def data_splitting(self):
        x = self.df.drop("price", axis=1)
        y = self.df["price"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        return x_train, x_test, y_train, y_test

@step
def split_data(df)-> tuple:
    preprocess = DataPreprocess(df)
    preprocess.data_transformation()
    preprocess.feature_selection()
    preprocess.one_hot_encoding()
    return preprocess.data_splitting()
