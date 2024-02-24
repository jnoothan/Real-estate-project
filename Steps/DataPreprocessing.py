# step 3
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# preprocess data - handle missing values
class DataPreprocess:
    def __init__(self, df) -> pd.DataFrame:
        self.df = df

    def data_transformation(
        self, df: pd.DataFrame, columns=["bed", "bath", "house_size", "price"]
    ):
        self.columns = columns
        for col in df[columns]:
            if (df[col].dtype) == "float64":
                df[col] = df[col].astype("int32")
        return df

    def feature_selection(self, df: pd.DataFrame, drop_columns=["status"]):
        df = df.drop(columns=drop_columns)
        return df

    def one_hot_encoding(df: pd.DataFrame, encode_columns=["city", "state"]):
        encoder = OneHotEncoder()
        for col in df[encode_columns]:
            df[col] = encoder.fit_transform(df[col])
        return df

    def Data_splitting(self, df: pd.DataFrame) -> tuple(pd.DataFrame, pd.Series):
        x = df.drop("price", axis=1)
        y = df["price"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    DataPreprocess()
