import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from zenml import step

class DataPreprocess:
    def __init__(self, df):
        self.df = df

    def data_transformation(self):
        # Convert float64 columns to int32
        float_cols = ['bed', 'bath','house_size', 'price']
        self.df[float_cols] = self.df[float_cols].astype('int32')

    def feature_selection(self):
        # Drop unnecessary columns
        drop_columns = ["status"]
        self.df = self.df.drop(columns=drop_columns)

    def label_encoding(self):
        # Label encode categorical columns
        encode_columns = ["city", "state"]
        encoder = LabelEncoder()
        for col in encode_columns:
            self.df[col] = encoder.fit_transform(self.df[col])

    def data_splitting(self):
        x = self.df.drop("price", axis=1)
        y = self.df["price"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        return x_train, x_test, y_train, y_test

@step
def split_data(df) -> tuple:
    preprocess = DataPreprocess(df)
    preprocess.data_transformation()
    preprocess.feature_selection()
    preprocess.label_encoding()
    return preprocess.data_splitting()

if __name__ == '__main__':
    df = pd.read_csv(r'F:\Data Science\ML Projects 22-09-2023\Real estate project\Ingested Data\Ingested_data.csv')
    split_data(df)
