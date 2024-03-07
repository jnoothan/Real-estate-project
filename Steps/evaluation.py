import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from zenml import step
import pickle
import sys
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')
from Steps.DataPreprocessing import split_data
import logging

# Set up logging
logging.basicConfig(filename='model_evaluation.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory of 'Steps' package to the Python path
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')

class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.load_model()

    def load_model(self):
        try:
            # Load the model
            with open('model/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")

    def model_score(self, x_test, y_test):
        try:
            # Return the score
            score = self.model.score(x_test, y_test)
            return score
        except Exception as e:
            logging.error(f"Error calculating model score: {e}")
            print(f"Error calculating model score: {e}")
            return None

    def model_MSE(self, y_test, y_pred):
        try:
            # Return the mean squared error
            mse = mean_squared_error(y_test, y_pred)
            return mse
        except Exception as e:
            logging.error(f"Error calculating mean squared error: {e}")
            print(f"Error calculating mean squared error: {e}")
            return None

    def model_r2(self, y_test, y_pred):
        try:
            # Return the R^2 score
            r2 = r2_score(y_test, y_pred)
            return r2
        except Exception as e:
            logging.error(f"Error calculating R^2 score: {e}")
            print(f"Error calculating R^2 score: {e}")
            return None

    def model_rmse(self, y_test, y_pred):
        try:
            # Return the root mean squared error
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            return rmse
        except Exception as e:
            logging.error(f"Error calculating root mean squared error: {e}")
            print(f"Error calculating root mean squared error: {e}")
            return None


@step
def model_accuracy(x_train, x_test, y_train, y_test):
    try:
        # Create an Evaluation instance
        eval = Evaluation(x_train, x_test, y_train, y_test)

        # Get predictions
        y_pred = eval.model.predict(x_test)

        # Calculate metrics using instance method
        score = eval.model_score(x_test, y_test)  # Pass both x_test and y_test directly
        mse = eval.model_MSE(y_test, y_pred)
        r2 = eval.model_r2(y_test, y_pred)
        rmse = eval.model_rmse(y_test, y_pred)

        # Print and return metrics
        print(score, mse, r2, rmse)
        return score, mse, r2, rmse
    except Exception as e:
        logging.error(f"Error in model accuracy calculation: {e}")
        print(f"Error in model accuracy calculation: {e}")
        return None


if __name__ == "__main__":
    try:
        df = pd.read_csv(r'F:\Data Science\ML Projects 22-09-2023\Real estate project\Ingested Data\Ingested_data.csv')
        x_train, x_test, y_train, y_test = split_data(df)
        model_accuracy(x_train, x_test, y_train, y_test)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
