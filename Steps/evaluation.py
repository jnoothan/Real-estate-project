from sklearn.metrics import mean_squared_error, r2_score
from zenml import step
import pickle
import os

class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def model_score(x_test, y_test):
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        score = model.score(x_test, y_test)
        return score

    @staticmethod
    def model_MSE(y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        return mse

    @staticmethod
    def model_r2(y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        return r2

    @staticmethod
    def model_rmse(y_test, y_pred):
        rmse = mean_squared_error(y_test, y_pred)
        return rmse

@step
@step
def model_accuracy(x_train, x_test, y_train, y_test):
    eval = Evaluation(x_train, x_test, y_train, y_test)
    score = eval.model_score(x_test, y_test)
    y_pred = eval.model.predict(x_test)
    mse = eval.model_MSE(y_test, y_pred)
    r2 = eval.model_r2(y_test, y_pred)
    rmse = eval.model_rmse(y_test, y_pred)
    return score, mse, r2, rmse

