# step 4
import os
import logging
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from steps.DataPreprocessing import DataPreprocess
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle

class model_training:
    def __init__(self, x_train, x_test, y_train, y_test) -> tuple:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def Model_selection_GridSearchCV(x_train, x_test, y_train, y_test) -> tuple:
        param_grid_decision_tree = {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }

        grid_search_decision_tree = GridSearchCV(
            DecisionTreeRegressor(),
            param_grid_decision_tree,
            scoring="neg_mean_squared_error",
            cv=5,
        )

        grid_search_decision_tree.fit(x_train, y_train)

        Decision_Tree = {
            "Name": "Decision Tree",
            "Best Parameters": grid_search_decision_tree.best_params_,
            "Best MSE": -grid_search_decision_tree.best_score_,
        }

        param_grid_xgboost = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "n_estimators": [50, 100, 200],
        }

        grid_search_xgboost = GridSearchCV(
            XGBRegressor(), param_grid_xgboost, scoring="neg_mean_squared_error", cv=5
        )

        grid_search_xgboost.fit(x_train, y_train)

        XGBoost = {
            "Name": "XGBoost",
            "Best Parameters": grid_search_xgboost.best_params_,
            "Best MSE": -grid_search_xgboost.best_score_,
        }

        param_grid_lightgbm = {
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "n_estimators": [50, 100, 200],
        }

        grid_search_lightgbm = GridSearchCV(
            LGBMRegressor(), param_grid_lightgbm, scoring="neg_mean_squared_error", cv=5
        )

        grid_search_lightgbm.fit(x_train, y_train)

        LightGBM = {
            "Name": "LightGBM",
            "Best Parameters": grid_search_lightgbm.best_params_,
            "Best MSE": -grid_search_lightgbm.best_score_,
        }

        param_grid_catboost = {
            "learning_rate": [0.01, 0.1, 0.2],
            "depth": [3, 5, 7],
            "n_estimators": [50, 100, 200],
        }

        grid_search_catboost = GridSearchCV(
            CatBoostRegressor(),
            param_grid_catboost,
            scoring="neg_mean_squared_error",
            cv=5,
        )

        grid_search_catboost.fit(x_train, y_train)

        CatBoost = {
            "Name": "CatBoost",
            "Best Parameters": grid_search_catboost.best_params_,
            "Best MSE": -grid_search_catboost.best_score_,
        }

        list_of_models = [Decision_Tree, XGBoost, LightGBM, CatBoost]

        # Find the model with the minimum MSE
        min_model = min(list_of_models, key=lambda x: x["Best MSE"])

        return min_model

    @staticmethod
    def model_fit( min_model,x_train, y_train):
        if min_model["Name"] == "Decision Tree":
            best_params = min_model["Best Parameters"]
            model = DecisionTreeRegressor(**best_params)
            model.fit(x_train, y_train)

        elif min_model["Name"] == "XGBoost":
            best_params = min_model["Best Parameters"]
            model = XGBRegressor(**best_params)
            model.fit(x_train, y_train)

        elif min_model["Name"] == "CatBoost":
            best_params = min_model["Best Parameters"]
            model = CatBoostRegressor(**best_params)
            model.fit(x_train, y_train)

        elif min_model["Name"] == "LightGBM":
            best_params = min_model["Best Parameters"]
            model = LGBMRegressor(**best_params)
            model.fit(x_train, y_train)

        return model

    @staticmethod
    def model_pickle(model):
        file_path = "model/model.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        return file_path

@step
def trained_model(x_train,x_test,y_train,y_test):
    min_model = model_training.Model_selection_GridSearchCV(x_train, x_test, y_train, y_test)
    trained_model = model_training.model_fit(min_model, x_train, y_train)
    return model_training.model_pickle(trained_model)


