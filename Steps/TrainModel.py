import os
import logging
import sys
sys.path.append('F:/Data Science/ML Projects 22-09-2023/Real estate project')
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from zenml import step
from Steps.DataPreprocessing import split_data

# Set up logging
logging.basicConfig(filename='model_training.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class model_training:
    @staticmethod
    def Model_selection_GridSearchCV(x_train, x_test, y_train, y_test) -> dict:
        try:
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

            """param_grid_catboost = {
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
            }"""

            list_of_models = [Decision_Tree, XGBoost]

            # Find the model with the minimum MSE
            min_model = min(list_of_models, key=lambda x: x["Best MSE"])

            return min_model
        except Exception as e:
            logging.error(f"Error in model selection: {e}")
            return {}

    @staticmethod
    def model_fit(min_model, x_train, y_train):
        try:
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

            return model
        except Exception as e:
            logging.error(f"Error in model fitting: {e}")
            return None

    @staticmethod
    def model_pickle(model):
        try:
            file_path = "model/model.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(model, f)
            return file_path
        except Exception as e:
            logging.error(f"Error in model pickling: {e}")
            return None

@step
def trained_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series)->str:
    try:
        min_model = model_training.Model_selection_GridSearchCV(x_train, x_test, y_train, y_test)
        trained_model = model_training.model_fit(min_model, x_train, y_train)
        return model_training.model_pickle(trained_model)
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        return None

if __name__ == '__main__':
    try:
        df = pd.read_csv('F:/Data Science/ML Projects 22-09-2023/Real estate project/Ingested Data/Ingested_data.csv')
        x_train, x_test, y_train, y_test = split_data(df)
        trained_model(x_train, x_test, y_train, y_test)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
