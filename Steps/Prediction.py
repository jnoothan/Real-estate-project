# step 5
import os
import logging
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from steps.DataPreprocessing import DataPreprocess
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)


@app.route("/predict_price", methods=["POST"])
def predict_price():
    data = request.get_json()
    # Load the trained model
    model = joblib.load("model/model.pkl")
    # Extract features from the request
    bed = data["bed"]
    bath = data["bath"]
    acre_lot = data["acre_lot"]
    city = data["city"]
    state = data["state"]
    house_size = data["house_size"]

    # Process the features as needed
    # For example, you may need to encode categorical features like city and state

    # Make the prediction
    prediction = model.predict([[bed, bath, acre_lot, house_size]])

    # Return the prediction as JSON
    return jsonify({"predicted_price": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)
