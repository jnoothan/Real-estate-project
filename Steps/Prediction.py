from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    bed = int(request.form['bed'])
    bath = int(request.form['bath'])
    acre_lot = float(request.form['acre_lot'])
    city = request.form['city']
    state = request.form['state']
    house_size = int(request.form['house_size'])

    # You can perform any additional processing or data cleaning here if needed

    # Make predictions using the loaded model
    prediction = model.predict([[bed, bath, acre_lot, house_size]])

    # Return the prediction
    return render_template('index.html', prediction=f'Predicted Price: ${prediction[0]:,.2f}')


if __name__ == '__main__':
    app.run(debug=True)
