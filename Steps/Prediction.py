from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model and other necessary data
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load label encoders for city and state
with open('city_encoder.pkl', 'rb') as file:
    city_encoder = pickle.load(file)

with open('state_encoder.pkl', 'rb') as file:
    state_encoder = pickle.load(file)

# Define Flask app
app = Flask(__name__)

# Dummy data for demonstration
state_names = ['New York', 'Massachusetts', 'Connecticut', 'New Hampshire', 'Pennsylvania',
               'New Jersey', 'Maine', 'Puerto Rico', 'Vermont', 'Rhode Island', 'Delaware',
               'Virgin Islands', 'West Virginia', 'Wyoming']

cities_by_state = {
    'New York': ['New York City', 'Albany', 'Buffalo'],  # Example cities
    'Massachusetts': ['Boston', 'Cambridge', 'Springfield'],  # Example cities
    # Add other states and their cities here
}

# Define routes
@app.route('/')
def index():
    return render_template('index.html', state_names=state_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        bed = int(request.form['bed'])
        bath = int(request.form['bath'])
        acre_lot = float(request.form['acre_lot'])
        city = request.form['city']
        state = request.form['state']
        house_size = float(request.form['house_size'])

        # Encode city and state
        encoded_city = city_encoder.transform([city])[0]
        encoded_state = state_encoder.transform([state])[0]

        # Perform prediction using the model
        prediction = model.predict([[bed, bath, acre_lot, encoded_city, encoded_state, house_size]])[0]

        return render_template('result.html', prediction=prediction)

@app.route('/get_cities', methods=['POST'])
def get_cities():
    state = request.form['state']
    cities = cities_by_state.get(state, [])
    return {'cities': cities}

if __name__ == '__main__':
    app.run(debug=True)
