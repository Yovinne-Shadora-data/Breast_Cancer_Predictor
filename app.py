from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('logistic_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the top 5 features from ANOVA
feature_names = [
    'perimeter_mean',
    'concave points_mean',
    'radius_worst',
    'perimeter_worst',
    'concave points_worst'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form and convert to float
        input_data = [float(request.form[feature]) for feature in feature_names]

        # Scale input data
        scaled_input = scaler.transform([input_data])

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        result = "Malignant" if prediction == 1 else "Benign"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
