from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
with open("./iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the iris dataset to get target names
iris = load_iris()

# Create Flask app
app = Flask(__name__)

# Define a prediction route
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Ensure correct feature inputs
    features = data.get('features')
    if not features or len(features) != 4:
        return jsonify({'error': 'Invalid input data. Expected 4 features.'}), 400
    
    # Convert input features to a numpy array and reshape for the model
    input_features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'class': iris.target_names[prediction[0]]
    })

# Serverless function export
def handler(event, context):
    return app(environ=event, start_response=context)
