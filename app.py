from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- MODEL LOADING ---
# Load the model and scaler from the root directory
def load_resources():
    try:
        model_path = 'logistic_regression_model.pkl'
        scaler_path = 'standard_scaler.pkl'
        
        with open(model_path, 'rb') as m_file:
            model = pickle.load(m_file)
        with open(scaler_path, 'rb') as s_file:
            scaler = pickle.load(s_file)
        
        print("Model and Scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None

model, scaler = load_resources()

# The exact feature order used during model.fit()
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/', methods=['GET'])
def health_check():
    return "Diabetes Prediction API is active and running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Server configuration error: Model files missing'}), 500

    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert input data to DataFrame with correct feature order
        # This handles input even if the user sends fields in the wrong order
        input_df = pd.DataFrame([data])
        
        # Ensure all required features are present, fill missing with 0
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training exactly
        input_df = input_df[feature_names]

        # Preprocessing: The training logic replaced 0s with NaNs then imputed.
        # To keep this API simple and robust, we assume the scaler handles the range,
        # but you should ideally apply the same imputation means here if 0s are present.
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        # Note: If you have specific mean values from your training data, 
        # you would fill them here.
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'result': 'Diabetes Positive' if int(prediction[0]) == 1 else 'Diabetes Negative',
            'confidence': {
                'no_diabetes': round(float(prediction_proba[0][0]), 4),
                'diabetes': round(float(prediction_proba[0][1]), 4)
            }
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Render assigns a port via environment variables
    port = int(os.environ.get("PORT", 5000))
    # host='0.0.0.0' is required for cloud deployment
    app.run(host='0.0.0.0', port=port)
