from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --- CONFIGURATION ---
# In a real app, these means should be calculated from your training set 
# and hardcoded here or saved in a JSON/PKL file.
TRAINING_MEANS = {
    'Glucose': 121.68,
    'BloodPressure': 72.40,
    'SkinThickness': 29.15,
    'Insulin': 155.54,
    'BMI': 32.45
}

# Load the model and scaler
def load_assets():
    try:
        with open('logistic_regression_model.pkl', 'rb') as m_file:
            model = pickle.load(m_file)
        with open('standard_scaler.pkl', 'rb') as s_file:
            scaler = pickle.load(s_file)
        return model, scaler
    except FileNotFoundError:
        print("Error: .pkl files not found in the current directory.")
        return None, None

model, scaler = load_assets()

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model assets not loaded on server'}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure all columns exist, fill missing ones with 0 or mean
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training
        input_df = input_df[feature_names]

        # 2. Imputation logic: Replace 0 with training means for specific columns
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if input_df.loc[0, col] == 0:
                input_df.loc[0, col] = TRAINING_MEANS[col]

        # 3. Scale and Predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': round(float(prediction_proba[0][0]), 4),
            'probability_diabetes': round(float(prediction_proba[0][1]), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use standard Flask run for local/server deployment
    # If using Colab, keep your ngrok logic here instead
    app.run(host='0.0.0.0', port=5000, debug=True)
