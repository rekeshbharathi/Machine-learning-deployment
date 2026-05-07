from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Install pyngrok if not already installed
!pip install pyngrok
from pyngrok import ngrok

# Initialise Flask app
app = Flask(__name__)

# Load the model and scaler
try:
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('standard_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the feature names (must be in the same order as during training)
# X.columns from the training notebook was: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json(force=True)

        # Convert input data to DataFrame with correct feature order
        input_df = pd.DataFrame([data], columns=feature_names)

        # Handle '0' values in specific columns by replacing with NaN, then imputing with mean
        # This must match the preprocessing done during training
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)

        # Impute missing values for prediction based on the training data's means
        # This requires the original means. For a simple API, we'll assume the scaler handles this implicitly
        # or that the input handling ensures no NaNs are passed to the scaler that it didn't see before.
        # A more robust solution might involve saving the imputation means as well.

        # For simplicity in this API, we will directly apply the scaler and ensure the input structure matches.

        # Note: The scaler was fitted on data where 0s were replaced by NaNs and then imputed.
        # The current implementation of the API assumes the input will come in with 0s for missing values
        # which need to be converted to NaNs before scaling, similar to the training preprocessing.
        # However, the scaler itself was trained on mean-imputed data, not raw data with NaNs.
        # To perfectly replicate: we'd need the means for imputation. For now, we scale 'as is' after NaN replacement.

        # Important: For a production-ready API, you would typically save the imputation means
        # during training and use them here. Since we only saved the scaler, we'll rely on the
        # scaler's transformation capabilities. The `StandardScaler` will handle NaNs if it was trained on them.
        # However, since we filled NaNs with mean before scaling, the scaler expects non-NaN values.
        # Let's adjust for correct preprocessing pipeline re-application.

        # Re-apply the imputation step using the stored 'X' DataFrame's means
        # This is crucial: the API needs to use the same imputation strategy as training
        # If X was preserved, we could do: input_df.fillna(X.mean(), inplace=True)
        # Since X is not directly available, for this demo, we'll assume the input comes pre-cleaned or
        # that we re-calculate the means here (which is not ideal as it won't be from training data).
        # A better approach is to save the imputation pipeline or the means.

        # For now, let's assume `input_df` comes with '0's for missing values as in raw data
        # and that the original `scaler` was fit on the *imputed* `X_train`.
        # So, incoming 0s must be handled. Let's use the mean of the training data 'X' directly if possible.

        # Accessing `X` (original features before splitting) for means:
        # Since `X` is a kernel variable, we can try to use its means.
        global X # Access the global X DataFrame for mean imputation
        if 'X' in globals():
            for col in zero_cols:
                if col in input_df.columns:
                    input_df[col] = input_df[col].fillna(X[col].mean()) # Use original X's mean
        else:
             # Fallback if X is not available (e.g., fresh kernel restart before X is defined)
             # In a real API deployment, these means would be stored and loaded.
             print("Warning: Original 'X' DataFrame not found for imputation. Using default value for NaNs if any remain.")
             for col in zero_cols:
                 input_df[col] = input_df[col].fillna(input_df[col].mean()) # Impute with current input's mean (less robust)


        # Scale the input data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask app with ngrok
def run_flask_app():
    # Authenticate ngrok. Replace 'YOUR_AUTHTOKEN' with your actual ngrok authtoken.
    # You can get one from https://ngrok.com/signup
    # For Colab, you might need to install pyngrok first: !pip install pyngrok

    # Make sure to set your NGROK_AUTH_TOKEN as an environment variable or pass it directly
    ngrok.set_auth_token("YOUR_AUTHTOKEN") # Uncomment and replace if not set as env variable

    try:
        public_url = ngrok.connect(5000)
        print(f" * ngrok tunnel available at: {public_url}")
        print(f" * Flask app running on: http://127.0.0.1:5000")
        app.run(port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error starting ngrok or Flask app: {e}")
        print("Please ensure `pyngrok` is installed (`!pip install pyngrok`) and your ngrok authtoken is configured.")


# This block ensures the Flask app runs when the cell is executed
if __name__ == '__main__':
  run_flask_app()
  




