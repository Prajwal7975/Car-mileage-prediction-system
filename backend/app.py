import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd


app = Flask(__name__)
CORS(app)

# Load pipeline model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "final_mileage_model.pkl")

model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # Add engineered features
        input_data["Car_Age"] = 2025 - input_data["Year"]
        input_data["Engine_to_Power"] = input_data["Engine"] / (input_data["Power"] + 1)
        input_data["KM_per_Year"] = input_data["Kilometers_Driven"] / (input_data["Car_Age"] + 1)
        
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)


        return jsonify({"mileage": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)