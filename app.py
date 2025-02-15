from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

#  Load the trained model & scaler
model = joblib.load("weather_model_rf_optimized.pkl")
scaler = joblib.load("scaler.pkl")

#  Load the test dataset for reference
test_df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")
test_df["date"] = pd.to_datetime(test_df["date"])

#  Define the home route
@app.route("/")
def home():
    return render_template("index.html")

#  Prediction API Route
@app.route("/predict", methods=["POST"])
def predict():
    date_input = request.form.get("date")
    
    # Convert user input to datetime
    date_input = pd.to_datetime(date_input)

    # Find the closest matching date in the dataset
    closest_data = test_df.loc[test_df["date"] == date_input, ["humidity", "wind_speed", "meanpressure"]]

    if not closest_data.empty:
        X_input_scaled = scaler.transform(closest_data)
        predicted_temp = model.predict(X_input_scaled)[0]
        return jsonify({"date": str(date_input.date()), "predicted_temperature": round(predicted_temp, 2)})
    
    return jsonify({"error": "Date not found in dataset"}), 400

if __name__ == "__main__":
    app.run(debug=True)
