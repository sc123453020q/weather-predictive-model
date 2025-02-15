import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load the test dataset
test_df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")

# Convert 'date' column to datetime format
test_df['date'] = pd.to_datetime(test_df['date'])

#  Load the trained model and scaler
model = joblib.load("weather_model_rf_optimized.pkl")
scaler = joblib.load("scaler.pkl")

#  Select Features for Testing
features = ["humidity", "wind_speed", "meanpressure"]
X_test = test_df[features]
y_test = test_df["meantemp"]

#  Apply Feature Scaling to Test Data
X_test_scaled = scaler.transform(X_test)

#  Make Predictions
y_pred = model.predict(X_test)

#  Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#  User Input for a Specific Date Prediction
date_input = input("Enter a date (YYYY-MM-DD) to predict temperature: ")
date_input = pd.to_datetime(date_input)

# Find the closest date in the dataset
closest_data = test_df.loc[test_df['date'] == date_input, features]

if not closest_data.empty:
    X_input_scaled = scaler.transform(closest_data)
    predicted_temp = model.predict(X_input_scaled)[0]
    print(f"🌡️ Predicted Temperature for {date_input.date()}: {predicted_temp:.2f}°C")
else:
    print("❌ Date not found in the dataset. Try another date!")

#  User Input for a Date Range to Visualize
start_date = input("Enter Start Date (YYYY-MM-DD) for visualization: ")
end_date = input("Enter End Date (YYYY-MM-DD) for visualization: ")

# Convert input strings to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

#  Filter Data Based on Input Dates
filtered_df = test_df[(test_df['date'] >= start_date) & (test_df['date'] <= end_date)]

# Plot Actual vs Predicted Temperature for Selected Date Range
plt.figure(figsize=(12, 5))
plt.plot(filtered_df['date'], filtered_df['meantemp'], label="Actual Temperature", color='blue')
plt.plot(filtered_df['date'], y_pred[test_df['date'].between(start_date, end_date)], label="Predicted Temperature", color='red')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"Actual vs Predicted Temperature ({start_date.date()} to {end_date.date()})")
plt.legend()
plt.show()
