import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

#  Load the test dataset
test_df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")

# Convert 'date' column to datetime format
test_df['date'] = pd.to_datetime(test_df['date'])

#  Load the Optimized Model and Scaler
model = joblib.load("weather_model_rf_optimized.pkl")  # ✅ Load optimized model
scaler = joblib.load("scaler.pkl")

# Select Features for Testing
features = ["humidity", "wind_speed", "meanpressure"]
X_test = test_df[features]
y_test = test_df["meantemp"]

#  Apply Feature Scaling to Test Data
X_test_scaled = scaler.transform(X_test)

#  Make Predictions
y_pred = model.predict(X_test_scaled)  # ✅ Predict using optimized model

#  Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

#  Plot Actual vs Predicted Temperature
plt.figure(figsize=(12, 5))
plt.plot(test_df['date'], y_test, label="Actual Temperature", color='blue')
plt.plot(test_df['date'], y_pred, label="Predicted Temperature", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs Predicted Temperature in Delhi (Optimized Model)")
plt.legend()
plt.show()
