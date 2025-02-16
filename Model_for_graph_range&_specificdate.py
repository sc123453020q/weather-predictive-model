import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------
#  Step 1: Load and Preprocess Training Data
# -----------------------------------------------
train_df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")

# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])

# Feature Engineering: Adding Previous Day's Temperature
train_df['temperature_lag'] = train_df['meantemp'].shift(1)

# Drop rows with NaN (first row will have NaN in 'temperature_lag')
train_df.dropna(inplace=True)

#  Removing Outliers Using Interquartile Range (IQR)
Q1 = train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].quantile(0.25)
Q3 = train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].quantile(0.75)
IQR = Q3 - Q1

train_df = train_df[~((train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']] < (Q1 - 1.5 * IQR)) | 
                      (train_df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Select Features & Target
features = ["humidity", "wind_speed", "meanpressure", "temperature_lag"]
X_train = train_df[features]
y_train = train_df["meantemp"]

#  Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#  Step 2: Train & Tune Model Using GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

#  Save the best model & scaler
joblib.dump(grid_search.best_estimator_, "weather_model_rf_optimized.pkl")
joblib.dump(scaler, "scaler.pkl")

print(" Model Retrained with Best Parameters:", grid_search.best_params_)

#  Step 3: Load Test Data & Evaluate Model

test_df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")
test_df['date'] = pd.to_datetime(test_df['date'])

# Feature Engineering: Add Previous Day's Temperature
test_df['temperature_lag'] = test_df['meantemp'].shift(1)
test_df.dropna(inplace=True)

# Load trained model & scaler
model = joblib.load("weather_model_rf_optimized.pkl")
scaler = joblib.load("scaler.pkl")

# Select Features for Testing
X_test = test_df[features]
y_test = test_df["meantemp"]

# Apply Feature Scaling
X_test_scaled = scaler.transform(X_test)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance (Optimized):")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")


#  Step 4: Predict Temperature for a User Input Date

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

#  Step 5: Visualize Predictions Over a Date Range

start_date = input("Enter Start Date (YYYY-MM-DD) for visualization: ")
end_date = input("Enter End Date (YYYY-MM-DD) for visualization: ")

# Convert input strings to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter Data Based on Input Dates
filtered_df = test_df[(test_df['date'] >= start_date) & (test_df['date'] <= end_date)]

# Generate Predicted Values for the Selected Date Range
filtered_predictions = model.predict(scaler.transform(filtered_df[features]))

# Plot Actual vs Predicted Temperature
plt.figure(figsize=(12, 5))
plt.plot(filtered_df['date'], filtered_df['meantemp'], label="Actual Temperature", color='blue', marker='o')
plt.plot(filtered_df['date'], filtered_predictions, label="Predicted Temperature", color='red', linestyle='dashed', marker='x')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"📈 Actual vs Predicted Temperature ({start_date.date()} to {end_date.date()})")
plt.legend()
plt.grid(True)
plt.show()
