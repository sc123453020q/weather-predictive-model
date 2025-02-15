import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

#  Load training dataset
train_df = pd.read_csv("SET_1_DailyDelhiClimateTrain.csv")

# Convert 'date' column to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])

#  Select Features (Independent Variables) and Target Variable
features = ["humidity", "wind_speed", "meanpressure"]  # X (Predictors)
target = "meantemp"  # y (Target Variable)

X_train = train_df[features]
y_train = train_df[target]

#  Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

#  Define Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [5, 10, 15],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Min samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Min samples per leaf node
}

#  Initialize RandomForest Model
rf_model = RandomForestRegressor(random_state=42)

#  Perform Grid Search for Best Hyperparameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

#  Get Best Parameters
best_params = grid_search.best_params_
print("✅ Best Parameters Found:", best_params)

#  Train Model with Best Parameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train_scaled, y_train)

#  Save the Best Model and Scaler
joblib.dump(best_rf_model, "weather_model_rf_optimized.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete. Optimized model is saved as 'weather_model_rf_optimized.pkl'.")
