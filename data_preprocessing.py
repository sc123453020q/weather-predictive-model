import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")  # Change filename if needed

# Display first few rows
print("First 5 rows:")
print(df.head())

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values (if any)
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Summary of dataset
print("\nDataset Summary:")
print(df.info())

# Save cleaned dataset
df.to_csv("cleaned_delhi_climate.csv", index=False)
