import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Load dataset
df = pd.read_csv("SET_1_DailyDelhiClimateTest.csv")

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Check if the data is loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fix Pandas FutureWarning (Use ffill() instead of method='ffill')
df.ffill(inplace=True)

# Remove duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Check column names
print("\nColumn Names:", df.columns)

#  EDA and Visualization
sns.set_style("darkgrid")

#  Plot temperature over time (Fixed column name)
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['meantemp'], color='r', label="Mean Temperature (°C)")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Trend in Delhi (2013-2017)")
plt.legend()
plt.show()

# Plot humidity over time
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['humidity'], color='b', label="Humidity (%)")
plt.xlabel("Year")
plt.ylabel("Humidity (%)")
plt.title("Humidity Trend in Delhi (2013-2017)")
plt.legend()
plt.show()

#  Boxplot to check seasonal variations in temperature
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['date'].dt.month, y=df['meantemp'], palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.title("Seasonal Temperature Variations")
plt.show()
