# Description: Short example for Time Series Visualization for Business Analysis with Python.



from data_io import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Load Time Series Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = read_csv(url, parse_dates=['Month'], index_col='Month')

# Plot the Time Series
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Passengers'], color='blue', label='Passengers')
plt.title("Monthly Airline Passenger Data")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.tight_layout()
plt.savefig('airline_passengers.png')
plt.show()

logger.info(df.head())
logger.info(f"Data range: {df.index.min()} to {df.index.max()}")

# Resample data by year
df_yearly = df.resample('Y').mean()

# Plot the Time Series
plt.figure(figsize=(10, 6))
plt.plot(df_yearly.index, df_yearly['Passengers'], color='blue', label='Passengers')
plt.title("Yearly Airline Passenger Data")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.tight_layout()
plt.savefig('airline_passengers_yearly.png')
plt.show()

logger.info(df_yearly.head())
logger.info(f"Data range: {df_yearly.index.min()} to {df_yearly.index.max()}")

# Resample data to daily frequency
df_daily = df.resample('D').interpolate(method='cubic')

# Plot the Time Series
plt.figure(figsize=(15, 6))
plt.plot(df_daily.index, df_daily['Passengers'], color='blue', label='Passengers (Interpolated)')
plt.title("Daily Airline Passenger Data (Interpolated)")
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.legend()
plt.tight_layout()
plt.savefig('airline_passengers_daily.png')
plt.show()

logger.info(df_daily.head())
logger.info(f"Data range: {df_daily.index.min()} to {df_daily.index.max()}")
logger.info(f"Total data points: {len(df_daily)}")

# Calculate Rolling Mean (12-month window)
rolling_mean = df.rolling(window=12).mean()
# Plot Original and Rolling Mean
plt.figure(figsize=(10, 6))
plt.plot(df.index, df.values, label='Original Data', alpha=0.5)
plt.plot(rolling_mean.index, rolling_mean.values, label='12-Month Rolling Mean', color='red', linewidth=2)
plt.title("12-Month Rolling Mean of Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.savefig('airline_passengers_rolling_mean.png')
plt.show()


# Perform seasonal decomposition
decomposition = seasonal_decompose(df['Passengers'], model='additive', period=12)

# Plot the Decomposed Components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))

ax1.plot(df.index, df['Passengers'], label='Original')
ax1.set_title('Original Time Series')
ax1.legend()

ax2.plot(decomposition.trend, label='Trend', color='red')
ax2.set_title('Trend')
ax2.legend()

ax3.plot(decomposition.seasonal, label='Seasonality', color='green')
ax3.set_title('Seasonality')
ax3.legend()

ax4.plot(decomposition.resid, label='Residuals', color='orange')
ax4.set_title('Residuals')
ax4.legend()

plt.tight_layout()
plt.savefig('airline_passengers_decomposition.png')
plt.show()


# Reshape Data for Heatmap (Year-Month)
df = df.reset_index()
df['Year'] = df['Month'].dt.year
df['Month'] = df['Month'].dt.month
pivot_table = df.pivot_table(index='Year', columns='Month', values='Passengers')

# Plot Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt=".0f")
plt.title("Heatmap of Monthly Airline Passengers")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
plt.savefig('airline_passengers_heatmap.png')
plt.show()

# Calculate rolling mean and standard deviation
window = 12  # 12-month rolling window
rolling_mean = df['Passengers'].rolling(window=window).mean()
rolling_std = df['Passengers'].rolling(window=window).std()

# Calculate dynamic threshold
threshold = rolling_mean + 2 * rolling_std

# Identify anomalies
anomalies = df[df['Passengers'] > threshold]

# Plot
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Passengers'], label="Original Data")
plt.plot(threshold.index, threshold, color='green', linestyle='--', label='Dynamic Threshold')
plt.scatter(anomalies.index, anomalies['Passengers'], color='red', s=50, label="Anomalies")

plt.title("Anomaly Detection in Airline Passengers (Dynamic Threshold)")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()

plt.tight_layout()
plt.savefig('airline_passengers_dynamic_anomalies.png')
plt.show()

logger.info("\nDetected Anomalies:")
logger.info(anomalies)

# Plotting the beehive data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['temperature'], label="Temperature (°C)", color='orange')
plt.plot(df.index, df['weight'], label="Hive Weight (kg)", color='brown')
plt.plot(df.index, df['bee_traffic'], label="Bee Traffic", color='green')
plt.title("Beehive Health Monitoring Over Time")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.savefig("beehive.png")
plt.show()
