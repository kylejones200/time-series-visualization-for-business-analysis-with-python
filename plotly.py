"""Generated from Jupyter notebook: plotly

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample energy demand data
np.random.seed(42)
dates = pd.date_range(
    start="2024-01-01", end="2024-12-31", freq="h"
)  # Notice the regular hyphens
demand = (
    1000
    + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    + np.random.normal(0, 50, len(dates))
)
df = pd.DataFrame({"timestamp": dates, "demand": demand})

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["demand"])
plt.title("Energy Demand Over Time (2024)")
plt.xlabel("Date")
plt.ylabel("Demand (units)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("energy_demand_plot.png")
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Generate data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
demand = (
    1000
    + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    + np.random.normal(0, 50, len(dates))
)
df = pd.DataFrame({"timestamp": dates, "demand": demand})

# Statistical analysis
decomposition = seasonal_decompose(df["demand"], period=24)
stationarity = adfuller(df["demand"])

# Create plot using statsmodels built-in plotting
fig = decomposition.plot()
fig.set_size_inches(12, 10)
plt.tight_layout()
plt.savefig("time_series_decomposition.png")
plt.show()

# Print stationarity test results
print("\nAugmented Dickey-Fuller Test Results:")
print(f"ADF Statistic: {stationarity[0]}")
print(f"p-value: {stationarity[1]}")
print("Critical Values:")
for key, value in stationarity[4].items():
    print(f"\t{key}: {value}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Generate data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
demand = (
    1000
    + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    + np.random.normal(0, 50, len(dates))
)
df = pd.DataFrame({"timestamp": dates, "demand": demand})


# Prepare features
def prepare_ml_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[["demand", "hour", "day_of_week"]])

    return pd.DataFrame(scaled_features, columns=["demand", "hour", "day_of_week"])


scaled_df = prepare_ml_features(df)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Box plot of demand by hour
plt.subplot(221)
sns.boxplot(x=df["hour"], y=df["demand"])
plt.title("Energy Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Demand")

# Plot 2: Box plot of demand by day of week
plt.subplot(222)
sns.boxplot(x=df["day_of_week"], y=df["demand"])
plt.title("Energy Demand by Day of Week")
plt.xlabel("Day of Week (0=Monday)")
plt.ylabel("Demand")

# Plot 3: Heatmap of correlations
plt.subplot(223)
correlation_matrix = scaled_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Scaled Features")

# Plot 4: Distribution of scaled demand
plt.subplot(224)
sns.histplot(scaled_df["demand"], kde=True)
plt.title("Distribution of Scaled Demand")
plt.xlabel("Standardized Demand")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("feature_analysis.png")
plt.show()

# Print some statistical information
print("\nScaled Features Statistics:")
print(scaled_df.describe())


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

# Generate data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
demand = (
    1000
    + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    + np.random.normal(0, 50, len(dates))
)
df = pd.DataFrame({"timestamp": dates, "demand": demand})


# Prophet forecast function
def prophet_forecast(df, periods=24):
    prophet_df = df.rename(columns={"timestamp": "ds", "demand": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq="h")
    forecast = model.predict(future)
    return forecast, model


# Generate forecast
prophet_results, m = prophet_forecast(df)

# Plot components
fig = m.plot_components(prophet_results)
fig.set_size_inches(15, 10)
plt.tight_layout()
plt.savefig("prophet_components.png")
plt.show()

# Print forecast metrics for next 24 hours
print("\nForecast Statistics (Next 24 Hours):")
print(
    prophet_results[prophet_results["ds"] > df["timestamp"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].describe()
)


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing

# Generate data
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
demand = (
    1000
    + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
    + np.random.normal(0, 50, len(dates))
)
df = pd.DataFrame({"timestamp": dates, "demand": demand})

# Convert to Darts TimeSeries
series = TimeSeries.from_dataframe(df, "timestamp", "demand")

# Create and train model
model = ExponentialSmoothing()
model.fit(series)

# Generate forecast
forecast = model.predict(24)

# Plotting
plt.figure(figsize=(15, 8))

# Plot actual data
series.plot(label="Actual")

# Plot forecast
forecast.plot(label="Forecast")

plt.title("Energy Demand Forecast (Exponential Smoothing)")
plt.xlabel("Time")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_comparison.png")
plt.show()

# Print forecast statistics
print("\nForecast Statistics:")
print(forecast.pd_dataframe().describe())


# --- code cell ---

# !pip install dask  # Jupyter-only


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Create figure with subplots
fig = plt.figure(figsize=(20, 30))  # Increased figure size

# Create GridSpec with more space between plots
gs = plt.GridSpec(4, 2, height_ratios=[1, 2, 2, 1], hspace=0.4, wspace=0.3)


def generate_data(start_date, periods=24 * 30):
    dates = pd.date_range(start=start_date, periods=periods, freq="h")
    demand = (
        1000
        + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
        + 100 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))
        + np.random.normal(0, 50, len(dates))
    )
    return pd.DataFrame({"timestamp": dates, "demand": demand})


def update(frame):
    plt.clf()  # Clear the entire figure

    # Generate new data with increasing timeframe
    start_date = pd.Timestamp("2024-01-01")
    current_date = start_date + pd.Timedelta(days=frame)
    df = generate_data(start_date, periods=(frame + 30) * 24)

    # Plot 1: Raw time series (spanning full width)
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(df["timestamp"], df["demand"], linewidth=2)
    ax1.set_title("Raw Time Series", fontsize=16, pad=20)
    ax1.tick_params(axis="both", labelsize=12)

    # Plot 2: Decomposition components
    decomposition = seasonal_decompose(df["demand"], period=24)

    # Trend
    ax2_trend = plt.subplot(gs[1, 0])
    ax2_trend.plot(df.index, decomposition.trend, linewidth=2)
    ax2_trend.set_title("Trend Component", fontsize=16, pad=20)
    ax2_trend.tick_params(axis="both", labelsize=12)

    # Seasonal
    ax2_seasonal = plt.subplot(gs[1, 1])
    ax2_seasonal.plot(df.index, decomposition.seasonal, linewidth=2)
    ax2_seasonal.set_title("Seasonal Component", fontsize=16, pad=20)
    ax2_seasonal.tick_params(axis="both", labelsize=12)

    # Create 2x2 grid for sklearn plots
    ax3_1 = plt.subplot(gs[2, 0])
    ax3_2 = plt.subplot(gs[2, 1])

    # Feature analysis plots
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    sns.boxplot(x="hour", y="demand", data=df, ax=ax3_1)
    ax3_1.set_title("Demand by Hour", fontsize=16, pad=20)
    ax3_1.tick_params(axis="both", labelsize=12)
    ax3_1.tick_params(axis="x", rotation=45)

    sns.boxplot(x="day_of_week", y="demand", data=df, ax=ax3_2)
    ax3_2.set_title("Demand by Day of Week", fontsize=16, pad=20)
    ax3_2.tick_params(axis="both", labelsize=12)

    # Plot 4: Prophet forecast (spanning full width)
    ax4 = plt.subplot(gs[3, :])

    prophet_df = df.rename(columns={"timestamp": "ds", "demand": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=24, freq="h")
    forecast = model.predict(future)

    ax4.plot(df["timestamp"], df["demand"], label="Actual", alpha=0.7, linewidth=2)
    ax4.plot(forecast["ds"], forecast["yhat"], label="Forecast", linewidth=2)
    ax4.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3
    )
    ax4.legend(fontsize=12)
    ax4.set_title("Prophet Forecast", fontsize=16, pad=20)
    ax4.tick_params(axis="both", labelsize=12)

    plt.suptitle(f"Time Series Analysis - Day {frame + 1}", fontsize=20, y=0.95)
    plt.tight_layout()


# Create animation with longer interval
anim = FuncAnimation(
    fig, update, frames=30, interval=2000, repeat=False
)  # Increased interval to 2 seconds

plt.show()

# Save animation with higher DPI
anim.save("time_series_animation.gif", writer="pillow", dpi=150)
