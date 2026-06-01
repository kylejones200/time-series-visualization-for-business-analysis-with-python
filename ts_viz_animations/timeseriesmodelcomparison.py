"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from pycaret.time_series import *
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesModelComparison:
    def __init__(self):
        np.random.seed(42)
        self.n = 200
        self.time = pd.date_range(start="2023-01-01", periods=self.n, freq="D")
        self.trend = np.linspace(10, 50, self.n)
        self.seasonality = 10 * np.sin(np.linspace(0, 2 * np.pi, self.n))
        self.noise = np.random.normal(0, 2, self.n)
        self.data = self.trend + self.seasonality + self.noise
        self.prepare_data_and_models()

    def prepare_data_and_models(self):
        self.df = pd.DataFrame({"value": self.data}, index=self.time)
        self.train = self.df.iloc[:-30]
        self.test = self.df.iloc[-30:]
        arima_model = ARIMA(self.train, order=(2, 1, 2))
        self.arima_result = arima_model.fit()
        self.arima_forecast = self.arima_result.forecast(steps=30)
        hw_model = ExponentialSmoothing(
            self.train, seasonal="add", seasonal_periods=30
        ).fit()
        self.hw_forecast = hw_model.forecast(30)

    def create_animation(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Time Series Model Comparison", fontsize=16)

        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            window = 50
            start_idx = frame // 2 % (len(self.df) - window)
            end_idx = start_idx + window
            ax1.plot(
                self.df.index[start_idx:end_idx],
                self.df.values[start_idx:end_idx],
                label="Original",
                color="blue",
            )
            ax1.plot(
                self.df.index[start_idx:end_idx],
                self.trend[start_idx:end_idx],
                label="Trend",
                color="red",
            )
            ax1.plot(
                self.df.index[start_idx:end_idx],
                self.seasonality[start_idx:end_idx],
                label="Seasonal",
                color="green",
            )
            ax1.set_title("Time Series Components")
            ax1.legend()
            train_end = len(self.train)
            visible_start = max(start_idx, train_end - window)
            ax2.plot(
                self.train.index[visible_start:train_end],
                self.train.values[visible_start:train_end],
                label="Training",
                color="blue",
            )
            ax2.plot(self.test.index, self.test.values, label="Actual", color="green")
            ax2.plot(
                self.test.index,
                self.arima_forecast,
                label="ARIMA Forecast",
                color="red",
            )
            ax2.set_title("ARIMA Forecast")
            ax2.legend()
            ax3.plot(
                self.train.index[visible_start:train_end],
                self.train.values[visible_start:train_end],
                label="Training",
                color="blue",
            )
            ax3.plot(self.test.index, self.test.values, label="Actual", color="green")
            ax3.plot(
                self.test.index, self.hw_forecast, label="Holt-Winters", color="red"
            )
            ax3.set_title("Holt-Winters Forecast")
            ax3.legend()
            test_window = 20
            test_start = max(0, frame // 2 % (len(self.test) - test_window))
            test_end = test_start + test_window
            ax4.plot(
                self.test.index[test_start:test_end],
                self.test.values[test_start:test_end],
                label="Actual",
                color="blue",
            )
            ax4.plot(
                self.test.index[test_start:test_end],
                self.arima_forecast[test_start:test_end],
                label="ARIMA",
                color="red",
            )
            ax4.plot(
                self.test.index[test_start:test_end],
                self.hw_forecast[test_start:test_end],
                label="Holt-Winters",
                color="green",
            )
            ax4.set_title("Model Comparison")
            ax4.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis="x", rotation=45)

        anim = FuncAnimation(
            fig, animate, frames=(len(self.df) - 50) * 2, interval=500, repeat=True
        )
        return anim
