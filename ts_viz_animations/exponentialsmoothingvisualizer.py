"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pycaret.time_series import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


class ExponentialSmoothingVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_points = 100
        self.time = np.arange(self.n_points)
        self.simple_data = 10 + np.random.normal(scale=0.5, size=self.n_points)
        self.trend_data = (
            10 + 0.5 * self.time + np.random.normal(scale=1.0, size=self.n_points)
        )
        self.seasonal_data = (
            10
            + 0.5 * self.time
            + 2 * np.sin(2 * np.pi * self.time / 12)
            + np.random.normal(scale=1.0, size=self.n_points)
        )
        self.fit_models()

    def fit_models(self):
        ses_model = SimpleExpSmoothing(self.simple_data)
        self.ses_fit = ses_model.fit(smoothing_level=0.5)
        des_model = ExponentialSmoothing(self.trend_data, trend="add")
        self.des_fit = des_model.fit(smoothing_level=0.5, smoothing_trend=0.5)
        hw_model = ExponentialSmoothing(
            self.seasonal_data, trend="add", seasonal="add", seasonal_periods=12
        )
        self.hw_fit = hw_model.fit(
            smoothing_level=0.5, smoothing_trend=0.5, smoothing_seasonal=0.5
        )

    def create_animation(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        fig.suptitle("Exponential Smoothing Methods", fontsize=16)

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 30
            start_idx = frame % (self.n_points - window)
            end_idx = start_idx + window
            ax1.plot(
                self.time[start_idx:end_idx],
                self.simple_data[start_idx:end_idx],
                "o-",
                label="Original",
                alpha=0.5,
            )
            ax1.plot(
                self.time[start_idx:end_idx],
                self.ses_fit.fittedvalues[start_idx:end_idx],
                "r-",
                label="SES",
                linewidth=2,
            )
            ax1.set_title("Simple Exponential Smoothing")
            ax1.legend()
            ax1.grid(False)
            ax2.plot(
                self.time[start_idx:end_idx],
                self.trend_data[start_idx:end_idx],
                "o-",
                label="Original",
                alpha=0.5,
            )
            ax2.plot(
                self.time[start_idx:end_idx],
                self.des_fit.fittedvalues[start_idx:end_idx],
                "r-",
                label="DES",
                linewidth=2,
            )
            ax2.set_title("Double Exponential Smoothing (with Trend)")
            ax2.legend()
            ax2.grid(False)
            ax3.plot(
                self.time[start_idx:end_idx],
                self.seasonal_data[start_idx:end_idx],
                "o-",
                label="Original",
                alpha=0.5,
            )
            ax3.plot(
                self.time[start_idx:end_idx],
                self.hw_fit.fittedvalues[start_idx:end_idx],
                "r-",
                label="Holt-Winters",
                linewidth=2,
            )
            ax3.set_title("Triple Exponential Smoothing (Holt-Winters)")
            ax3.legend()
            ax3.grid(False)
            ax1.text(
                0.02,
                0.98,
                "Handles constant level with noise",
                transform=ax1.transAxes,
                verticalalignment="top",
            )
            ax2.text(
                0.02,
                0.98,
                "Handles trend + noise",
                transform=ax2.transAxes,
                verticalalignment="top",
            )
            ax3.text(
                0.02,
                0.98,
                "Handles trend + seasonality + noise",
                transform=ax3.transAxes,
                verticalalignment="top",
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        anim = FuncAnimation(
            fig, animate, frames=self.n_points, interval=500, repeat=True
        )
        return anim
