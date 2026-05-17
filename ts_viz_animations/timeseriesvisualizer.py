"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from orbit.diagnostics.metrics import smape
from orbit.diagnostics.plot import plot_predicted_data
from orbit.models import DLT, KTR
from orbit.utils.dataset import load_iclaims
from pycaret.time_series import *
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

class TimeSeriesVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_points = 200
        self.time = np.linspace(0, 4 * np.pi, self.n_points)
        self.trend = 0.5 * self.time
        self.seasonality = 10 * np.sin(self.time)
        self.noise = np.random.normal(0, 1, self.n_points)
        self.data = self.trend + self.seasonality + self.noise
        self.calculate_derived_series()

    def calculate_derived_series(self):
        window = 20
        self.moving_avg = np.convolve(self.data, np.ones(window) / window, mode="valid")
        self.diff = np.diff(self.data)
        self.seasonal_pattern = 10 * np.sin(self.time)
        self.residuals = self.data - (self.trend + self.seasonal_pattern)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.clear()
            window = 50
            start_idx = frame % (self.n_points - window)
            end_idx = start_idx + window
            ax1.plot(
                self.time[start_idx:end_idx],
                self.data[start_idx:end_idx],
                "b-",
                label="Original Series",
            )
            ax1.set_title("Raw Time Series")
            ax1.legend()
            ax1.grid(True)
            if start_idx >= window:
                ax2.plot(
                    self.time[start_idx:end_idx],
                    self.data[start_idx:end_idx],
                    "b-",
                    alpha=0.5,
                    label="Original",
                )
                ax2.plot(
                    self.time[start_idx:end_idx],
                    self.moving_avg[start_idx - window : end_idx - window],
                    "r-",
                    label="Moving Average",
                )
            ax2.set_title("Moving Average")
            ax2.legend()
            ax2.grid(True)
            if start_idx < len(self.diff):
                ax3.plot(
                    self.time[start_idx + 1 : end_idx],
                    self.diff[start_idx : end_idx - 1],
                    "g-",
                    label="First Difference",
                )
                ax3.axhline(y=0, color="k", linestyle="--")
            ax3.set_title("First Difference")
            ax3.legend()
            ax3.grid(True)
            ax4.plot(
                self.time[start_idx:end_idx],
                self.trend[start_idx:end_idx],
                "r-",
                label="Trend",
            )
            ax4.plot(
                self.time[start_idx:end_idx],
                self.seasonal_pattern[start_idx:end_idx],
                "g-",
                label="Seasonal",
            )
            ax4.plot(
                self.time[start_idx:end_idx],
                self.residuals[start_idx:end_idx],
                "b-",
                alpha=0.5,
                label="Residual",
            )
            ax4.set_title("Decomposition")
            ax4.legend()
            ax4.grid(True)
            ax5.hist(self.data[start_idx:end_idx], bins=20, density=True, alpha=0.7)
            ax5.set_title("Distribution")
            ax5.grid(True)
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=self.n_points, interval=100, repeat=True
        )
        return anim

