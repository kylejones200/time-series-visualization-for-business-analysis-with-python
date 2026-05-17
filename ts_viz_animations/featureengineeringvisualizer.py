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

class FeatureEngineeringVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_points = 200
        self.time = np.linspace(0, 4 * np.pi, self.n_points)
        self.trend = 0.5 * self.time
        self.seasonality = 10 * np.sin(self.time)
        self.noise = np.random.normal(0, 1, self.n_points)
        self.data = self.trend + self.seasonality + self.noise
        self.calculate_features()

    def calculate_features(self):
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data.reshape(-1, 1)).flatten()
        self.diff = np.diff(self.data)
        self.rolling_mean = pd.Series(self.data).rolling(window=20).mean()
        self.rolling_std = pd.Series(self.data).rolling(window=20).std()
        self.roc = np.gradient(self.data)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
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
                label="Original",
                color="blue",
            )
            ax1.plot(
                self.time[start_idx:end_idx],
                self.trend[start_idx:end_idx],
                label="Trend",
                color="red",
            )
            ax1.plot(
                self.time[start_idx:end_idx],
                self.seasonality[start_idx:end_idx],
                label="Seasonal",
                color="green",
            )
            ax1.set_title("Time Series Components")
            ax1.legend()
            ax2.plot(
                self.time[start_idx:end_idx],
                self.data[start_idx:end_idx],
                label="Original",
                alpha=0.5,
            )
            ax2.plot(
                self.time[start_idx:end_idx],
                self.scaled_data[start_idx:end_idx],
                label="Scaled",
            )
            ax2.set_title("Min-Max Scaling")
            ax2.legend()
            if start_idx < len(self.diff):
                ax3.plot(
                    self.time[start_idx + 1 : end_idx],
                    self.diff[start_idx : end_idx - 1],
                    label="First Difference",
                )
                ax3.axhline(y=0, color="r", linestyle="--")
            ax3.set_title("First Difference")
            ax3.legend()
            ax4.plot(
                self.time[start_idx:end_idx],
                self.data[start_idx:end_idx],
                label="Original",
                alpha=0.5,
            )
            ax4.plot(
                self.time[start_idx:end_idx],
                self.rolling_mean[start_idx:end_idx],
                label="Rolling Mean",
            )
            ax4.fill_between(
                self.time[start_idx:end_idx],
                self.rolling_mean[start_idx:end_idx]
                - self.rolling_std[start_idx:end_idx],
                self.rolling_mean[start_idx:end_idx]
                + self.rolling_std[start_idx:end_idx],
                alpha=0.2,
            )
            ax4.set_title("Rolling Statistics")
            ax4.legend()
            ax5.plot(
                self.time[start_idx:end_idx],
                self.roc[start_idx:end_idx],
                label="Rate of Change",
            )
            ax5.axhline(y=0, color="r", linestyle="--")
            ax5.set_title("Rate of Change")
            ax5.legend()
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=self.n_points, interval=200, repeat=True
        )
        return anim

