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

class MFLEVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_series = 5
        self.n_timepoints = 200
        self.n_components = 3
        t = np.linspace(0, 4 * np.pi, self.n_timepoints)
        trend = np.linspace(0, 2, self.n_timepoints)
        seasonal = np.sin(t)
        cyclical = np.cos(t / 2)
        self.data = np.zeros((self.n_series, self.n_timepoints))
        for i in range(self.n_series):
            weights = np.random.rand(3)
            self.data[i] = (
                weights[0] * trend
                + weights[1] * seasonal
                + weights[2] * cyclical
                + np.random.normal(0, 0.1, self.n_timepoints)
            )
        self.svd = TruncatedSVD(n_components=self.n_components)
        self.latent_features = self.svd.fit_transform(self.data)
        self.components = self.svd.components_
        self.reconstructed = self.latent_features @ self.components

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])

        def animate(frame):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            window = 50
            start_idx = frame % (self.n_timepoints - window)
            end_idx = start_idx + window
            for i in range(self.n_series):
                ax1.plot(
                    range(start_idx, end_idx),
                    self.data[i, start_idx:end_idx],
                    label=f"Series {i + 1}",
                    alpha=0.7,
                )
            ax1.set_title("Original Time Series")
            ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
            for i in range(self.n_components):
                ax2.plot(
                    range(start_idx, end_idx),
                    self.components[i, start_idx:end_idx],
                    label=f"Component {i + 1}",
                )
            ax2.set_title("Latent Components")
            ax2.legend()
            error = np.mean((self.data - self.reconstructed) ** 2, axis=0)
            ax3.plot(range(start_idx, end_idx), error[start_idx:end_idx], color="red")
            ax3.set_title("Reconstruction Error")
            ax3.set_ylabel("MSE")
            for i in range(self.n_series):
                ax4.plot(
                    range(start_idx, end_idx),
                    self.reconstructed[i, start_idx:end_idx],
                    label=f"Reconstructed {i + 1}",
                    alpha=0.7,
                )
            ax4.set_title("Reconstructed Time Series")
            ax4.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=self.n_timepoints - 50, interval=200, repeat=True
        )
        return anim

