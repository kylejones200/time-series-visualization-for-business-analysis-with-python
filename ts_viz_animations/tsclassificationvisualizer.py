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

class TSClassificationVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_timesteps = 100
        t = np.linspace(0, 4 * np.pi, self.n_timesteps)
        self.class1 = np.array(
            [
                np.sin(t) + np.random.normal(0, 0.2, self.n_timesteps)
                for _ in range(self.n_samples)
            ]
        )
        self.class2 = np.array(
            [
                np.where(np.sin(t) > 0, 1, -1)
                + np.random.normal(0, 0.2, self.n_timesteps)
                for _ in range(self.n_samples)
            ]
        )
        self.prepare_data()

    def prepare_data(self):
        self.features1 = self.extract_features(self.class1)
        self.features2 = self.extract_features(self.class2)
        self.dtw_matrix = self.compute_dtw_matrix(self.class1[0], self.class2[0])
        self.cnn_features = self.simulate_cnn_features()

    def extract_features(self, data):
        return np.array(
            [
                [
                    np.mean(series),
                    np.std(series),
                    np.percentile(series, 75) - np.percentile(series, 25),
                ]
                for series in data
            ]
        )

    def compute_dtw_matrix(self, s1, s2):
        n, m = (len(s1), len(s2))
        dtw_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                cost = (s1[i] - s2[j]) ** 2
                if i > 0 and j > 0:
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    )
                elif i > 0:
                    dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j]
                elif j > 0:
                    dtw_matrix[i, j] = cost + dtw_matrix[i, j - 1]
                else:
                    dtw_matrix[i, j] = cost
        return dtw_matrix

    def simulate_cnn_features(self):
        features = []
        for i in range(self.n_timesteps - 10):
            feature = np.mean(self.class1[0][i : i + 10])
            features.append(feature)
        return np.array(features)

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 50
            start_idx = frame % (self.n_timesteps - window)
            end_idx = start_idx + window
            ax1.plot(self.class1[0][start_idx:end_idx], label="Class 1", color="blue")
            ax1.plot(self.class2[0][start_idx:end_idx], label="Class 2", color="red")
            if frame % 3 == 0:
                current_window = self.class1[0][start_idx:end_idx]
                ax1.axhline(
                    y=np.mean(current_window),
                    color="green",
                    linestyle="--",
                    label="Mean",
                )
                ax1.axhspan(
                    np.percentile(current_window, 25),
                    np.percentile(current_window, 75),
                    color="yellow",
                    alpha=0.3,
                    label="IQR",
                )
            ax1.set_title("Time Series Classification")
            ax1.legend()
            im = ax2.imshow(
                self.dtw_matrix[:end_idx, :end_idx], aspect="auto", cmap="viridis"
            )
            ax2.set_title("DTW Alignment Matrix")
            plt.colorbar(im, ax=ax2)
            ax3.plot(self.cnn_features[:end_idx], label="CNN Features", color="purple")
            ax3.axvline(x=frame, color="red", linestyle="--")
            ax3.set_title("Deep Learning Features")
            ax3.legend()
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=self.n_timesteps, interval=100, repeat=True
        )
        return anim

