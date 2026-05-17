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

class HistogramVisualizer:
    def __init__(self):
        self.mean = 100
        self.std = 15
        self.n_total = 1000
        np.random.seed(42)
        self.all_data = np.random.normal(self.mean, self.std, self.n_total)
        self.count = 0
        self.current_value = None
        self.collected_data = []

    def create_animation(self):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        ax_count = fig.add_subplot(gs[0, 0])
        ax_current = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        def animate(frame):
            ax_count.clear()
            ax_current.clear()
            ax_hist.clear()
            self.count = frame + 1
            self.current_value = self.all_data[frame]
            self.collected_data.append(self.current_value)
            ax_count.text(
                0.5, 0.5, f"Count: {self.count}", ha="center", va="center", fontsize=20
            )
            ax_count.axis("off")
            ax_current.text(
                0.5,
                0.5,
                f"Current Value: {self.current_value:.2f}",
                ha="center",
                va="center",
                fontsize=20,
            )
            ax_current.axis("off")
            n_bins = int(np.sqrt(len(self.collected_data)))
            n, bins, patches = ax_hist.hist(
                self.collected_data,
                bins=max(10, n_bins),
                density=True,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            if len(self.collected_data) > 1:
                sample_mean = np.mean(self.collected_data)
                sample_std = np.std(self.collected_data)
                x = np.linspace(min(bins), max(bins), 100)
                normal_dist = (
                    1
                    / (sample_std * np.sqrt(2 * np.pi))
                    * np.exp(-((x - sample_mean) ** 2) / (2 * sample_std**2))
                )
                ax_hist.plot(x, normal_dist, "r-", lw=2, label="Normal Distribution")
            ax_hist.axvline(
                x=self.current_value, color="red", linestyle="--", alpha=0.5
            )
            ax_hist.set_title("Growing Histogram")
            ax_hist.set_xlabel("Value")
            ax_hist.set_ylabel("Density")
            if len(self.collected_data) > 1:
                stats_text = f"Mean: {sample_mean:.2f}\nStd Dev: {sample_std:.2f}"
                ax_hist.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax_hist.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=min(1000, self.n_total), interval=100, repeat=False
        )
        return anim

