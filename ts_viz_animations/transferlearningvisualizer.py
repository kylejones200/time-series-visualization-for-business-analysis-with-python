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

class TransferLearningVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n_points = 200
        self.time = np.linspace(0, 4 * np.pi, self.n_points)
        self.source_trend = np.linspace(10, 30, self.n_points)
        self.source_seasonal = 10 * np.sin(self.time) + 5 * np.cos(self.time * 2)
        self.source_noise = np.random.normal(0, 1, self.n_points)
        self.source_data = self.source_trend + self.source_seasonal + self.source_noise
        self.target_trend = np.linspace(5, 15, self.n_points)
        self.target_seasonal = 5 * np.sin(self.time) + 2.5 * np.cos(self.time * 2)
        self.target_noise = np.random.normal(0, 0.5, self.n_points)
        self.target_data = self.target_trend + self.target_seasonal + self.target_noise
        self.adaptation_steps = []
        self.errors = []
        self.simulate_transfer_learning()

    def simulate_transfer_learning(self):
        n_steps = 50
        alpha = np.linspace(0, 1, n_steps)
        for a in alpha:
            adapted_trend = self.source_trend * (1 - a) + self.target_trend * a
            adapted_seasonal = self.source_seasonal * (1 - a) + self.target_seasonal * a
            adapted = adapted_trend + adapted_seasonal + self.target_noise
            self.adaptation_steps.append(adapted)
            error = np.mean((adapted - self.target_data) ** 2)
            self.errors.append(error)
        self.adaptation_steps = np.array(self.adaptation_steps)
        self.errors = np.array(self.errors)

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
            start_idx = max(0, frame - window)
            end_idx = min(self.n_points, frame + 1)
            ax1.plot(
                self.time[start_idx:end_idx],
                self.source_data[start_idx:end_idx],
                "b-",
                label="Source Domain",
                alpha=0.5,
            )
            ax1.plot(
                self.time[start_idx:end_idx],
                self.target_data[start_idx:end_idx],
                "g-",
                label="Target Domain",
                alpha=0.5,
            )
            adaptation_idx = min(frame // 4, len(self.adaptation_steps) - 1)
            if adaptation_idx >= 0:
                ax1.plot(
                    self.time[start_idx:end_idx],
                    self.adaptation_steps[adaptation_idx][start_idx:end_idx],
                    "r-",
                    label="Adapted Model",
                    linewidth=2,
                )
            ax1.set_title("Time Series Domains")
            ax1.legend()
            ax1.grid(False)
            delay = 5
            source_features = np.column_stack(
                (self.source_data[:-delay], self.source_data[delay:])
            )
            target_features = np.column_stack(
                (self.target_data[:-delay], self.target_data[delay:])
            )
            ax2.scatter(
                source_features[:, 0],
                source_features[:, 1],
                c="blue",
                alpha=0.3,
                label="Source Features",
            )
            ax2.scatter(
                target_features[:, 0],
                target_features[:, 1],
                c="green",
                alpha=0.3,
                label="Target Features",
            )
            if adaptation_idx >= 0:
                adapted_features = np.column_stack(
                    (
                        self.adaptation_steps[adaptation_idx][:-delay],
                        self.adaptation_steps[adaptation_idx][delay:],
                    )
                )
                ax2.scatter(
                    adapted_features[:, 0],
                    adapted_features[:, 1],
                    c="red",
                    alpha=0.5,
                    label="Adapted Features",
                )
            ax2.set_title("Feature Space")
            ax2.legend()
            ax2.grid(False)
            if adaptation_idx >= 0:
                current_errors = self.errors[: adaptation_idx + 1]
                progress = np.linspace(0, 1, len(current_errors))
                ax3.plot(progress, current_errors, "r-", label="Adaptation Error")
                ax3.set_title("Learning Progress")
                ax3.set_xlabel("Adaptation Progress")
                ax3.set_ylabel("Mean Squared Error")
                ax3.set_ylim(0, max(self.errors) * 1.1)
                ax3.legend()
                ax3.grid(False)
            plt.tight_layout()

        anim = FuncAnimation(
            fig,
            animate,
            frames=self.n_points + len(self.adaptation_steps),
            interval=50,
            repeat=True,
        )
        return anim

