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

class PyCaretTimeSeriesVisualizer:
    def __init__(self):
        self.data = pd.Series(
            [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 10,
            name="Sales",
        )
        self.data.index = pd.date_range(
            start="2010-01-01", periods=len(self.data), freq="M"
        )
        self.df = self.data.to_frame()
        self.setup_and_train_models()

    def setup_and_train_models(self):
        self.setup_data = setup(
            data=self.df,
            target="Sales",
            session_id=123,
            seasonal_period=12,
            silent=True,
            verbose=False,
        )
        self.models = {}
        model_types = ["arima", "ets", "prophet", "theta"]
        for model_type in model_types:
            model = create_model(model_type)
            self.models[model_type] = model
        self.forecasts = {}
        for name, model in self.models.items():
            forecast = predict_model(model)
            self.forecasts[name] = forecast

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            window = 24
            start_idx = frame % (len(self.data) - window)
            end_idx = start_idx + window
            current_model = list(self.models.keys())[frame // 40 % len(self.models)]
            ax1.plot(
                self.data.index[start_idx:end_idx],
                self.data.values[start_idx:end_idx],
                "b-",
                label="Actual",
                alpha=0.7,
            )
            forecast = self.forecasts[current_model]
            ax1.plot(
                forecast.index[start_idx:end_idx],
                forecast["y_pred"].values[start_idx:end_idx],
                "r-",
                label=f"{current_model.upper()} Forecast",
            )
            ax1.set_title(f"Time Series with {current_model.upper()}")
            ax1.legend()
            ax1.grid(True)
            ax2.plot(
                self.data.index[start_idx:end_idx],
                self.data.values[start_idx:end_idx],
                "k--",
                label="Actual",
                alpha=0.5,
            )
            for name, forecast in self.forecasts.items():
                ax2.plot(
                    forecast.index[start_idx:end_idx],
                    forecast["y_pred"].values[start_idx:end_idx],
                    label=name.upper(),
                    alpha=0.5,
                )
            ax2.set_title("Model Comparison")
            ax2.legend()
            ax2.grid(True)
            metrics = {}
            for name, forecast in self.forecasts.items():
                metrics[name] = np.mean((forecast["y_pred"] - self.data) ** 2)
            names = list(metrics.keys())
            values = list(metrics.values())
            bars = ax3.bar(names, values)
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )
            ax3.set_title("Mean Squared Error")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            ax3.grid(False)
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=len(self.models) * 40, interval=200, repeat=True
        )
        return anim

