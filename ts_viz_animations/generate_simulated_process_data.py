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

def generate_simulated_process_data() -> None:
    np.random.seed(42)

    time = pd.date_range(start="2023-01-01", periods=100, freq="D")

    values = np.random.normal(50, 2, 100)

    values[30:35] += 8

    values[70:75] -= 8

    df = pd.DataFrame({"Time": time, "Value": values})

    mean = df["Value"].mean()

    std_dev = df["Value"].std()

    ucl = mean + 3 * std_dev

    lcl = mean - 3 * std_dev

    plt.figure(figsize=(12, 6))

    plt.plot(df["Time"], df["Value"], label="Process Data", marker="o", linestyle="-")

    plt.axhline(mean, color="blue", linestyle="--", label="Mean")

    plt.axhline(ucl, color="red", linestyle="--", label="Upper Control Limit (UCL)")

    plt.axhline(lcl, color="red", linestyle="--", label="Lower Control Limit (LCL)")

    out_of_control = (df["Value"] > ucl) | (df["Value"] < lcl)

    plt.scatter(
        df["Time"][out_of_control],
        df["Value"][out_of_control],
        color="red",
        label="Out of Control",
    )

    plt.fill_between(
        df["Time"],
        ucl,
        lcl,
        where=(df["Value"] > ucl) | (df["Value"] < lcl),
        color="red",
        alpha=0.1,
    )

    title = "Control Chart with Out-of-Control Areas"

    plt.title(title)

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.grid()

    plt.savefig(f"{title}.png")

    plt.show()

