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

def generate_two_related_time_series() -> None:
    "\nworks\n"

    visualizer = CausalInferenceVisualizer()

    anim = visualizer.create_animation()

    anim.save("causal_inference.gif", writer="pillow", fps=10)

    plt.close()


def generate_synthetic_multivariate_time_series_data() -> None:
    visualizer = MFLEVisualizer()

    anim = visualizer.create_animation()

    anim.save("mfle_visualization.gif", writer="pillow", fps=10)

    plt.close()


def generate_simulated_data() -> None:
    visualizer = TimeSeriesModelComparison()

    anim = visualizer.create_animation()

    anim.save("time_series_models.gif", writer="pillow", fps=10)

    plt.close()


def generate_synthetic_data() -> None:
    visualizer = ExponentialSmoothingVisualizer()

    anim = visualizer.create_animation()

    anim.save("exponential_smoothing.gif", writer="pillow", fps=5)

    plt.close()


def generate_simulated_data_2() -> None:
    visualizer = TimeSeriesModelComparison()

    anim = visualizer.create_animation()

    anim.save("time_series_models.gif", writer="pillow", fps=10)

    plt.close()


def generate_simulated_data_3() -> None:
    visualizer = TimeSeriesModelComparison()

    anim = visualizer.create_animation()

    anim.save("time_series_models.gif", writer="pillow", fps=5)

    plt.close()


def generate_synthetic_time_series_data() -> None:
    visualizer = FeatureEngineeringVisualizer()

    anim = visualizer.create_animation()

    anim.save("feature_engineering.gif", writer="pillow", fps=5)

    plt.close()


def generate_source_domain_data_e_g_office_energy_co() -> None:
    visualizer = TransferLearningVisualizer()

    anim = visualizer.create_animation()

    anim.save("transfer_learning.gif", writer="pillow", fps=10)

    plt.close()


def generate_synthetic_time_series_data_for_differen() -> None:
    visualizer = TSClassificationVisualizer()

    anim = visualizer.create_animation()

    anim.save("ts_classification.gif", writer="pillow", fps=10)

    plt.close()


def generate_synthetic_time_series_data_2() -> None:
    visualizer = TimeSeriesVisualizer()

    anim = visualizer.create_animation()

    anim.save("time_series_viz.gif", writer="pillow", fps=10)

    plt.close()


def parameters_for_normal_distribution() -> None:
    visualizer = HistogramVisualizer()

    anim = visualizer.create_animation()

    anim.save("growing_histogram.gif", writer="pillow", fps=10)

    plt.close()


def generate_source_domain_data() -> None:
    visualizer = TransferLearningVisualizer()

    anim = visualizer.create_animation()

    anim.save("transfer_learning.gif", writer="pillow", fps=20)

    plt.close()


def create_sample_data() -> None:
    visualizer = PyCaretTimeSeriesVisualizer()

    anim = visualizer.create_animation()

    anim.save("pycaret_time_series.gif", writer="pillow", fps=5)

    plt.close()


def generate_sample_data() -> None:
    visualizer = VolatilityVisualizer()

    anim = visualizer.create_animation()

    anim.save("volatility_analysis.gif", writer="pillow", fps=10, dpi=150)

    plt.close()

    print("Animation has been saved as 'volatility_analysis.gif'")


def main() -> None:
    generate_simulated_process_data()
    generate_two_related_time_series()
    generate_synthetic_multivariate_time_series_data()
    generate_simulated_data()
    generate_synthetic_data()
    generate_simulated_data_2()
    generate_simulated_data_3()
    generate_synthetic_time_series_data()
    generate_source_domain_data_e_g_office_energy_co()
    generate_synthetic_time_series_data_for_differen()
    generate_synthetic_time_series_data_2()
    parameters_for_normal_distribution()
    generate_source_domain_data()
    create_sample_data()
    load_and_prepare_data()
    load_and_prepare_data_2()
    generate_sample_data()

