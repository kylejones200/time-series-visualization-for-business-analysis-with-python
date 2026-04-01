import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from PIL import Image
from statsmodels.tsa.seasonal import seasonal_decompose


def set_visualization_style():
    """
    Set global matplotlib visualization style parameters.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
        }
    )


def set_plot_style(ax, data: pd.DataFrame, time_column, value_columns):
    """
    Set the style for a given plot axis based on the dataframe content.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to style.
        data (pandas.DataFrame): DataFrame containing time and value columns.
        time_column (str): The name of the time column in the DataFrame.
        value_columns (list): List of value column names.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))

    data[time_column] = pd.to_datetime(data[time_column])  # Ensure datetime format
    time_range = data[time_column].max() - data[time_column].min()

    # Adjust X-Axis ticks dynamically
    if time_range < pd.Timedelta(days=365):  # If data is shorter than 1 year
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif time_range < pd.Timedelta(days=365 * 10):  # If data spans less than 10 years
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:  # For longer datasets, use 50-year intervals
        ax.xaxis.set_major_locator(mdates.YearLocator(50))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_xlim(data[time_column].min(), data[time_column].max())

    # Y-Axis scaling based on percentiles
    all_values = np.concatenate([data[col].dropna().values for col in value_columns])
    y_20, y_mean, y_80 = np.percentile(all_values, [20, 50, 80])

    ax.set_yticks([y_20, y_mean, y_80])
    ax.set_yticklabels(
        [f"{int(y_20)}", f"{int(y_mean)}", f"{int(y_80)}"]
    )  # Force integer labels

    # Force plain integer formatting for Y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Force integer formatting for X-axis if numeric
    if data[time_column].dtype in [np.int64, np.float64]:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))


def plot_time_series(
    data: pd.DataFrame,
    time_column,
    value_columns,
    title: str = "Time Series Plot",
    filename=None,
):
    """
    Plot time series data from a DataFrame.

    Parameters:
        data (pandas.DataFrame): DataFrame containing time and value data.
        time_column (str): The name of the time column.
        value_columns (list): List of column names to plot.
        title (str, optional): Title of the plot.
        filename (str, optional): Filename to save the plot.
    """
    set_visualization_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(value_columns)))

    data[time_column] = pd.to_datetime(data[time_column])  # Ensure datetime format

    for i, col in enumerate(value_columns):
        ax.plot(data[time_column], data[col], linewidth=2, color=colors[i])
        last_x = data[time_column].iloc[-1] + pd.Timedelta(days=10)
        last_y = data[col].iloc[-1]
        ax.text(
            last_x,
            last_y,
            col,
            fontsize=12,
            color=colors[i],
            verticalalignment="center",
        )

    set_plot_style(ax, data, time_column, value_columns)

    if title:
        ax.set_title(title)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_decomposition(
    data: pd.Series, model: str = "additive", title: str = "Time Series Decomposition"
):
    """
    Perform and plot seasonal decomposition of a time series.

    Parameters:
        data (pandas.Series): The time series data to decompose.
        model (str): Type of seasonal component ('additive' or 'multiplicative').
        title (str): Title for the decomposition plots.
    """
    set_visualization_style()
    period = max(2, len(data) // 10)
    decomposition = seasonal_decompose(data, model=model, period=period)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(data, label="Original", color="black")
    axes[0].set_title("Original Series")

    axes[1].plot(decomposition.trend, label="Trend", color="black")
    axes[1].set_title("Trend")

    axes[2].plot(decomposition.seasonal, label="Seasonal", color="black")
    axes[2].set_title("Seasonal")

    axes[3].plot(decomposition.resid, label="Residual", color="black")
    axes[3].set_title("Residual")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def plot_clusters(model, title: str = "Time Series Cluster"):
    """
    Plot cluster centroids for a given clustering model.

    Parameters:
        model: A clustering model with an attribute `cluster_centers_`.
        title (str): Title of the plot.
    """
    set_visualization_style()
    plt.figure(figsize=(10, 4))
    for centroid in model.cluster_centers_:
        plt.plot(centroid.ravel(), linewidth=2)
    plt.title(title)
    plt.show()


def plot_sample_time_series(X, title: str = "Time Series Data"):
    """
    Plot a few sample time series from the dataset.

    Parameters:
        X (iterable): An iterable of time series arrays.
        title (str): Title of the plot.
    """
    set_visualization_style()
    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(X))):  # Plot first 5 series
        plt.plot(X[i].ravel(), linewidth=2, label=f"Series {i + 1}")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage for time series plotting and decomposition
    time = pd.date_range(start="1950-01-01", periods=100, freq="Y")
    data = pd.DataFrame({"Date": time, "Value": np.cumsum(np.random.randn(100))})

    plot_time_series(data, "Date", ["Value"], title="Sample Time Series")
    plot_decomposition(data["Value"], model="additive", title="Sample Decomposition")
