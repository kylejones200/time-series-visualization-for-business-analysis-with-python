"""Generated from Jupyter notebook: Line Plot with Moving Average

Magics and shell lines are commented out. Run with a normal Python interpreter."""



def main():
    # --- code cell ---

    """
    This dataset provides a simple time series to demonstrate different visualization methods.
    Enhanced with timesmith and plotsmith for streamlined time series operations and visualizations.
    """

    import numpy as np
    import pandas as pd
    import plotsmith as ps
    import timesmith as ts

    np.random.seed(42)
    time = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = {
        "Date": time,
        "Value": 100
        + 2 * np.arange(100)
        + np.sin(np.linspace(0, 10, 100)) * 10
        + np.random.normal(0, 5, 100),
    }
    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)  # Set Date as index for timesmith compatibility


    # --- code cell ---

    # Use timesmith's RollingFeaturizer to calculate moving average
    # RollingFeaturizer adds enhanced rolling statistics to pandas
    rolling_feat = ts.RollingFeaturizer(windows=[7], functions=["mean"])
    rolling_features = rolling_feat.fit_transform(df[["Value"]])
    df["Moving_Avg"] = rolling_features["rolling_mean_7"]  # Extract the 7-day mean

    # Use plotsmith for enhanced time series visualization
    # Ensure the DataFrame has the datetime index preserved
    plot_df = pd.DataFrame(
        {"Original Data": df["Value"], "7-Day Moving Average": df["Moving_Avg"]},
        index=df.index,
    )

    ps.plot_timeseries(
        plot_df,
        title="Time Series with Moving Average",
        xlabel="Date",
        ylabel="Value",
        figsize=(10, 6),
    )


    # --- code cell ---

    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposed = seasonal_decompose(df["Value"], period=30, model="additive")

    # Use plotsmith to visualize each component
    # Ensure the DataFrame has the datetime index preserved
    decomp_df = pd.DataFrame(
        {
            "Trend": decomposed.trend,
            "Seasonal": decomposed.seasonal,
            "Residual": decomposed.resid,
        },
        index=df.index,
    )

    # Plot each component separately with plotsmith
    for component in ["Trend", "Seasonal", "Residual"]:
        ps.plot_timeseries(
            decomp_df[[component]],
            title=f"{component} Component",
            xlabel="Date",
            ylabel=component,
            figsize=(10, 3),
        )


    # --- code cell ---

    # Prepare data for heatmap
    df_reset = df.reset_index()
    df_reset["Day_of_Week"] = df_reset["Date"].dt.day_name()
    df_reset["Month"] = df_reset["Date"].dt.month
    pivot_table = df_reset.pivot_table(
        values="Value", index="Day_of_Week", columns="Month", aggfunc="mean"
    )

    # Use plotsmith's plot_heatmap for enhanced visualization
    ps.plot_heatmap(
        pivot_table,
        title="Heatmap of Daily Values",
        xlabel="Month",
        ylabel="Day of Week",
        figsize=(10, 6),
    )


    # --- code cell ---

    # Create a second series
    df["Value_2"] = df["Value"] + np.random.normal(loc=0, scale=10, size=len(df))

    # Use plotsmith to plot multiple series together
    # Ensure the DataFrame has the datetime index preserved
    multi_series_df = pd.DataFrame(
        {"Series 1": df["Value"], "Series 2": df["Value_2"]}, index=df.index
    )

    ps.plot_timeseries(
        multi_series_df,
        title="Multiple Time Series Comparison",
        xlabel="Date",
        ylabel="Value",
        figsize=(10, 6),
    )


    # --- code cell ---

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Value"], mode="lines", name="Value"))
    fig.add_annotation(
        x=df.index[50],
        y=df["Value"].iloc[50],
        text="Notable Point",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title="Interactive Line Plot", xaxis_title="Date", yaxis_title="Value"
    )
    fig.show()


    # --- code cell ---

    # Dual axis plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Value"], name="Series 1", yaxis="y1"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Value_2"], name="Series 2", yaxis="y2"))
    fig.update_layout(
        title="Dual Axis Plot",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Series 1", side="left"),
        yaxis2=dict(title="Series 2", overlaying="y", side="right"),
    )
    fig.show()


    # --- code cell ---

    # Time series with range slider for interactive zooming
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Value"], mode="lines", name="Value"))
    fig.update_layout(
        title="Time Series with Range Slider",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    fig.show()


if __name__ == "__main__":
    main()
