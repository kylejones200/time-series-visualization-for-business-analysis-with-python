"""Time series visualization using Polars and DuckDB."""

import duckdb
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_time_series(
    df: pl.DataFrame, value_col: str, date_col: str, title: str, output_path: Path
):
    dates  = df[date_col].to_list()
    values = df[value_col].to_list()

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, values, color="#4A90A4", linewidth=1.2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def plot_multiple_series(
    df: pl.DataFrame, date_col: str, columns: List[str], title: str, output_path: Path
):
    colors = ["#4A90A4", "#D4A574", "#8B6F9E", "#A8C5A0", "#E8A87C"]
    dates = df[date_col].to_list()

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, col in enumerate(columns):
            ax.plot(dates, df[col].to_list(), label=col,
                    color=colors[i % len(colors)], linewidth=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def decompose_trend(
    df: pl.DataFrame, date_col: str, value_col: str, period: int
) -> pl.DataFrame:
    """Centered rolling mean (trend) + detrended series via DuckDB window functions."""
    half = period // 2
    return duckdb.sql(f"""
        SELECT
            "{date_col}",
            "{value_col}",
            AVG("{value_col}") OVER (
                ORDER BY "{date_col}"
                ROWS BETWEEN {half} PRECEDING AND {half} FOLLOWING
            ) AS trend,
            "{value_col}" - AVG("{value_col}") OVER (
                ORDER BY "{date_col}"
                ROWS BETWEEN {half} PRECEDING AND {half} FOLLOWING
            ) AS seasonal
        FROM df
        ORDER BY "{date_col}"
    """).pl()


def plot_seasonal_decomposition(
    df: pl.DataFrame, value_col: str, date_col: str,
    period: int, title: str, output_path: Path,
):
    decomposed = decompose_trend(df, date_col, value_col, period)
    dates = decomposed[date_col].to_list()

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axes[0].plot(dates, decomposed[value_col].to_list(), color="#4A90A4", linewidth=1.2)
        axes[0].set_ylabel("Original")

        axes[1].plot(dates, decomposed["trend"].to_list(),    color="#D4A574", linewidth=1.2)
        axes[1].set_ylabel("Trend")

        axes[2].plot(dates, decomposed["seasonal"].to_list(), color="#8B6F9E", linewidth=1.2)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Detrended")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close()
