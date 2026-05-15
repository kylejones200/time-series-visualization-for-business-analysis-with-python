#!/usr/bin/env python3
"""Time series visualization — Polars + DuckDB rewrite."""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from core import plot_seasonal_decomposition, plot_time_series

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="TS visualization — Polars + DuckDB")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    if args.data_path and args.data_path.exists():
        df = pl.read_csv(args.data_path, try_parse_dates=True)
    elif config["data"]["generate_synthetic"]:
        rng = np.random.default_rng(config["data"]["seed"])
        n = config["data"]["n_periods"]
        start = date(2023, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n)]
        values = (np.sin(np.arange(n) / 30) * 100 + 500 + rng.normal(0, 20, n)).tolist()
        df = pl.DataFrame({"date": dates, "value": values})
    else:
        raise ValueError("No data source specified")

    if config["visualization"]["time_series"]:
        plot_time_series(
            df, "value", "date", "Business Time Series", output_dir / "time_series.png"
        )
        logging.info("time_series.png saved")

    if config["visualization"]["seasonal_decomposition"]:
        plot_seasonal_decomposition(
            df,
            "value",
            "date",
            period=config["visualization"]["seasonal_period"],
            title="Seasonal Decomposition",
            output_path=output_dir / "seasonal_decomposition.png",
        )
        logging.info("seasonal_decomposition.png saved")

    logging.info(f"Visualization complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
