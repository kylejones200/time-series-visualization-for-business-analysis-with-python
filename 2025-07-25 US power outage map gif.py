"""Generated from Jupyter notebook: 2025-07-25 US power outage map gif

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import os

import geopandas as gpd
import imageio
import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("/content/eaglei_outages_2024.csv", parse_dates=["run_start_time"])
    df["fips_code"] = df["fips_code"].astype(str).str.zfill(5)
    df["outage_prop"] = df["customers_out"] / df["total_customers"]
    df["hour"] = df["run_start_time"].dt.floor("H")
    gdf = gpd.read_file("/content/cb_2021_us_county_20m.shx")
    gdf = gdf[~gdf["STATEFP"].isin(["02", "15", "72"])]
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    vmin = 0
    vmax = df["outage_prop"].max()
    os.makedirs("frames", exist_ok=True)
    all_hours = sorted(df["hour"].unique())
    for ts in all_hours:
        df_hour = df[df["hour"] == ts]
        merged = gdf.merge(df_hour, left_on="GEOID", right_on="fips_code", how="left")
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        merged.plot(
            column="outage_prop",
            ax=ax,
            cmap="magma_r",
            linewidth=0,
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            missing_kwds={"color": "lightgrey"},
            legend_kwds={"label": "Proportion Without Power", "shrink": 0.5},
        )
        ax.set_title(f"Outage Proportion — {ts:%Y-%m-%d %H:%M}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        fname = f"frames/{ts:%Y%m%d%H%M}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
    with imageio.get_writer("outages_by_county.gif", mode="I", duration=0.5) as writer:
        for ts in all_hours:
            fname = f"frames/{ts:%Y%m%d%H%M}.png"
            image = imageio.imread(fname)
            writer.append_data(image)
    print("GIF saved as outages_by_county.gif")
    import os

    import geopandas as gpd
    import imageio
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("/content/eaglei_outages_2024.csv", parse_dates=["run_start_time"])
    df["fips_code"] = df["fips_code"].astype(str).str.zfill(5)
    df["day"] = df["run_start_time"].dt.floor("D")
    df_daily = df.groupby(["fips_code", "day", "state", "county"], as_index=False).agg(
        {"customers_out": "sum", "total_customers": "first"}
    )
    df_daily["outage_prop"] = df_daily["customers_out"] / df_daily["total_customers"]
    gdf = gpd.read_file("cb_2021_us_county_20m.shp")
    gdf = gdf[~gdf["STATEFP"].isin(["02", "15", "72"])]
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    vmin = 0
    vmax = df_daily["outage_prop"].max()
    os.makedirs("frames", exist_ok=True)
    all_days = sorted(df_daily["day"].unique())
    for ts in all_days:
        df_day = df_daily[df_daily["day"] == ts]
        merged = gdf.merge(df_day, left_on="GEOID", right_on="fips_code", how="left")
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        merged.plot(
            column="outage_prop",
            ax=ax,
            cmap="magma_r",
            linewidth=0,
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            missing_kwds={"color": "lightgrey"},
            legend_kwds={"label": "Proportion Without Power", "shrink": 0.5},
        )
        ax.set_title(f"Average Outage Proportion — {ts:%Y-%m-%d}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        fname = f"frames/{ts:%Y%m%d}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
    with imageio.get_writer("outages_by_county.gif", mode="I", duration=0.5) as writer:
        for ts in all_days:
            fname = f"frames/{ts:%Y%m%d}.png"
            image = imageio.imread(fname)
            writer.append_data(image)
    print("GIF saved as outages_by_county.gif")
    df_daily.to_parquet("daily_outages.parquet", index=False)
    import os

    import geopandas as gpd
    import imageio
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv("eaglei_outages_2024.csv", parse_dates=["run_start_time"])
    df["fips_code"] = df["fips_code"].astype(str).str.zfill(5)
    df["day"] = df["run_start_time"].dt.floor("D")
    all_days = df["day"].unique()
    all_fips = df["fips_code"].unique()
    county_template = pd.DataFrame(
        [(fips, day) for fips in all_fips for day in all_days],
        columns=["fips_code", "day"],
    )
    agg = df.groupby(["fips_code", "day"], as_index=False).agg(
        {"customers_out": "sum", "total_customers": "first"}
    )
    df_daily_full = county_template.merge(agg, on=["fips_code", "day"], how="left")
    df_daily_full["customers_out"] = df_daily_full["customers_out"].fillna(0)
    df_daily_full["total_customers"] = df_daily_full["total_customers"].fillna(
        method="ffill"
    )
    df_daily_full["outage_prop"] = (
        df_daily_full["customers_out"] / df_daily_full["total_customers"]
    )
    df_daily_full["outage_prop"] = df_daily_full["outage_prop"].fillna(0)
    df_daily_full.to_parquet("daily_outages.parquet", index=False)
    gdf = gpd.read_file("cb_2021_us_county_20m.shp")
    gdf = gdf[~gdf["STATEFP"].isin(["02", "15", "72"])]
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    vmin = 0
    vmax = df_daily_full["outage_prop"].max()
    os.makedirs("frames", exist_ok=True)
    all_days_sorted = sorted(df_daily_full["day"].unique())
    for ts in all_days_sorted:
        df_day = df_daily_full[df_daily_full["day"] == ts]
        merged = gdf.merge(df_day, left_on="GEOID", right_on="fips_code", how="left")
        merged["outage_prop"] = merged["outage_prop"].fillna(0)
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        merged.plot(
            column="outage_prop",
            ax=ax,
            cmap="magma_r",
            linewidth=0,
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={"label": "Proportion Without Power", "shrink": 0.5},
        )
        ax.set_title(f"Daily Outage Proportion — {ts:%Y-%m-%d}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        fname = f"frames/{ts:%Y%m%d}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
    with imageio.get_writer("outages_by_county.gif", mode="I", duration=0.5) as writer:
        for ts in all_days_sorted:
            fname = f"frames/{ts:%Y%m%d}.png"
            image = imageio.imread(fname)
            writer.append_data(image)
    print("GIF saved as outages_by_county.gif")
    import os

    import geopandas as gpd
    import imageio
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_parquet("daily_outages.parquet")
    df["fips_code"] = df["fips_code"].astype(str).str.zfill(5)
    df["day"] = pd.to_datetime(df["day"])
    any_outages = (
        df.groupby("fips_code")["customers_out"]
        .sum()
        .reset_index()
        .query("customers_out > 0")
    )
    active_fips = set(any_outages["fips_code"])
    df = df[df["fips_code"].isin(active_fips)]
    all_days = df["day"].unique()
    template = pd.DataFrame(
        [(f, d) for f in active_fips for d in all_days], columns=["fips_code", "day"]
    )
    agg = df.groupby(["fips_code", "day"], as_index=False).agg({"customers_out": "sum"})
    df_daily = template.merge(agg, on=["fips_code", "day"], how="left")
    df_daily["customers_out"] = df_daily["customers_out"].fillna(0)
    gdf = gpd.read_file("cb_2021_us_county_20m.shp")
    gdf = gdf[~gdf["STATEFP"].isin(["02", "15", "72"])]
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    gdf = gdf[gdf["GEOID"].isin(active_fips)]
    vmin = 0
    vmax = 5000
    os.makedirs("frames", exist_ok=True)
    all_days_sorted = sorted(df_daily["day"].unique())
    for ts in all_days_sorted:
        df_day = df_daily[df_daily["day"] == ts]
        merged = gdf.merge(df_day, left_on="GEOID", right_on="fips_code", how="left")
        merged["customers_out"] = merged["customers_out"].fillna(0)
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        merged.plot(
            column="customers_out",
            ax=ax,
            cmap="inferno",
            linewidth=0,
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            legend_kwds={"label": "People Without Power", "shrink": 0.5},
        )
        ax.set_title(f"Daily Outages — {ts:%Y-%m-%d}", fontsize=16)
        ax.axis("off")
        plt.tight_layout()
        fname = f"frames/{ts:%Y%m%d}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
    with imageio.get_writer(
        "outages_absolute_capped.gif", mode="I", duration=0.5
    ) as writer:
        for ts in all_days_sorted:
            fname = f"frames/{ts:%Y%m%d}.png"
            image = imageio.imread(fname)
            writer.append_data(image)
    print("GIF saved as outages_absolute_capped.gif")


def main() -> None:
    main()


if __name__ == "__main__":
    main()
