"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from matplotlib.animation import FuncAnimation
from pycaret.time_series import *


class VolatilityVisualizer:
    def __init__(self):
        np.random.seed(42)
        self.n = 1000
        self.returns = self.generate_garch_process()
        self.model = arch_model(self.returns, vol="GARCH", p=1, q=1)
        self.result = self.model.fit(disp="off")
        self.conditional_vol = np.sqrt(self.result.conditional_volatility)

    def generate_garch_process(self):
        omega = 0.1
        alpha = 0.1
        beta = 0.8
        returns = np.zeros(self.n)
        sigma2 = np.zeros(self.n)
        for t in range(1, self.n):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)
        return returns

    def create_animation(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        def animate(frame):
            window = 200
            start_idx = frame % (self.n - window)
            end_idx = start_idx + window
            for ax in [ax1, ax2, ax3]:
                ax.clear()
            ax1.plot(
                range(start_idx, end_idx),
                self.returns[start_idx:end_idx],
                label="Returns",
                alpha=0.7,
            )
            ax1.plot(
                range(start_idx, end_idx),
                self.conditional_vol[start_idx:end_idx],
                "r--",
                label="Conditional Volatility",
                alpha=0.7,
            )
            ax1.fill_between(
                range(start_idx, end_idx),
                -self.conditional_vol[start_idx:end_idx],
                self.conditional_vol[start_idx:end_idx],
                color="r",
                alpha=0.1,
            )
            ax1.set_title("Returns and Conditional Volatility")
            ax1.legend(loc="upper right")
            ax1.grid(False)
            rolling_vol = pd.Series(self.returns[start_idx:end_idx]).rolling(20).std()
            ax2.plot(
                range(start_idx, end_idx),
                rolling_vol,
                label="20-day Rolling Volatility",
            )
            ax2.set_title("Rolling Volatility")
            ax2.legend(loc="upper right")
            ax2.grid(False)
            ax3.hist(
                self.returns[start_idx:end_idx],
                bins=30,
                density=True,
                alpha=0.7,
                label="Returns Distribution",
            )
            ax3.set_title("Returns Distribution")
            ax3.legend(loc="upper right")
            ax3.grid(False)
            plt.tight_layout()

        anim = FuncAnimation(
            fig, animate, frames=self.n - 200, interval=100, repeat=True
        )
        return anim
