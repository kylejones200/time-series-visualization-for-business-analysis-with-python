"""Auto-split from legacy monolithic script."""

import pandas as pd
from orbit.diagnostics.metrics import smape
from orbit.diagnostics.plot import plot_predicted_data
from orbit.models import DLT, KTR
from orbit.utils.dataset import load_iclaims
from pycaret.time_series import *


def load_and_prepare_data() -> None:
    data = load_iclaims()
    data["week"] = pd.to_datetime(data["week"])
    data = data.rename(columns={"week": "date", "claims": "value"})
    model = DLT(response_col="value", date_col="date", seasonality=52)
    model.fit(df=data)
    predictions = model.predict(df=data)
    plot_predicted_data(
        data, predictions, date_col="date", actual_col="value", pred_col="prediction"
    )
    true_values = data["value"]
    predicted_values = predictions["prediction"]
    print("SMAPE:", smape(true_values, predicted_values))
    model_damped = DLT(
        response_col="value", date_col="date", seasonality=52, damped=True
    )
    model_damped.fit(df=data)
    model_ktr = KTR(
        response_col="value", date_col="date", seasonality=52, level_knot_prior=0.5
    )
    model_ktr.fit(df=data)
    data["recession"] = [1 if x % 12 < 3 else 0 for x in range(len(data))]
    model_multivariate = DLT(
        response_col="value",
        date_col="date",
        seasonality=52,
        regressor_col=["recession"],
    )
    model_multivariate.fit(df=data)
