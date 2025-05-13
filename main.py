from pydantic import BaseModel
from typing import Dict
import numpy as np
import pandas as pd
from fastapi import FastAPI
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Clasa care descrie datele pentru fiecare categorie de cheltuieli
class ExpenseByCategory(BaseModel):
    billsCategory: float = 0.0
    entertainmentCategory: float = 0.0
    groceryCategory: float = 0.0
    otherCategory: float = 0.0
    restaurantCategory: float = 0.0
    transportCategory: float = 0.0

# Clasa care descrie datele financiare pentru fiecare lună
class MonthlyData(BaseModel):
    income: float
    expenseTarget: float
    expenses: float
    expenseByCategory: ExpenseByCategory

# Clasa care reprezintă istoricul cheltuielilor
class HistoryInput(BaseModel):
    history: Dict[str, MonthlyData]  # Aici folosim un dicționar cu chei de tip string (ex: "2025_01")
    months_to_predict: int = 2  # numărul de luni pentru care se vor face predicții

# Funcție de utilitate pentru a verifica valorile NaN și a le înlocui
def verifyNanValues(itemApi):
    values = []
    for item in itemApi.values[0]:
        values.append(0 if np.isnan(item) else item)
    return values

# Funcție pentru a prezice serii temporale cu ARIMA
def forecast_series(series, months):
    if len(series) < 3:
        return [series[-1]] * months  # Fallback pentru date puține

    try:
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months)
        return forecast.tolist()
    except Exception as e:
        return [series[-1]] * months  # Fallback pe ultimul element

# Endpoint-ul principal pentru predicții multiple cu ARIMA
@app.post("/predict-multi-arima")
def predict_multi_arima(data: HistoryInput):
    # Sortăm istoricul pe lună
    history_sorted = dict(sorted(data.history.items()))

    # Creăm DataFrame-ul pe baza istoricului
    rows = []
    for month, entry in history_sorted.items():
        row = {
            "expenses": entry.expenses,
            **entry.expenseByCategory.dict()
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Pregătim predicțiile pentru fiecare categorie de cheltuieli și pentru total
    predictions = []
    for month_offset in range(1, data.months_to_predict + 1):
        result = {
            "month_offset": month_offset,
            "total_expense": 0.0,
            "categories": {}
        }

        total_series = df["expenses"].tolist()
        forecast_total = forecast_series(total_series, data.months_to_predict)
        result["total_expense"] = round(forecast_total[month_offset - 1], 2)

        for cat in ["billsCategory", "entertainmentCategory", "groceryCategory", "otherCategory", "restaurantCategory",
                    "transportCategory"]:
            cat_series = df[cat].tolist()
            forecast_cat = forecast_series(cat_series, data.months_to_predict)
            result["categories"][cat] = round(forecast_cat[month_offset - 1], 2)

        predictions.append(result)

    return {
        "months_predicted": data.months_to_predict,
        "predictions": predictions
    }
