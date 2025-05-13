from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import wbgapi as wb
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

# Permite apeluri de pe mobil sau alte surse (ex: Android app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verifyNanValues(itemApi):
    values = []
    for item in itemApi.values[0]:
        values.append(0 if np.isnan(item) else item)
    return values

@app.get("/predict")
def predict():
    flag = 'ROU'
    real_inflation = wb.data.DataFrame('FP.CPI.TOTL.ZG', flag, range(1980, 2023))
    undemployment_rate = wb.data.DataFrame('SL.UEM.TOTL.NE.ZS', flag, range(1980, 2023))
    broad_money = wb.data.DataFrame('FM.LBL.BMNY.GD.ZS', flag, range(1980, 2023))
    interest_rate = wb.data.DataFrame('FR.INR.LEND', flag, range(1980, 2023))

    valueItemInflation = verifyNanValues(real_inflation)
    valueItemInterestRate = verifyNanValues(interest_rate)
    valueItemUnemployment = verifyNanValues(undemployment_rate)
    valueItemBroadMoney = verifyNanValues(broad_money)

    data = pd.DataFrame({
        'Broad_Money_Percent_GDP': valueItemBroadMoney,
        'Interest_Rate': valueItemInterestRate,
        'External_Shocks': valueItemUnemployment,
        'Inflation': valueItemInflation
    })

    X = data[['Broad_Money_Percent_GDP', 'Interest_Rate', 'External_Shocks']]
    y = data['Inflation']

    model = ARIMA(y, exog=X, order=(1, 0, 1))
    model_fit = model.fit()

    predictions = model_fit.predict(exog=X)
    years = list(range(1980, 2023))

    return {
        "years": years,
        "real_inflation": valueItemInflation,
        "predicted_inflation": predictions.tolist()
    }


class ExpenseByCategory(BaseModel):
    billsCategory: float
    entertainmentCategory: float
    groceryCategory: float
    otherCategory: float
    restaurantCategory: float
    transportCategory: float


class MonthlyData(BaseModel):
    income: float
    expenseTarget: float
    expenses: float
    expenseByCategory: ExpenseByCategory


class HistoryInput(BaseModel):
    history: Dict[str, MonthlyData]
    months_to_predict: int = 1


# ==== FUNCȚIE UTILITARĂ ====

def forecast_series(series, months):
    """ Rulează ARIMA simplu pe o serie 1D pentru a prezice următoarele luni """
    if len(series) < 3:
        # fallback pentru date puține
        return [series[-1]] * months

    try:
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months)
        return forecast.tolist()
    except Exception as e:
        return [series[-1]] * months  # fallback pe ultimul element


# ==== ENDPOINT ====

@app.post("/predict-multi-arima")
def predict_multi_arima(data: HistoryInput):
    history_sorted = dict(sorted(data.history.items()))

    # Construim dataframe
    rows = []
    for month, entry in history_sorted.items():
        row = {
            "expenses": entry.expenses,
            **entry.expenseByCategory.dict()
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Serii pentru fiecare categorie + total
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
