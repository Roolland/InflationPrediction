from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import wbgapi as wb
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()

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