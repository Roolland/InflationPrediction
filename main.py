from pydantic import BaseModel
from typing import Dict
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from statsmodels.tsa.arima.model import ARIMA
import logging
import wbdata
import wbgapi as wb
import datetime
from fastapi.responses import JSONResponse
# Configurare basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ExpenseByCategory(BaseModel):
    billsCategory: float = 0.0
    entertainmentCategory: float = 0.0
    groceryCategory: float = 0.0
    otherCategory: float = 0.0
    restaurantCategory: float = 0.0
    transportCategory: float = 0.0

class MonthlyData(BaseModel):
    income: float
    expenseTarget: float
    expenses: float
    expenseByCategory: ExpenseByCategory

class HistoryInput(BaseModel):
    history: Dict[str, MonthlyData]
    months_to_predict: int = 1

def forecast_series(series, months):
    if len(series) < 3:
        logger.warning("Serie prea scurtă pentru ARIMA, folosim fallback.")
        return [series[-1]] * months

    try:
        model = ARIMA(series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months)
        return forecast.tolist()
    except Exception as e:
        logger.error(f"Eroare la forecast_series: {e}")
        return [series[-1]] * months

@app.post("/predict-multi-arima")
async def predict_multi_arima(data: HistoryInput, request: Request):
    logger.info("🟢 Cerere nouă primită pentru /predict-multi-arima")
    logger.info(f"📦 Body primit: {await request.body()}")
    logger.info(f"📊 Months to predict: {data.months_to_predict}")
    logger.info(f"📅 Chei în istoric: {list(data.history.keys())}")

    try:
        history_sorted = dict(sorted(data.history.items()))
        logger.info("✅ Istoric sortat cu succes.")

        rows = []
        for month, entry in history_sorted.items():
            logger.info(f"➡️ Procesăm luna: {month}")
            row = {
                "expenses": entry.expenses,
                **entry.expenseByCategory.model_dump()
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"📈 DataFrame construit:\n{df}")

        predictions = []
        for month_offset in range(1, data.months_to_predict + 1):
            logger.info(f"🔮 Generăm predicție pentru luna +{month_offset}")
            result = {
                "month_offset": month_offset,
                "total_expense": 0.0,
                "categories": {}
            }

            total_series = df["expenses"].tolist()
            forecast_total = forecast_series(total_series, data.months_to_predict)
            result["total_expense"] = round(forecast_total[month_offset - 1], 2)

            for cat in ["billsCategory", "entertainmentCategory", "groceryCategory", "otherCategory", "restaurantCategory", "transportCategory"]:
                cat_series = df[cat].tolist()
                forecast_cat = forecast_series(cat_series, data.months_to_predict)
                result["categories"][cat] = round(forecast_cat[month_offset - 1], 2)

            logger.info(f"✅ Predicție luna +{month_offset}: {result}")
            predictions.append(result)

        logger.info("🟢 Toate predicțiile au fost generate cu succes.")
        return {
            "months_predicted": data.months_to_predict,
            "predictions": predictions
        }

    except Exception as e:
        logger.error(f"❌ Eroare în endpoint /predict-multi-arima: {e}")
        return {"error": str(e)}

@app.get("/inflation-average")
def get_romania_inflation_average():
    try:
        logger.info("📥 Începe procesarea /inflation-average")

        # Inițializăm conexiunea la World Bank API
        logger.info("🌍 Inițializăm sursa World Bank")
        wb.source()

        # Iterăm anii 2014–2023
        values = []
        for year in range(2014, 2024):
            logger.info(f"📅 Preluăm inflația pentru anul {year}")
            try:
                val = wb.data.get('FP.CPI.TOTL.ZG', economy='RO', time=year)

                if val and isinstance(val, list) and val[0]['value'] is not None:
                    logger.info(f"✅ Inflație găsită pentru {year}: {val[0]['value']}%")
                    values.append(val[0]['value'])
                else:
                    logger.warning(f"⚠️ Inflație lipsă pentru {year}")

            except Exception as e:
                logger.warning(f"❌ Eroare la preluarea inflației pentru {year}: {e}")

        if not values:
            logger.error("❌ Nu s-au găsit date valide pentru niciun an.")
            return JSONResponse(status_code=404, content={"error": "Fără date valide pentru inflație."})

        # Calculăm media
        average = round(sum(values) / len(values), 2)
        logger.info(f"📊 Inflație medie calculată pe {len(values)} ani: {average}%")

        return average

    except Exception as e:
        logger.error(f"💥 Eroare critică în /inflation-average: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


