import sys
sys.path.insert(0, 'd:/project-TA/backend')

from database import SessionLocal
from models import TrainingData
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

def forecast_variable(data_series, periods=1, order=(1,1,1)):
    """Test ARIMA forecast"""
    try:
        if len(data_series) < 3:
            return float(data_series.mean())
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(data_series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            return max(0, float(forecast.iloc[0]))
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return float(data_series.mean())

# Test dengan product_id 1
db = SessionLocal()
training_data = (
    db.query(TrainingData)
    .filter(TrainingData.product_id == 1)
    .order_by(TrainingData.tahun.asc(), TrainingData.bulan.asc())
    .all()
)

print(f"Found {len(training_data)} training records")

if training_data:
    df = pd.DataFrame([{
        "pengunjung": t.pengunjung,
        "tayangan": t.tayangan,
        "pesanan": t.pesanan
    } for t in training_data])
    
    print("\nData:")
    print(df)
    
    print("\nForecasting...")
    forecasted_pengunjung = forecast_variable(df["pengunjung"])
    forecasted_tayangan = forecast_variable(df["tayangan"])
    forecasted_pesanan = forecast_variable(df["pesanan"])
    
    print(f"\nResults:")
    print(f"Pengunjung: {forecasted_pengunjung}")
    print(f"Tayangan: {forecasted_tayangan}")
    print(f"Pesanan: {forecasted_pesanan}")
else:
    print("No training data found")

db.close()
