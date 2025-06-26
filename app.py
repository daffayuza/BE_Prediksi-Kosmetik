from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from database import SessionLocal
from models import TrainingData, ModelStore

app = Flask(__name__)
CORS(app)

@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    try:
        df = pd.read_excel(file)

        X = df[["pengunjung", "tayangan", "pesanan"]]
        y = df["terjual"]

        # Simpan data ke DB
        db = SessionLocal()
        for _, row in df.iterrows():
            db.add(TrainingData(
                pengunjung=int(row["pengunjung"]),
                tayangan=int(row["tayangan"]),
                pesanan=int(row["pesanan"]),
                terjual=int(row["terjual"])
            ))
        db.commit()

        # Ambil semua data training yang ada di DB
        all_data = db.query(TrainingData).all()
        data_X = [[d.pengunjung, d.tayangan, d.pesanan] for d in all_data]
        data_y = [d.terjual for d in all_data]

        # Latih model regresi
        model = LinearRegression()
        model.fit(data_X, data_y)
        y_pred = model.predict(data_X)

        r2 = r2_score(data_y, y_pred)
        mae = mean_absolute_error(data_y, y_pred)
        mse = mean_squared_error(data_y, y_pred)

        # Simpan model terbaru ke DB (hapus lama)
        db.query(ModelStore).delete()
        db.add(ModelStore(
            intercept=model.intercept_,
            b1=model.coef_[0],
            b2=model.coef_[1],
            b3=model.coef_[2],
            r2_score=r2,
            mae=mae,
            mse=mse
        ))
        db.commit()
        db.close()

        return jsonify({
            "message": "Model berhasil dilatih",
            "jumlah_data_latih": len(data_y),
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "intercept": round(model.intercept_, 4),
            "b1": round(model.coef_[0], 4),
            "b2": round(model.coef_[1], 4),
            "b3": round(model.coef_[2], 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pengunjung = float(request.form["pengunjung"])
        tayangan = float(request.form["tayangan"])
        pesanan = float(request.form["pesanan"])

        db = SessionLocal()
        model_data = db.query(ModelStore).order_by(ModelStore.created_at.desc()).first()
        db.close()

        if not model_data:
            return jsonify({"error": "Model belum tersedia"}), 400

        prediksi = (
            model_data.intercept
            + model_data.b1 * pengunjung
            + model_data.b2 * tayangan
            + model_data.b3 * pesanan
        )

        return jsonify({"prediksi_terjual": round(prediksi)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)