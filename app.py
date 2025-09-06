from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from database import SessionLocal
from models import TrainingData, ModelStore, ModelEvaluation, TestingData, User
from datetime import datetime
from pytz import timezone
import numpy as np
from functools import wraps
from werkzeug.security import check_password_hash

app = Flask(__name__)

# Secret key wajib untuk session
app.secret_key = "supersecretkey"

# Aktifkan CORS + credentials
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()

    if user and check_password_hash(user.password_hash, password):
        session["user_id"] = user.id
        return jsonify({"message": "Login berhasil", "username": user.username}), 200
    else:
        return jsonify({"error": "Username atau password salah"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()  # hapus session user
    return jsonify({"message": "Logout berhasil"}), 200

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Untuk handle Train Data
@app.route("/train", methods=["POST"])
@login_required
def train_model():
    """Train model using all training data in DB (replace ModelStore & ModelEvaluation)"""
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    try:
        # Validasi file Excel
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({"error": "File harus berformat Excel (.xlsx atau .xls)"}), 400

        df = pd.read_excel(file)

        # Validasi kolom
        required_columns = ["pengunjung", "tayangan", "pesanan", "terjual"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"File harus memiliki kolom: {required_columns}"}), 400

        # Validasi nilai kosong / negatif
        if df[required_columns].isnull().values.any():
            return jsonify({"error": "Data tidak boleh mengandung nilai kosong"}), 400
        if (df[required_columns] < 0).any().any():
            return jsonify({"error": "Data tidak boleh mengandung nilai negatif"}), 400

        db = SessionLocal()
        try:
            # Simpan data latih baru
            for _, row in df.iterrows():
                training_row = TrainingData(
                    pengunjung=int(row["pengunjung"]),
                    tayangan=int(row["tayangan"]),
                    pesanan=int(row["pesanan"]),
                    terjual=int(row["terjual"]),
                    created_at=datetime.utcnow()
                )
                db.add(training_row)
            db.commit()

            # Ambil SEMUA data training dari DB
            all_training = db.query(TrainingData).all()
            if not all_training:
                return jsonify({"error": "Tidak ada data latih di database"}), 400

            df_all = pd.DataFrame([{
                "pengunjung": t.pengunjung,
                "tayangan": t.tayangan,
                "pesanan": t.pesanan,
                "terjual": t.terjual
            } for t in all_training])

            # Latih model dengan seluruh data
            X = df_all[["pengunjung", "tayangan", "pesanan"]]
            y = df_all["terjual"]

            model = LinearRegression()
            model.fit(X, y)

            intercept = float(model.intercept_)
            b1, b2, b3 = map(float, model.coef_)

            # Update ModelStore
            latest_model = db.query(ModelStore).order_by(ModelStore.id.desc()).first()
            if latest_model:
                latest_model.intercept = intercept
                latest_model.b1 = b1
                latest_model.b2 = b2
                latest_model.b3 = b3
                latest_model.created_at = datetime.utcnow()
            else:
                latest_model = ModelStore(
                    intercept=intercept,
                    b1=b1,
                    b2=b2,
                    b3=b3,
                    created_at=datetime.utcnow()
                )
                db.add(latest_model)
                db.commit()

            # Update evaluasi jika ada data testing
            testing_data = db.query(TestingData).all()
            if testing_data:
                df_test = pd.DataFrame([{
                    "pengunjung": d.pengunjung,
                    "tayangan": d.tayangan,
                    "pesanan": d.pesanan,
                    "terjual": d.terjual
                } for d in testing_data])

                X_test = df_test[["pengunjung", "tayangan", "pesanan"]]
                y_test = df_test["terjual"]

                prediksi = (
                    intercept
                    + b1 * X_test["pengunjung"]
                    + b2 * X_test["tayangan"]
                    + b3 * X_test["pesanan"]
                ).clip(lower=0)

                r2 = r2_score(y_test, prediksi)
                mae = mean_absolute_error(y_test, prediksi)
                mape = mean_absolute_percentage_error(y_test, prediksi)

                evaluation = db.query(ModelEvaluation).filter_by(model_id=latest_model.id).first()
                if evaluation:
                    evaluation.r2_score = r2
                    evaluation.mae = mae
                    evaluation.mape = mape
                    evaluation.created_at = datetime.utcnow()
                else:
                    evaluation = ModelEvaluation(
                        model_id=latest_model.id,
                        r2_score=r2,
                        mae=mae,
                        mape=mape,
                        created_at=datetime.utcnow()
                    )
                    db.add(evaluation)

            db.commit()

            return jsonify({
                "message": "Model berhasil dilatih ulang berdasarkan semua data training",
                "model": {
                    "intercept": intercept,
                    "b1": b1,
                    "b2": b2,
                    "b3": b3
                }
            })

        finally:
            db.close()

    except Exception as e:
        return jsonify({"error": f"Gagal melatih model: {str(e)}"}), 500


# untuk menampilkan semua data training pada tabel
@app.route("/training-data", methods=["GET"])
def get_training_data():
    try:
        db = SessionLocal()
        all_data = db.query(TrainingData).all()
        db.close()

        result = [
            {
                "id": data.id,
                "pengunjung": data.pengunjung,
                "tayangan": data.tayangan,
                "pesanan": data.pesanan,
                "terjual": data.terjual
            }
            for data in all_data
        ]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# untuk prediksi penjualan
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
    

# untuk menampilkan Koefisien regresi model    
@app.route("/model-info", methods=["GET"])
def model_info():
    db = SessionLocal()
    data_count = db.query(TrainingData).count()

    model_data = db.query(ModelStore).order_by(ModelStore.id.desc()).first()
    if not model_data:
        return jsonify({"error": "Model belum tersedia"}), 400
    
     # Konversi waktu UTC ke Asia/Jakarta (WIB)
    utc_time = model_data.created_at
    jakarta_tz = timezone("Asia/Jakarta")
    wib_time = utc_time.astimezone(jakarta_tz)

    return jsonify({
        "jumlah_data": data_count,
        "intercept": model_data.intercept,
        "b1": model_data.b1,
        "b2": model_data.b2,
        "b3": model_data.b3,
        "updated_at": wib_time.strftime("%d-%m-%Y %H:%M:%S")
    })
    

@app.route("/latest-evaluation", methods=["GET"])
def get_latest_evaluation():
    """Get latest model evaluation metrics"""
    try:
        db = SessionLocal()
        try:
            # Ambil model terbaru
            model_data = db.query(ModelStore).order_by(ModelStore.created_at.desc()).first()
            
            if not model_data:
                return jsonify({"error": "Model belum tersedia"}), 400
            
            # Ambil evaluasi terbaru untuk model ini
            evaluation = db.query(ModelEvaluation).filter_by(model_id=model_data.id).first()
            
            if not evaluation:
                return jsonify({"error": "Evaluasi belum dilakukan"}), 400
            
            # Hitung statistik data testing
            testing_count = db.query(TestingData).count()
            
            return jsonify({
                "model_id": model_data.id,
                "evaluasi": {
                    "r2_score": round(evaluation.r2_score, 4),
                    "mae": round(evaluation.mae, 4),
                    "mape": round(evaluation.mape, 4)
                },
                "data_info": {
                    "testing_samples": testing_count,
                    "evaluation_date": evaluation.created_at.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            
        finally:
            db.close()
            
    except Exception as e:
        # logger.error(f"Error getting latest evaluation: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

# untuk logic evaluasi model
@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate model with testing data (tanpa update ModelStore)"""
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    try:
        # Validasi file Excel
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({"error": "File harus berformat Excel (.xlsx atau .xls)"}), 400

        df = pd.read_excel(file)

        # Validasi kolom
        required_columns = ["pengunjung", "tayangan", "pesanan", "terjual"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"File harus memiliki kolom: {required_columns}"}), 400

        # Validasi nilai negatif
        if (df[required_columns] < 0).any().any():
            return jsonify({"error": "Data tidak boleh mengandung nilai negatif"}), 400

        # Validasi tidak kosong
        if len(df) == 0:
            return jsonify({"error": "File tidak boleh kosong"}), 400

        X = df[["pengunjung", "tayangan", "pesanan"]]
        y = df["terjual"]

        db = SessionLocal()
        try:
            # =========================
            # 1. Ambil model terakhir dari ModelStore
            # =========================
            model_data = db.query(ModelStore).order_by(ModelStore.id.desc()).first()
            if not model_data:
                return jsonify({"error": "Belum ada model yang dilatih dari data training"}), 400

            # Buat ulang model LinearRegression dari parameter yang tersimpan
            model = LinearRegression()
            model.intercept_ = model_data.intercept
            model.coef_ = np.array([model_data.b1, model_data.b2, model_data.b3])

            # =========================
            # 2. Prediksi data testing
            # =========================
            y_pred = model.predict(X)
            y_pred_clipped = np.clip(y_pred, 0, None)

            # =========================
            # 3. Hitung metrik evaluasi
            # =========================
            r2 = r2_score(y, y_pred_clipped)
            mae = mean_absolute_error(y, y_pred_clipped)
            mape = mean_absolute_percentage_error(y, y_pred_clipped)

            # =========================
            # 4. Hapus & simpan TestingData baru
            # =========================
            db.query(TestingData).delete()
            for i, row in df.iterrows():
                db.add(TestingData(
                    pengunjung=int(row["pengunjung"]),
                    tayangan=int(row["tayangan"]),
                    pesanan=int(row["pesanan"]),
                    terjual=int(row["terjual"]),
                    predicted=round(float(y_pred_clipped[i]), 2)
                ))

            # =========================
            # 5. Update / Simpan Evaluasi Model
            # =========================
            existing_eval = db.query(ModelEvaluation).filter_by(model_id=model_data.id).first()
            if existing_eval:
                existing_eval.r2_score = float(r2)
                existing_eval.mae = float(mae)
                existing_eval.mape = float(mape)
                existing_eval.created_at = datetime.utcnow()
            else:
                evaluation = ModelEvaluation(
                    model_id=model_data.id,
                    r2_score=float(r2),
                    mae=float(mae),
                    mape=float(mape)
                )
                db.add(evaluation)

            db.commit()

            return jsonify({
                "message": "Evaluasi berhasil (model tidak diubah, hanya dievaluasi)",
                "jumlah_data_test": len(df),
                "evaluasi": {
                    "r2_score": round(r2, 4),
                    "mae": round(mae, 4),
                    "mape": round(mape, 4)
                },
                "summary": {
                    "best_prediction": float(y_pred_clipped.max()),
                    "worst_prediction": float(y_pred_clipped.min()),
                    "avg_prediction": float(y_pred_clipped.mean()),
                    "total_actual": int(y.sum()),
                    "total_predicted": int(y_pred_clipped.sum())
                }
            })

        finally:
            db.close()

    except pd.errors.EmptyDataError:
        return jsonify({"error": "File Excel kosong atau tidak valid"}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Format file Excel tidak valid"}), 400
    except Exception as e:
        return jsonify({"error": f"Error dalam evaluasi: {str(e)}"}), 500


@app.route("/testing-data", methods=["GET"])
def get_testing_data():
    """Get all testing data with predictions"""
    try:
        db = SessionLocal()
        try:
            # Ambil data testing yang sudah ada prediksinya
            all_data = db.query(TestingData).order_by(TestingData.created_at.desc()).all()

            if not all_data:
                return jsonify([])

            result = []
            for data in all_data:
                # Hitung error jika ada prediksi
                error = None
                if data.predicted is not None:
                    error = abs(data.terjual - data.predicted)
                
                result.append({
                    "id": data.id,
                    "pengunjung": data.pengunjung,
                    "tayangan": data.tayangan,
                    "pesanan": data.pesanan,
                    "terjual": data.terjual,
                    "prediksi": data.predicted,
                    "error": round(error, 2) if error is not None else None,
                    "created_at": data.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            return jsonify(result)
            
        finally:
            db.close()
            
    except Exception as e:
        # logger.error(f"Error getting testing data: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/train/delete-all", methods=["DELETE"])
def delete_all_training_data():
    try:
        db = SessionLocal()

        # Hapus semua data dari tabel training
        deleted_rows = db.query(TrainingData).delete()
        db.commit()

        db.close()
        return jsonify({"message": f"{deleted_rows} data latih berhasil dihapus."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)