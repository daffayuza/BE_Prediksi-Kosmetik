from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from database import SessionLocal
from models import TrainingData, ModelStore, ModelEvaluation, TestingData, User, Product, PredictionResult
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from pytz import timezone
import numpy as np
from sqlalchemy import or_
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
import warnings

app = Flask(__name__)

# Secret key wajib untuk session
app.secret_key = "supersecretkey"

# Aktifkan CORS + credentials
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

@app.route("/register", methods=["POST"])
def register():
    db = SessionLocal()
    data = request.get_json()

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username dan password wajib diisi"}), 400

    # cek user sudah ada
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        return jsonify({"error": "Username sudah terdaftar"}), 400

    # hash password
    hashed_password = generate_password_hash(password)  

    new_user = User(username=username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()

    return jsonify({"message": "Registrasi berhasil, silakan login"})

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

# Helper function untuk ARIMA forecasting
def forecast_variable(data_series, periods=1, order=(2,1,0)):
    """
    Forecast time series menggunakan ARIMA
    
    Args:
        data_series: pandas Series dengan data historis
        periods: jumlah periode yang akan di-forecast
        order: (p,d,q) order untuk ARIMA
    
    Returns:
        forecasted value (non-negative)
    """
    try:
        # Minimal data check
        if len(data_series) < 3:
            return float(data_series.mean())
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = ARIMA(data_series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            return max(0, float(forecast.iloc[0]))  # ensure non-negative
    except Exception as e:
        # Fallback: gunakan mean jika ARIMA gagal
        return float(data_series.mean())

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Untuk handle Train Data
@app.route("/train/<int:product_id>", methods=["POST"])
def train_model(product_id):
    """Train model untuk produk tertentu, simpan model ke database, update evaluasi jika ada data testing."""
    
    # --- Validasi file ---
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    # Validasi format file Excel
    if not file.filename.endswith(('.xlsx', '.xls')):
        return jsonify({"error": "File harus berformat Excel (.xlsx atau .xls)"}), 400

    # PINDAHKAN PEMBACAAN FILE KE DALAM TRY-EXCEPT TERPISAH
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"File tidak dapat dibaca. Pastikan file adalah Excel yang valid: {str(e)}"}), 400

    # Validasi kolom
    required_columns = ["tahun", "bulan","pengunjung", "tayangan", "pesanan", "terjual"]
    if not all(col in df.columns for col in required_columns):
        return jsonify({"error": f"File harus memiliki kolom: tahun, bulan, pengunjung, tayangan, pesanan, terjual"}), 400

    # Validasi nilai
    if (df[required_columns] < 0).any().any():
        return jsonify({"error": "Data tidak boleh mengandung nilai negatif"}), 400
    
    if df[required_columns].isnull().any().any():
        return jsonify({"error": "Data tidak boleh mengandung nilai kosong"}), 400
    
    if not df["bulan"].between(1, 12).all():
        return jsonify({"error": "Nilai bulan harus antara 1 sampai 12"}), 400

    if len(df) == 0:
        return jsonify({"error": "File tidak boleh kosong"}), 400
    
    db = None

    try:
        db = SessionLocal()

        # ----------------------------------------------
        # 1. Hapus training lama untuk PRODUCT TERKAIT
        # ----------------------------------------------
        db.query(TrainingData).filter_by(product_id=product_id).delete()
        db.commit()

        # ----------------------------------------------
        # 2. Simpan training baru
        # ----------------------------------------------
        for _, row in df.iterrows():
            db.add(TrainingData(
                product_id=product_id,
                tahun=int(row["tahun"]),
                bulan=int(row["bulan"]),
                pengunjung=int(row["pengunjung"]),
                tayangan=int(row["tayangan"]),
                pesanan=int(row["pesanan"]),
                terjual=int(row["terjual"]),
                created_at=datetime.utcnow()
            ))
        db.commit()

        # ----------------------------------------------
        # 3. Ambil seluruh training untuk produk ini
        # ----------------------------------------------
        all_training = db.query(TrainingData).filter_by(product_id=product_id).all()
        if not all_training:
            return jsonify({"error": "Tidak ada data latih untuk produk ini"}), 400

        df_all = pd.DataFrame([{
            "pengunjung": t.pengunjung,
            "tayangan": t.tayangan,
            "pesanan": t.pesanan,
            "terjual": t.terjual
        } for t in all_training])

        X = df_all[["pengunjung", "tayangan", "pesanan"]]
        y = df_all["terjual"]

        # ----------------------------------------------
        # 4. Train Linear Regression
        # ----------------------------------------------
        model = LinearRegression()
        model.fit(X, y)

        intercept = float(model.intercept_)
        b1, b2, b3 = map(float, model.coef_)

        # ----------------------------------------------
        # 5. Update ModelStore
        # ----------------------------------------------
        latest_model = db.query(ModelStore)\
            .filter_by(product_id=product_id)\
            .order_by(ModelStore.id.desc()).first()

        if latest_model:
            # Update model existing
            latest_model.intercept = intercept
            latest_model.b1 = b1
            latest_model.b2 = b2
            latest_model.b3 = b3
            latest_model.created_at = datetime.utcnow()
        else:
            # Simpan model baru
            latest_model = ModelStore(
                product_id=product_id,
                intercept=intercept,
                b1=b1,
                b2=b2,
                b3=b3,
                created_at=datetime.utcnow()
            )
            db.add(latest_model)

        db.commit()

        # ----------------------------------------------
        # 6. Hitung evaluasi jika ada data testing
        # ----------------------------------------------
        testing_data = db.query(TestingData).filter_by(product_id=product_id).all()

        if testing_data:
            df_test = pd.DataFrame([{
                "pengunjung": d.pengunjung,
                "tayangan": d.tayangan,
                "pesanan": d.pesanan,
                "terjual": d.terjual
            } for d in testing_data])

            X_test = df_test[["pengunjung", "tayangan", "pesanan"]]
            y_test = df_test["terjual"]

            pred = (
                intercept
                + b1 * X_test["pengunjung"]
                + b2 * X_test["tayangan"]
                + b3 * X_test["pesanan"]
            ).clip(lower=0)

            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            mape = mean_absolute_percentage_error(y_test, pred)

            evaluation = db.query(ModelEvaluation).filter_by(model_id=latest_model.id).first()
            if evaluation:
                evaluation.r2_score = r2
                evaluation.mae = mae
                evaluation.mape = mape
                evaluation.created_at = datetime.utcnow()
            else:
                db.add(ModelEvaluation(
                    model_id=latest_model.id,
                    r2_score=r2,
                    mae=mae,
                    mape=mape,
                    created_at=datetime.utcnow()
                ))

        db.commit()

        return jsonify({
            "message": "Model berhasil dilatih untuk produk ini",
            "model": {
                "intercept": intercept,
                "b1": b1,
                "b2": b2,
                "b3": b3
            }
        })

    except Exception as e:
        if db:
            db.rollback()
        return jsonify({"error": f"Gagal melatih model: {str(e)}"}), 500

    finally:
        if db:
            db.close()


# untuk menampilkan semua data training pada tabel
@app.route("/training-data", methods=["GET"])
def get_training_data():
    try:
        db = SessionLocal()

        # Ambil parameter product_id dari query
        product_id = request.args.get("product_id", type=int)

        query = db.query(TrainingData, Product).join(Product, TrainingData.product_id == Product.id)

        # Filter berdasarkan produk jika ada
        if product_id:
            query = query.filter(TrainingData.product_id == product_id)

        rows = query.all()
        db.close()

        # Format JSON rapi
        result = [
            {
                "id": data.id,
                "product_id": data.product_id,
                "product_name": product.name,
                "tahun": data.tahun,
                "bulan": data.bulan,
                "pengunjung": data.pengunjung,
                "tayangan": data.tayangan,
                "pesanan": data.pesanan,
                "terjual": data.terjual,
                "created_at": data.created_at.isoformat()
            }
            for data, product in rows
        ]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# untuk prediksi penjualan
@app.route("/predict", methods=["POST"])
def predict():
    try:
        product_id = request.form.get("product_id", type=int)
        tahun = request.form.get("tahun", type=int)
        bulan = request.form.get("bulan", type=int)

        pengunjung = request.form.get("pengunjung", type=float)
        tayangan = request.form.get("tayangan", type=float)
        pesanan = request.form.get("pesanan", type=float)

        # ---------------- VALIDASI ----------------
        if not all([product_id, tahun, bulan]):
            return jsonify({"error": "product_id, tahun, dan bulan wajib diisi"}), 400

        if bulan < 1 or bulan > 12:
            return jsonify({"error": "Bulan harus antara 1 sampai 12"}), 400

        # ---------------- AMBIL MODEL ----------------
        db = SessionLocal()

        model_data = (
            db.query(ModelStore)
            .filter_by(product_id=product_id)
            .order_by(ModelStore.created_at.desc())
            .first()
        )

        if not model_data:
            return jsonify({"error": "Model belum tersedia untuk produk ini"}), 400

        # ---------------- HITUNG PREDIKSI ----------------
        prediksi = (
            model_data.intercept
            + model_data.b1 * pengunjung
            + model_data.b2 * tayangan
            + model_data.b3 * pesanan
        )

        prediksi = round(prediksi)

        # ---------------- SIMPAN KE DATABASE ----------------
        existing = (
            db.query(PredictionResult)
            .filter_by(product_id=product_id, tahun=tahun, bulan=bulan)
            .first()
        )

        if existing:
            # Update jika sudah ada
            existing.pengunjung = int(pengunjung)
            existing.tayangan = int(tayangan)
            existing.pesanan = int(pesanan)
            existing.predicted_terjual = prediksi
            existing.created_at = datetime.utcnow()
        else:
            db.add(PredictionResult(
                product_id=product_id,
                tahun=tahun,
                bulan=bulan,
                pengunjung=int(pengunjung),
                tayangan=int(tayangan),
                pesanan=int(pesanan),
                predicted_terjual=prediksi
            ))

        db.commit()

        return jsonify({
            "message": "Prediksi berhasil",
            "data": {
                "product_id": product_id,
                "tahun": tahun,
                "bulan": bulan,
                "predicted_terjual": prediksi
            }
        })

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/predictions/<int:product_id>", methods=["GET"])
def get_predictions(product_id):
    db = SessionLocal()
    try:
        data = (
            db.query(PredictionResult)
            .filter_by(product_id=product_id)
            .order_by(PredictionResult.tahun, PredictionResult.bulan)
            .all()
        )

        return jsonify([
            {
                "id": d.id,
                "tahun": d.tahun,
                "bulan": d.bulan,
                "pengunjung": d.pengunjung,
                "tayangan": d.tayangan,
                "pesanan": d.pesanan,
                "predicted_terjual": round(d.predicted_terjual, 2),
                "created_at": d.created_at.isoformat()
            }
            for d in data
        ])
    finally:
        db.close()

@app.route("/predictions/<int:product_id>", methods=["DELETE"])
def delete_all_predictions(product_id):
    db = SessionLocal()
    try:
        deleted = (
            db.query(PredictionResult)
            .filter(PredictionResult.product_id == product_id)
            .delete()
        )
        db.commit()

        return jsonify({
            "message": "Semua riwayat prediksi berhasil dihapus",
            "product_id": product_id,
            "deleted_count": deleted
        })
    except Exception as e:
        db.rollback()
        return jsonify({
            "error": f"Gagal menghapus riwayat prediksi: {str(e)}"
        }), 500
    finally:
        db.close()


# untuk menampilkan Koefisien regresi model    
@app.route("/model-info", methods=["GET"])
def model_info():
    db = SessionLocal()

    # Parameter produk wajib
    product_id = request.args.get("product_id", type=int)
    if not product_id:
        return jsonify({"error": "product_id wajib disertakan"}), 400

    # Hitung jumlah data latih untuk produk ini
    data_count = db.query(TrainingData).filter_by(product_id=product_id).count()

    # Ambil model terbaru untuk produk ini
    model_data = (
        db.query(ModelStore)
        .filter(ModelStore.product_id == product_id)
        .order_by(ModelStore.id.desc())
        .first()
    )

    if not model_data:
        return jsonify({"error": "Model untuk produk ini belum tersedia"}), 400

    # Konversi waktu UTC → WIB
    utc_time = model_data.created_at
    jakarta_tz = timezone("Asia/Jakarta")
    wib_time = utc_time.astimezone(jakarta_tz)

    db.close()

    return jsonify({
        "product_id": product_id,
        "jumlah_data": data_count,
        "intercept": model_data.intercept,
        "b1": model_data.b1,
        "b2": model_data.b2,
        "b3": model_data.b3,
        "updated_at": wib_time.strftime("%d-%m-%Y %H:%M:%S")
    })


@app.route("/latest-evaluation", methods=["GET"])
def get_latest_evaluation():
    """Get latest evaluation metrics per product"""

    product_id = request.args.get("product_id", type=int)
    if not product_id:
        return jsonify({"error": "product_id wajib disertakan"}), 400

    try:
        db = SessionLocal()
        try:
            # =======================
            # 1. Ambil model per produk
            # =======================
            model_data = (
                db.query(ModelStore)
                .filter(ModelStore.product_id == product_id)
                .order_by(ModelStore.created_at.desc())
                .first()
            )

            if not model_data:
                return jsonify({"error": "Model untuk produk ini belum tersedia"}), 400

            # =======================
            # 2. Ambil evaluasi model
            # =======================
            evaluation = (
                db.query(ModelEvaluation)
                .filter_by(model_id=model_data.id)
                .first()
            )

            if not evaluation:
                return jsonify({"error": "Evaluasi untuk model produk ini belum tersedia"}), 400

            # =======================
            # 3. Hitung jumlah testing data produk ini
            # =======================
            testing_count = (
                db.query(TestingData)
                .filter(TestingData.product_id == product_id)
                .count()
            )

            # =======================
            # 4. Response
            # =======================
            return jsonify({
                "product_id": product_id,
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
        return jsonify({"error": f"Error: {str(e)}"}), 500
    

# untuk logic evaluasi model
@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate model per produk dengan data testing"""
    product_id = request.form.get("product_id", type=int)
    if not product_id:
        return jsonify({"error": "product_id wajib disertakan"}), 400

    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    try:
        # Validasi format file Excel
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({"error": "File harus berformat Excel (.xlsx atau .xls)"}), 400

        df = pd.read_excel(file)

        # Validasi kolom
        required_columns = ["tahun", "bulan", "pengunjung", "tayangan", "pesanan", "terjual"]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"File harus memiliki kolom: tahun, bulan, pengunjung, tayangan, pesanan, terjual"}), 400

        # Validasi nilai
        if (df[required_columns] < 0).any().any():
            return jsonify({"error": "Data tidak boleh mengandung nilai negatif"}), 400
        
        if df[required_columns].isnull().any().any():
            return jsonify({"error": "Data tidak boleh mengandung nilai kosong"}), 400
        
        if not df["bulan"].between(1, 12).all():
            return jsonify({"error": "Nilai bulan harus antara 1 sampai 12"}), 400

        if len(df) == 0:
            return jsonify({"error": "File tidak boleh kosong"}), 400

        X = df[["pengunjung", "tayangan", "pesanan"]]
        y = df["terjual"]

        db = SessionLocal()
        try:
            # =========================
            # 1. Ambil model berdasarkan produk
            # =========================
            model_data = (
                db.query(ModelStore)
                .filter(ModelStore.product_id == product_id)
                .order_by(ModelStore.id.desc())
                .first()
            )

            if not model_data:
                return jsonify({"error": "Belum ada model untuk produk ini"}), 400

            # Buat ulang model dari database
            model = LinearRegression()
            model.intercept_ = model_data.intercept
            model.coef_ = np.array([model_data.b1, model_data.b2, model_data.b3])

            # Prediksi
            y_pred = model.predict(X)
            y_pred_clipped = np.clip(y_pred, 0, None)

            # Evaluasi
            r2 = r2_score(y, y_pred_clipped)
            mae = mean_absolute_error(y, y_pred_clipped)
            mape = mean_absolute_percentage_error(y, y_pred_clipped)

            # =========================
            # 4. Hapus testing lama produk ini & simpan yang baru
            # =========================
            db.query(TestingData).filter_by(product_id=product_id).delete()

            for i, row in df.iterrows():
                db.add(TestingData(
                    product_id=product_id,
                    tahun=int(row["tahun"]),
                    bulan=int(row["bulan"]),
                    pengunjung=int(row["pengunjung"]),
                    tayangan=int(row["tayangan"]),
                    pesanan=int(row["pesanan"]),
                    terjual=int(row["terjual"]),
                    predicted=float(y_pred_clipped[i])
                ))

            # =========================
            # 5. Update Evaluation Model Per Produk
            # =========================
            existing_eval = (
                db.query(ModelEvaluation)
                .filter_by(model_id=model_data.id)
                .first()
            )

            if existing_eval:
                existing_eval.r2_score = float(r2)
                existing_eval.mae = float(mae)
                existing_eval.mape = float(mape)
                existing_eval.created_at = datetime.utcnow()
            else:
                db.add(ModelEvaluation(
                    model_id=model_data.id,
                    r2_score=float(r2),
                    mae=float(mae),
                    mape=float(mape)
                ))

            db.commit()

            return jsonify({
                "message": "Evaluasi berhasil",
                "product_id": product_id,
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

    except Exception as e:
        return jsonify({"error": f"Gagal evaluasi: {str(e)}"}), 500


@app.route("/testing-data/<int:product_id>", methods=["GET"])
def get_testing_data_by_product(product_id):
    """Mengambil data testing beserta prediksi untuk 1 produk tertentu"""
    try:
        db = SessionLocal()
        try:
            # Ambil data testing untuk produk tertentu
            all_data = (
                db.query(TestingData)
                .filter(TestingData.product_id == product_id)
                .order_by(TestingData.created_at.desc())
                .all()
            )

            if not all_data:
                return jsonify([]), 200

            result = []
            for data in all_data:

                # Hitung error jika tersedia prediksi
                error = None
                if data.predicted is not None:
                    error = round(abs(data.terjual - data.predicted), 2)

                result.append({
                    "id": data.id,
                    "product_id": data.product_id,
                    "tahun": data.tahun,
                    "bulan": data.bulan,
                    "pengunjung": data.pengunjung,
                    "tayangan": data.tayangan,
                    "pesanan": data.pesanan,
                    "terjual": data.terjual,
                    "prediksi": data.predicted,
                    "error": error,
                    "created_at": data.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })

            return jsonify(result), 200

        finally:
            db.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/train/delete-by-product/<int:product_id>", methods=["DELETE"])
def delete_training_by_product(product_id):
    try:
        db = SessionLocal()

        # Hapus data training hanya untuk product_id tertentu
        deleted_rows = db.query(TrainingData).filter(
            TrainingData.product_id == product_id
        ).delete()

        db.commit()
        db.close()

        return jsonify({
            "message": f"{deleted_rows} data latih untuk produk ID {product_id} berhasil dihapus."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# untuk menambahkan produk
@app.route("/products", methods=["POST"])
def create_product():
    try:
        data = request.json
        name = str(data.get("name", "")).strip() if data.get("name") else ""
        kode = str(data.get("kode", "")).strip() if data.get("kode") else ""

        if not name:
            return jsonify({"error": "Nama produk wajib diisi"}), 400
        
        if not kode:
            return jsonify({"error": "Kode produk wajib diisi"}), 400

        db = SessionLocal()

        # Cek duplikasi
        existing_name = db.query(Product).filter(Product.name == name).first()
        if existing_name:
            db.close()
            return jsonify({"error": "Produk dengan nama ini sudah ada"}), 400

        existing_kode = db.query(Product).filter(Product.kode == kode).first()
        if existing_kode:
            db.close()
            return jsonify({"error": "Produk dengan kode ini sudah ada"}), 400

        new_product = Product(
            name=name,
            kode=kode
        )

        db.add(new_product)
        db.commit()

        result = {
            "id": new_product.id,
            "name": new_product.name,
            "kode": new_product.kode,
            "created_at": new_product.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

        db.close()
        return jsonify(result), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# untuk menampilkan semua daftar produk
@app.route("/products", methods=["GET"])
def get_products():
    try:
        db = SessionLocal()
        products = db.query(Product).order_by(Product.created_at.asc()).all()
        db.close()

        result = [
            {
                "id": p.id,
                "name": p.name,
                "kode": p.kode,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for p in products
        ]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/products/<int:product_id>", methods=["PUT"])
def update_product(product_id):
    """
    Update nama produk berdasarkan product_id
    Body (JSON):
    {
        "name": "Nama Produk Baru"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Data wajib diisi"}), 400

        name = str(data.get("name", "")).strip() if data.get("name") else ""
        kode = str(data.get("kode", "")).strip() if data.get("kode") else ""

        if not name:
            return jsonify({"error": "Field 'name' wajib diisi"}), 400
        
        if not kode:
            return jsonify({"error": "Field 'kode' wajib diisi"}), 400

        db = SessionLocal()
        try:
            product = db.query(Product).filter_by(id=product_id).first()

            if not product:
                return jsonify({"error": "Produk tidak ditemukan"}), 404

            # Validasi duplikasi nama produk (kecuali produk yang sedang diupdate)
            existing_name = db.query(Product).filter(
                Product.name == name,
                Product.id != product_id
            ).first()
            
            if existing_name:
                return jsonify({"error": "Produk dengan nama ini sudah ada"}), 400

            # Validasi duplikasi kode produk (kecuali produk yang sedang diupdate)
            existing_kode = db.query(Product).filter(
                Product.kode == kode,
                Product.id != product_id
            ).first()
            
            if existing_kode:
                return jsonify({"error": "Produk dengan kode ini sudah ada"}), 400

            # Update nama dan kode produk
            product.name = name
            product.kode = kode
            db.commit()

            return jsonify({
                "message": "Produk berhasil diperbarui",
                "product": {
                    "id": product.id,
                    "name": product.name,
                    "kode": product.kode
                }
            }), 200

        finally:
            db.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/products/<int:product_id>", methods=["DELETE"])
def delete_product(product_id):
    try:
        db = SessionLocal()
        product = db.query(Product).filter(Product.id == product_id).first()

        if not product:
            db.close()
            return jsonify({"error": "Produk tidak ditemukan"}), 404

        db.delete(product)
        db.commit()
        db.close()

        return jsonify({"message": "Produk berhasil dihapus"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/training-data/last/<int:product_id>", methods=["GET"])
def get_last_training_data(product_id):
    db = SessionLocal()
    try:
        data = (
            db.query(TrainingData)
            .filter(TrainingData.product_id == product_id)
            .order_by(
                TrainingData.tahun.desc(),
                TrainingData.bulan.desc()
            )
            .first()
        )

        if not data:
            return jsonify({
                "error": "Data historis belum tersedia"
            }), 404

        return jsonify({
            "data": {
                "pengunjung": data.pengunjung,
                "tayangan": data.tayangan,
                "pesanan": data.pesanan
            }
        })
    finally:
        db.close()

@app.route("/forecast-inputs/<int:product_id>", methods=["POST"])
def forecast_inputs(product_id):
    """
    Forecast pengunjung, tayangan, pesanan untuk periode berikutnya
    menggunakan ARIMA berdasarkan data training
    
    Request body (optional):
    {
        "tahun": 2024,
        "bulan": 3
    }
    
    Response:
    {
        "forecasted_pengunjung": 1250.5,
        "forecasted_tayangan": 3400.2,
        "forecasted_pesanan": 450.8,
        "next_period": {
            "tahun": 2024,
            "bulan": 3
        }
    }
    """
    db = None
    try:
        db = SessionLocal()
        
        # Ambil semua data training untuk produk ini, urutkan berdasarkan waktu
        training_data = (
            db.query(TrainingData)
            .filter(TrainingData.product_id == product_id)
            .order_by(TrainingData.tahun.asc(), TrainingData.bulan.asc())
            .all()
        )
        
        if not training_data:
            return jsonify({
                "error": "Tidak ada data training untuk produk ini"
            }), 400
        
        if len(training_data) < 3:
            return jsonify({
                "error": "Data training minimal 3 periode untuk forecasting"
            }), 400
        
        # Konversi ke DataFrame
        df = pd.DataFrame([{
            "tahun": t.tahun,
            "bulan": t.bulan,
            "pengunjung": t.pengunjung,
            "tayangan": t.tayangan,
            "pesanan": t.pesanan
        } for t in training_data])
        
        # Forecast untuk masing-masing variabel
        forecasted_pengunjung = forecast_variable(df["pengunjung"])
        forecasted_tayangan = forecast_variable(df["tayangan"])
        forecasted_pesanan = forecast_variable(df["pesanan"])
        
        # Hitung periode berikutnya
        last_data = training_data[-1]
        next_bulan = last_data.bulan + 1
        next_tahun = last_data.tahun
        
        if next_bulan > 12:
            next_bulan = 1
            next_tahun += 1
        
        return jsonify({
            "forecasted_pengunjung": round(forecasted_pengunjung),
            "forecasted_tayangan": round(forecasted_tayangan),
            "forecasted_pesanan": round(forecasted_pesanan),
            "next_period": {
                "tahun": next_tahun,
                "bulan": next_bulan
            },
            "message": "Forecast berhasil menggunakan ARIMA"
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Gagal melakukan forecast: {str(e)}"
        }), 500
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    app.run(port=5000, debug=True)

