from flask import Flask, render_template, request
import joblib
import os
import sqlite3
from datetime import datetime


app = Flask(__name__)


# ============================================
# PROJECT PATH
# ============================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================
# MODEL PATHS
# ============================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "heart_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "scaler.pkl"
)


# ============================================
# DATABASE PATH
# ============================================

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "prediction_history.db"
)


# ============================================
# LOAD MODEL
# ============================================

try:

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Heart disease model loaded successfully.")

except Exception as error:

    model = None
    scaler = None

    print("Model loading error:")
    print(error)


# ============================================
# DATABASE INITIALIZATION
# ============================================

def initialize_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date_time TEXT NOT NULL,

            age REAL NOT NULL,

            sex REAL NOT NULL,

            chest_pain REAL NOT NULL,

            blood_pressure REAL NOT NULL,

            cholesterol REAL NOT NULL,

            fasting_blood_sugar REAL NOT NULL,

            resting_ecg REAL NOT NULL,

            max_heart_rate REAL NOT NULL,

            exercise_angina REAL NOT NULL,

            oldpeak REAL NOT NULL,

            slope REAL NOT NULL,

            ca REAL NOT NULL,

            thal REAL NOT NULL,

            result TEXT NOT NULL,

            probability REAL

        )
    """)

    connection.commit()
    connection.close()


initialize_database()


# ============================================
# DASHBOARD
# ============================================

@app.route("/")
@app.route("/dashboard")
def dashboard():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # Total predictions
    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total_predictions = cursor.fetchone()[0]

    # High risk predictions
    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE result = 'High Risk'
    """)

    high_risk = cursor.fetchone()[0]

    # Low risk predictions
    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE result = 'Low Risk'
    """)

    low_risk = cursor.fetchone()[0]

    # Recent predictions
    cursor.execute("""
        SELECT
            date_time,
            result,
            probability
        FROM predictions
        ORDER BY id DESC
        LIMIT 5
    """)

    recent_predictions = cursor.fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        high_risk=high_risk,
        low_risk=low_risk,
        recent_predictions=recent_predictions
    )


# ============================================
# PREDICTION PAGE
# ============================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":

        return render_template(
            "prediction.html"
        )

    try:

        # ====================================
        # GET FORM VALUES
        # ====================================

        age = float(
            request.form["age"]
        )

        sex = float(
            request.form["sex"]
        )

        cp = float(
            request.form["cp"]
        )

        trestbps = float(
            request.form["trestbps"]
        )

        chol = float(
            request.form["chol"]
        )

        fbs = float(
            request.form["fbs"]
        )

        restecg = float(
            request.form["restecg"]
        )

        thalach = float(
            request.form["thalach"]
        )

        exang = float(
            request.form["exang"]
        )

        oldpeak = float(
            request.form["oldpeak"]
        )

        slope = float(
            request.form["slope"]
        )

        ca = float(
            request.form["ca"]
        )

        thal = float(
            request.form["thal"]
        )


        # ====================================
        # CREATE INPUT
        # ====================================

        input_data = [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]]


        # ====================================
        # MODEL CHECK
        # ====================================

        if model is None or scaler is None:

            return render_template(
                "result.html",
                error="Model could not be loaded."
            )


        # ====================================
        # SCALE INPUT
        # ====================================

        input_scaled = scaler.transform(
            input_data
        )


        # ====================================
        # PREDICT
        # ====================================

        prediction = model.predict(
            input_scaled
        )[0]


        # ====================================
        # PROBABILITY
        # ====================================

        probability = None

        try:

            probability = model.predict_proba(
                input_scaled
            )[0][1]

            probability = round(
                probability * 100,
                2
            )

        except Exception:

            pass


        # ====================================
        # RESULT
        # ====================================

        if prediction == 1:

            result = "High Risk"

            message = (
                "The model predicts a higher "
                "likelihood of heart disease."
            )

        else:

            result = "Low Risk"

            message = (
                "The model predicts a lower "
                "likelihood of heart disease."
            )


        # ====================================
        # SAVE PREDICTION
        # ====================================

        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO predictions (

                date_time,
                age,
                sex,
                chest_pain,
                blood_pressure,
                cholesterol,
                fasting_blood_sugar,
                resting_ecg,
                max_heart_rate,
                exercise_angina,
                oldpeak,
                slope,
                ca,
                thal,
                result,
                probability

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?)
        """, (

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal,
            result,
            probability
        ))

        connection.commit()
        connection.close()


        # ====================================
        # RESULT PAGE
        # ====================================

        return render_template(
            "result.html",
            prediction=int(prediction),
            result=result,
            probability=probability,
            message=message
        )


    except Exception as error:

        print("Prediction Error:")
        print(error)

        return render_template(
            "result.html",
            error=str(error)
        )


# ============================================
# PREDICTION HISTORY
# ============================================

@app.route("/history")
def history():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    predictions = cursor.fetchall()

    connection.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# ============================================
# DELETE HISTORY
# ============================================

@app.route("/history/delete/<int:prediction_id>")
def delete_history(prediction_id):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE id = ?",
        (prediction_id,)
    )

    connection.commit()
    connection.close()

    return history()


# ============================================
# CLEAR ALL HISTORY
# ============================================

@app.route("/history/clear")
def clear_history():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    connection.commit()
    connection.close()

    return history()


# ============================================
# HEALTH CHECK
# ============================================

@app.route("/health")
def health():

    if model is not None and scaler is not None:

        return {
            "status": "ok",
            "model": "loaded"
        }

    return {
        "status": "error",
        "model": "not loaded"
    }, 500


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":

    app.run(
        debug=True
    )