
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(_name_)

# Load the normalizer and model
with open("normalizer.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rc_acc_68.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("form.html")

@app.route("/portfolio")
def portfolio():
    return render_template("portfolio.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        name = request.form.get("name")
        alt = float(request.form.get("alt"))
        ast = float(request.form.get("ast"))
        alp = float(request.form.get("alp"))
        bilirubin = float(request.form.get("bilirubin"))
        albumin = float(request.form.get("albumin"))
        age = float(request.form.get("age"))

        # Prepare data and normalize
        features = np.array([[alt, ast, alp, bilirubin, albumin, age]])
        scaled_features = scaler.transform(features)

        # Predict
        result = model.predict(scaled_features)[0]
        prediction_text = f"Prediction for {name}: Liver condition is {'abnormal' if result else 'normal'}."

        return render_template("form.html", prediction=prediction_text)

    except Exception as e:
        return f"Error: {str(e)}"

if _name_ == "_main_":
    app.run(debug=True)
