
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(_name_)

# Load scaler and model once at startup
scaler = pickle.load(open('normalizer.pkl', 'rb'))
model = pickle.load(open('rc_acc_68.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        age = float(request.form['age'])
        bilirubin = float(request.form['bilirubin'])
        alk_phosphate = float(request.form['alk_phosphate'])
        sgpt = float(request.form['sgpt'])
        sgot = float(request.form['sgot'])
        proteins = float(request.form['proteins'])
        albumin = float(request.form['albumin'])
        ag_ratio = float(request.form['ag_ratio'])

        # Arrange in order expected by scaler/model
        input_data = np.array([[sgpt, sgot, alk_phosphate, bilirubin, albumin, age]])
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction_raw = model.predict(scaled_input)[0]
        prediction = "🟢 No Liver Disease Detected" if prediction_raw == 0 else "🔴 Liver Disease Detected"

        return render_template('form.html', prediction=prediction, patient_name=name)
    
    except Exception as e:
        return render_template('form.html', prediction="❌ Error during prediction: " + str(e), patient_name="")

if _name_ == '_main_':
    app.run(debug=True)
