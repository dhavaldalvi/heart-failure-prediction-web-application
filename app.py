from flask import Flask, url_for, redirect, render_template, request
import pickle
import numpy as np
import subprocess
import pandas as pd

app = Flask(__name__)

# Running model file to get pickle file
initialized = False
@app.before_request
def run_model_script():
    global initialized
    if not initialized:
        subprocess.run(['python','model.py'], check=True)
        initialized = True

run_model_script()

# Loading the pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Rendering main page
@app.route('/')
def home():
    return render_template('index.html')

# Rendering result page 
@app.route('/predict', methods=['POST','GET'])
def predict():
    # Accessing featues
    Age = int(request.form["Age"])
    Sex = request.form['Sex']
    ChestPainType = request.form['ChestPainType']
    RestingBP = request.form['RestingBP']
    Cholesterol = request.form['Cholesterol']
    FastingBS = int(request.form['FastingBS'])
    RestingECG = request.form['RestingECG']
    MaxHR = request.form['MaxHR']
    ExerciseAngina = request.form['ExerciseAngina']
    Oldpeak = request.form['Oldpeak']
    ST_Slope = request.form['ST_Slope']

    # Making pandas dataframe to feed the featuers in model
    dict = [{'Age':Age, 'Sex':Sex, 'ChestPainType':ChestPainType, 'RestingBP':RestingBP, 'Cholesterol':Cholesterol, 'FastingBS':FastingBS,
        'RestingECG':RestingECG, 'MaxHR':MaxHR, 'ExerciseAngina':ExerciseAngina, 'Oldpeak':Oldpeak, 'ST_Slope':ST_Slope}]
    features = pd.DataFrame(dict)
    
    # Prediction
    value = model.predict_proba(features)[0][0]*100

    # returns the response
    return render_template('result.html', prediction = round(value,2))


if __name__ == '__main__':
    app.run(debug=True)