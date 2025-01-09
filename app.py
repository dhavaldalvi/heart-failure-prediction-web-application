from flask import Flask, url_for, redirect, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
sex_encoder = pickle.load(open('feature_sex.pkl', 'rb'))
chestpain_encoder = pickle.load(open('feature_chestpain.pkl', 'rb'))
exerciseangina_encoder = pickle.load(open('feature_exerciseangina.pkl', 'rb'))
restingecg_encoder = pickle.load(open('feature_restingecg.pkl', 'rb'))
st_slope_encoder = pickle.load(open('feature_stslope.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

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

    # Transfroming variables
    sex_final = sex_encoder.transform([Sex]).ravel()
    chestpain_final = chestpain_encoder.transform([ChestPainType]).ravel()
    restingecg_final = restingecg_encoder.transform([RestingECG]).ravel()
    exerciseangina_final = exerciseangina_encoder.transform([ExerciseAngina]).ravel()
    stslope_final = st_slope_encoder.transform([ST_Slope]).ravel()   
    age_final = np.array([Age])
    restingbp_final = np.array([RestingBP])
    cholesterol_final = np.array([Cholesterol])
    fastingbs_final = np.array([FastingBS])
    maxhr_final = np.array([MaxHR])
    oldpeak_final = np.array([Oldpeak])
    features = np.concatenate((age_final,sex_final,restingbp_final,cholesterol_final,fastingbs_final,maxhr_final,exerciseangina_final,oldpeak_final,chestpain_final,restingecg_final,stslope_final))
    features_scaled = scaler.transform(features.reshape(1,18))
    
    # Prediction
    value = model.predict_proba(features_scaled)

    # returns the response
    return render_template('result.html', prediction = value[0][0]*100)


if __name__ == '__main__':
    app.run(debug=True)