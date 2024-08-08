import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app =application


# import ridge regressor and standard  scaler pickle

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_data_point():
    if request.method == 'POST':
        temprature = float(request.form.get('Temprature'))
        rh = float(request.form.get('Rh'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[temprature,ws,rh,rain,ffmc,dmc,isi,classes,region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])

    return render_template('home.html')
  

if __name__ == '__main__':
    app.run(host='0.0.0.0')