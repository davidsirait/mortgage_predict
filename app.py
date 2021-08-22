from flask import Flask
from flask import Flask, request
from flask import render_template, jsonify, url_for
import pandas as pd
import numpy as np
import pickle
import datetime
import time


app = Flask(__name__)

@app.route('/predict' ,methods=['POST'])
def Predict():

    with open('models/randomforest.pkl', 'rb') as f:
        model = pickle.load(f)
    final_feature = ['modIncomelog',
                     'modLoanlog',
                     'Loan_Amount_Term',
                     'Gender_Female',
                     'Gender_Male',
                     'Married_No',
                     'Married_Yes',
                     'Education_Graduate',
                     'Education_Not Graduate',
                     'Have_CC_No',
                     'Have_CC_Yes']

    cek = request.form
    print(cek)
    cek_df = pd.DataFrame(0, columns=final_feature, index=[0])
    cek_df['modIncomelog'] = np.log(int(cek['income']) + 1)
    cek_df['modLoanlog'] = np.log(int(cek['loan']) + 1)
    cek_df['Loan_Amount_Term'] =int(cek.get('tenor'))
    if cek.get('gender') == 'Male':
        cek_df['Gender_Male'] = 1
    else:
        cek_df['Gender_Female'] = 1

    if cek.get('education') in ['SMP', 'SMA', 'SMK', 'D3', 'D4', 'S1']:
        cek_df['Education_Not Graduate'] = 1
    else:
        cek_df['Education_Graduate'] = 1

    if cek.get('cc') == 'Yes':
        cek_df['Have_CC_Yes'] = 1
    else:
        cek_df['Have_CC_No'] = 1

    if cek.get('married') == 'Yes':
        cek_df['Married_Yes'] = 1
    else:
        cek_df['Married_No'] = 1

    score = model.predict_proba(cek_df)[:, 1] * 100
    results = {'names': cek['names'], 'score': int(score)}
    time.sleep(3)

    if score < 55 :
        return render_template('reject.html', results=results)
    else:
        return render_template('success.html', results=results)

@app.route('/')
@app.route('/index')
def Homepage():
    return render_template('index.html')


if __name__=='__main__':
   app.run(debug=False)