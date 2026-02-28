from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'payments.pkl')

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Model file '{model_path}' not found.")

# Initialize StandardScaler with the expected parameters
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [
            float(request.form['step']),
            float(request.form['type']),
            float(request.form['amount']),
            float(request.form['oldbalanceOrg']),
            float(request.form['newbalanceOrig']),
            float(request.form['oldbalanceDest']),
            float(request.form['newbalanceDest'])
        ]
        
        df = pd.DataFrame([features], columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
        
        # Scale the features
        df_scaled = scaler.fit_transform(df)
        
        prediction = model.predict(df_scaled)[0]
        
        result = "Not Fraud" if prediction == 0 else "Fraud"
        
        return render_template('submit.html', result=result)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)