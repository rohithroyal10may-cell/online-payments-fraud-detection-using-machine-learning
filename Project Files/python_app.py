from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load trained model and scaler
try:
    model = joblib.load(os.path.join("code files", "fraud_model.pkl"))
    scaler = joblib.load(os.path.join("code files", "scaler.pkl"))
    encoders = joblib.load(os.path.join("code files", "encoders.pkl"))
    print("✅ Model, scaler, and encoders loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    model = None
    scaler = None
    encoders = None

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Fraud Detection System</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Arial', sans-serif; 
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #fff;
            }
            .container {
                background: rgba(20, 20, 40, 0.9);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
                max-width: 600px;
                width: 100%;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                color: #00d4ff;
                font-size: 28px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #00d4ff;
                font-weight: bold;
            }
            input {
                width: 100%;
                padding: 12px;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                color: #fff;
                font-size: 14px;
            }
            input::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }
            input:focus {
                outline: none;
                border-color: #00ff88;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
            }
            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #00d4ff, #00ff88);
                color: #000;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0, 212, 255, 0.4);
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 18px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .fraud {
                background: rgba(255, 50, 50, 0.2);
                border: 2px solid #ff3232;
                color: #ff6b6b;
            }
            .legitimate {
                background: rgba(50, 255, 100, 0.2);
                border: 2px solid #32ff64;
                color: #6bff77;
            }
            .confidence {
                margin-top: 10px;
                font-size: 16px;
                color: #00d4ff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💳 Fraud Detection System</h1>
            
            <div class="info" style="background: rgba(0,212,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 13px;">
                <strong>Transaction Types:</strong><br>
                0: CASH_IN | 1: CASH_OUT | 2: DEBIT | 3: PAYMENT | 4: TRANSFER
            </div>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label>Step (1-743)</label>
                    <input type="number" id="step" placeholder="e.g., 1" value="1" required>
                </div>
                
                <div class="form-group">
                    <label>Type (0-4)</label>
                    <input type="number" id="type" placeholder="e.g., 0" value="0" required>
                </div>
                
                <div class="form-group">
                    <label>Amount</label>
                    <input type="number" id="amount" placeholder="e.g., 1000" value="1000" required>
                </div>
                
                <div class="form-group">
                    <label>Name Orig (Sender Name)</label>
                    <input type="text" id="nameOrig" placeholder="e.g., C1000001" value="C1000001" required>
                </div>
                
                <div class="form-group">
                    <label>Old Balance Org</label>
                    <input type="number" id="oldbalanceOrg" placeholder="e.g., 50000" value="50000" required>
                </div>
                
                <div class="form-group">
                    <label>New Balance Orig</label>
                    <input type="number" id="newbalanceOrig" placeholder="e.g., 49000" value="49000" required>
                </div>
                
                <div class="form-group">
                    <label>Name Dest (Receiver Name)</label>
                    <input type="text" id="nameDest" placeholder="e.g., C2000001" value="C2000001" required>
                </div>
                
                <div class="form-group">
                    <label>Old Balance Dest</label>
                    <input type="number" id="oldbalanceDest" placeholder="e.g., 10000" value="10000" required>
                </div>
                
                <div class="form-group">
                    <label>New Balance Dest</label>
                    <input type="number" id="newbalanceDest" placeholder="e.g., 11000" value="11000" required>
                </div>
                
                <button type="submit">🔍 Predict</button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const data = {
                    step: parseFloat(document.getElementById('step').value),
                    type: document.getElementById('type').value,
                    amount: parseFloat(document.getElementById('amount').value),
                    nameOrig: document.getElementById('nameOrig').value,
                    oldbalanceOrg: parseFloat(document.getElementById('oldbalanceOrg').value),
                    newbalanceOrig: parseFloat(document.getElementById('newbalanceOrig').value),
                    nameDest: document.getElementById('nameDest').value,
                    oldbalanceDest: parseFloat(document.getElementById('oldbalanceDest').value),
                    newbalanceDest: parseFloat(document.getElementById('newbalanceDest').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.error) {
                        resultDiv.className = 'result show fraud';
                        resultDiv.innerHTML = `❌ Error: ${result.error}`;
                    } else {
                        const confidence = (result.confidence * 100).toFixed(2);
                        
                        if (result.prediction === 1) {
                            resultDiv.className = 'result show fraud';
                            resultDiv.innerHTML = `🚨 FRAUD DETECTED<br><div class="confidence">Confidence: ${confidence}%</div>`;
                        } else {
                            resultDiv.className = 'result show legitimate';
                            resultDiv.innerHTML = `✅ LEGITIMATE TRANSACTION<br><div class="confidence">Confidence: ${confidence}%</div>`;
                        }
                    }
                } catch (error) {
                    alert('Error: ' + error);
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Run train.py first!'}), 500
        
        data = request.json
        
        # Create DataFrame with correct column order (EXACTLY as trained)
        df = pd.DataFrame([{
            'step': data['step'],
            'type': data['type'],
            'amount': data['amount'],
            'nameOrig': data['nameOrig'],
            'oldbalanceOrg': data['oldbalanceOrg'],
            'newbalanceOrig': data['newbalanceOrig'],
            'nameDest': data['nameDest'],
            'oldbalanceDest': data['oldbalanceDest'],
            'newbalanceDest': data['newbalanceDest']
        }])
        
        # Encode categorical columns using the saved encoders
        for col in ['type', 'nameOrig', 'nameDest']:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except:
                    # If value not seen during training, use a default
                    df[col] = 0
        
        # Scale using the trained scaler
        df_scaled = scaler.transform(df)
        
        # Predict
        pred = model.predict(df_scaled)[0]
        prob_array = model.predict_proba(df_scaled)[0]
        confidence = float(prob_array[int(pred)])
        
        return jsonify({
            'prediction': int(pred),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)