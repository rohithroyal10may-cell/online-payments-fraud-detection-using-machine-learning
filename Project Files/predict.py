import joblib
import pandas as pd
import numpy as np
import os

print("="*80)
print("FRAUD DETECTION PREDICTION SYSTEM")
print("="*80)

try:
    # Load model and encoders
    model = joblib.load(os.path.join("code files", "fraud_model.pkl"))
    scaler = joblib.load(os.path.join("code files", "scaler.pkl"))
    encoders = joblib.load(os.path.join("code files", "encoders.pkl"))
    
    print("\n[1] Model loaded successfully")
    
    # Create simple test data with numeric values
    print("\n[2] Creating test transaction...")
    test_data = {
        "step": [1],
        "type": [1],
        "amount": [50000.0],
        "nameOrig": [100],
        "oldbalanceOrg": [100000.0],
        "newbalanceOrig": [50000.0],
        "nameDest": [200],
        "oldbalanceDest": [5000.0],
        "newbalanceDest": [55000.0]
    }
    
    df = pd.DataFrame(test_data)
    print("Transaction data created")
    
    # Scale
    print("\n[3] Scaling features...")
    df_scaled = scaler.transform(df)
    print("Features scaled")
    
    # Predict
    print("\n[4] Making prediction...")
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    
    if prediction == 1:
        print("Status: FRAUD DETECTED")
        print(f"Confidence: {probability[1]:.4f} ({probability[1]*100:.2f}%)")
    else:
        print("Status: LEGITIMATE TRANSACTION")
        print(f"Confidence: {probability[0]:.4f} ({probability[0]*100:.2f}%)")
    
    print("="*80)
    print("Prediction complete!")
    print("="*80)
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print(f"Error type: {type(e).__name__}")