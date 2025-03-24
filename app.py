from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'models', 'fraud_detection_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

# Ensure the model and scaler exist before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Dummy encoders for categorical variables (Ensure these match training data)
le_transaction_type = LabelEncoder().fit(['UPI', 'Card', 'NetBanking'])
le_payment_gateway = LabelEncoder().fit(['Gateway1', 'Gateway2', 'Gateway3'])
le_city = LabelEncoder().fit(['City1', 'City2', 'City3'])
le_state = LabelEncoder().fit(['State1', 'State2', 'State3'])
le_ip = LabelEncoder().fit(['192.168.1.1', '10.0.0.1'])
le_status = LabelEncoder().fit(['Pending', 'Completed', 'Failed'])
le_device_os = LabelEncoder().fit(['Android', 'iOS', 'Windows'])
le_merchant_category = LabelEncoder().fit(['Food', 'Retail', 'Travel'])
le_channel = LabelEncoder().fit(['Mobile', 'Web', 'POS'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data
            hour = int(request.form['hour'])
            day = int(request.form['day'])
            month = int(request.form['month'])
            year = int(request.form['year'])
            amount = float(request.form['amount'])
            
            # Convert date to day of the week
            day_of_week = pd.Timestamp(f'{year}-{month}-{day}').dayofweek

            # Prepare DataFrame with required features
            data = pd.DataFrame({
                'Hour': [hour],
                'Day_of_Week': [day_of_week],
                'amount': [amount],
                'Transaction_Frequency': [1],  # Placeholder, should be computed dynamically
                'Transaction_Amount_Deviation': [0],  # Placeholder, update logic if needed
                'Days_Since_Last_Transaction': [0],  # Placeholder
                'Transaction_Type': [le_transaction_type.transform(['UPI'])[0]],  # Assuming UPI
                'Payment_Gateway': [le_payment_gateway.transform(['Gateway1'])[0]],  # Default
                'Transaction_City': [le_city.transform(['City1'])[0]],  # Default
                'Transaction_State': [le_state.transform(['State1'])[0]],  # Default
                'IP_Address': [le_ip.transform(['192.168.1.1'])[0]],  # Default
                'Transaction_Status': [le_status.transform(['Pending'])[0]],  # Default
                'Device_OS': [le_device_os.transform(['Android'])[0]],  # Default
                'Merchant_Category': [le_merchant_category.transform(['Retail'])[0]],  # Default
                'Transaction_Channel': [le_channel.transform(['Mobile'])[0]],  # Default
                'Is_High_Deviation': [0]  # Placeholder
            })

            # Normalize numerical columns
            num_columns = ['Transaction_Frequency', 'Transaction_Amount_Deviation', 
                           'Days_Since_Last_Transaction', 'amount']
            data[num_columns] = scaler.transform(data[num_columns])

            # Predict fraud
            prediction = model.predict(data)[0]
            prediction = 'Fraudulent' if prediction == 1 else 'Non-Fraudulent'
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
