from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import datetime

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r'UPI_Fraud_Dataset_Expanded.csv', index_col=0)

# Fraud detection function using the dataset
def predict_fraud_from_csv(upi_number, transaction_amount, date, pincode):
    try:
        # Extract date components
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        trans_year = date_obj.year
        trans_month = date_obj.month
        trans_day = date_obj.day

        # Filter the dataset based on input details
        filtered_df = df[
            (df['UPI Number'] == int(upi_number)) &
            (df['Transaction Amount'] == float(transaction_amount)) &
            (df['Transaction Year'] == trans_year) &
            (df['Transaction Month'] == trans_month) &
            (df['Zip Code'] == int(pincode))
        ]
        
        # If a match is found, return the fraud status
        if not filtered_df.empty:
            return int(filtered_df.iloc[0]['Fraud'])
        else:
            # If no match is found, return 'not fraud' by default
            return 1
    except ValueError as e:
        print(f"Input error: {e}")
        return 1  # Default to 'not fraud' for invalid inputs
    except KeyError as e:
        print(f"Column not found: {e}")
        return 1  # Default to 'not fraud' for missing columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        # Extract form data
        upi_id = request.form['upi-id']
        transaction_amount = request.form['transaction-amount']
        date = request.form['date']
        pincode = request.form['pincode']

        # Call the prediction function
        prediction = predict_fraud_from_csv(upi_id, transaction_amount, date, pincode)

        # Interpret the prediction
        result_text = "Fraud Transaction" if prediction == 0 else "Not Fraud Transaction"

        # Pass the result to the result page
        return render_template('result.html', prediction=result_text)

    return render_template('details.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)