from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


with open('mlp_classifier_model.pkl', 'rb') as mlp_model_file:
    mlp_model = pickle.load(mlp_model_file)


@app.route('/')
def home():
    return render_template('index.html')


from flask import request, jsonify

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
      
        type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        type_CASH_OUT = None
        type_DEBIT = None
        type_PAYMENT = None
        type_TRANSFER = None 

        if type == 'type_CASH_OUT':
            type_CASH_OUT = True
            type_DEBIT = False
            type_PAYMENT = False
            type_TRANSFER = False 
            type = 'CASH_OUT'
        elif type == 'type_DEBIT':
            type_CASH_OUT = False
            type_DEBIT = True
            type_PAYMENT = False
            type_TRANSFER = False
            type = 'DEBIT'
        if type == 'type_PAYMENT':
            type_CASH_OUT = False
            type_DEBIT = False
            type_PAYMENT = True
            type_TRANSFER = False 
            type = 'PAYMENT'
        if type == 'type_TRANSFER':
            type_CASH_OUT = False
            type_DEBIT = False
            type_PAYMENT = False
            type_TRANSFER = True
            type = 'TRANSFER'


        data = {
            'type': type,
            'type_CASH_OUT':type_CASH_OUT,
            'type_DEBIT': type_DEBIT,
            'type_PAYMENT': type_PAYMENT,
            'type_TRANSFER': type_TRANSFER,
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
        }
        df = pd.DataFrame(data)
        print(df)
        df['Log_Transaction_Amount'] = np.log1p(df['amount'])
        df['Transaction_Amount_Difference'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['Origin_Account_Balance_Difference'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['Destination_Account_Balance_Difference'] = df['newbalanceDest'] - df['oldbalanceDest']
        df['Origin_Account_Balance_Ratio'] = df['oldbalanceOrg'] / df['newbalanceOrig']
        df['Destination_Account_Balance_Ratio'] = df['oldbalanceDest'] / df['newbalanceDest']
        df['Transaction_Amount_to_Balance_Difference_Ratio'] = df['amount'] / df['Origin_Account_Balance_Difference']
        df['Transaction_Amount_to_Destination_Account_Balance_Ratio'] = df['amount'] / df['Destination_Account_Balance_Ratio']
        mean_transaction_amount_by_type = df.groupby('type')['amount'].mean()
        median_transaction_amount_by_type = df.groupby('type')['amount'].median()
        std_dev_transaction_amount_by_type = df.groupby('type')['amount'].std()

    
        df = df.merge(mean_transaction_amount_by_type, on='type', suffixes=('', '_mean_by_type'))
        df = df.merge(median_transaction_amount_by_type, on='type', suffixes=('', '_median_by_type'))
        df = df.merge(std_dev_transaction_amount_by_type, on='type', suffixes=('', '_std_dev_by_type'))
   
        columns_to_include = [
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'Log_Transaction_Amount', 'Transaction_Amount_Difference',
            'Origin_Account_Balance_Difference', 'Destination_Account_Balance_Difference',
            'Origin_Account_Balance_Ratio', 'Destination_Account_Balance_Ratio',
            'Transaction_Amount_to_Balance_Difference_Ratio',
            'Transaction_Amount_to_Destination_Account_Balance_Ratio',
            'amount_mean_by_type', 'amount_median_by_type', 'amount_std_dev_by_type',
           
        ]
      

        new_df = df[columns_to_include]
        
        columns_with_nan = ['Origin_Account_Balance_Ratio', 'Destination_Account_Balance_Ratio', 'Transaction_Amount_to_Balance_Difference_Ratio', 'Transaction_Amount_to_Destination_Account_Balance_Ratio']
        new_df = new_df.drop(columns=columns_with_nan)

        new_df['isFlaggedFraud'] = 0
        scaler = MinMaxScaler()

 
        columns_to_normalize = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'Log_Transaction_Amount', 'Transaction_Amount_Difference',
            'Origin_Account_Balance_Difference', 'Destination_Account_Balance_Difference',
            'amount_mean_by_type', 'amount_median_by_type', 'amount_std_dev_by_type',
        ]

        # Fit the scaler to the selected columns and transform the data
        new_df[columns_to_normalize] = scaler.fit_transform(new_df[columns_to_normalize])
        columns_to_replace_nan_with_zero = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'Log_Transaction_Amount', 'Transaction_Amount_Difference',
            'Origin_Account_Balance_Difference', 'Destination_Account_Balance_Difference',
            'amount_mean_by_type', 'amount_median_by_type', 'amount_std_dev_by_type',
        ]

        for col in columns_to_replace_nan_with_zero:
            if new_df[col].isna().any():
                new_df[col] = new_df[col].fillna(1)
                res = mlp_model.predict(new_df)

        print(res)

        if res == [1]:
            response = {'message': 'FRAUDLENT ACTIVITY'}
        else:
            response = {'message': 'No Fraudulent Behavior Detected'}

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

