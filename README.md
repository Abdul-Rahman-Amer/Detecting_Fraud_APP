# Detecting_Fraud_APP

## Overview
This project deploys a Neural Network in a Flask application to take user input and detect fraudulent activity. The process involves understanding the data, preprocessing, feature engineering, model testing, and creating a user-friendly UI.

## Step 1 - Understanding the Data
Right from the start, it's evident that there's a significant minority class in the data, which prompted the use of SMOTE.

### SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE stands for "Synthetic Minority Over-sampling Technique." It is a technique used in machine learning and data analysis, particularly for imbalanced datasets. It addresses the issue where the number of instances in one class is significantly lower than in another, leading to challenges in model performance. 

#### How SMOTE Works
1. For each minority class instance, SMOTE selects *k* nearest neighbors from the same class.
2. A random neighbor is selected from the *k* nearest neighbors, and a synthetic example is created by interpolating between the selected instance and the random neighbor.
3. This process is repeated for a specified number of synthetic examples, effectively increasing the number of minority class instances in the dataset.

SMOTE helps balance the class distribution in the dataset, improving machine learning model performance.

![SMOTE](/photos_for_readme/smote.png)

### PCA (Principal Component Analysis)

PCA, or Principal Component Analysis, is used for dimensionality reduction in high-dimensional datasets while preserving essential information.

#### How PCA Works
1. Data is centered by subtracting the mean from each feature, ensuring the first principal component passes through the data's center.
2. A covariance matrix is constructed from the centered data, capturing feature relationships.
3. Eigenvalues and eigenvectors of the covariance matrix are found, with eigenvalues representing variances along corresponding eigenvectors.
4. Principal components are selected based on their eigenvalues, capturing the most critical data information.
5. Data is projected onto selected principal components, reducing dimensionality while retaining valuable information.

![PCA Visualization](/photos_for_readme/PCA_Visualization_1.png)

Feature selection with PCA proved useful, revealing insights into the importance of different features.

## In-Depth Analysis

To gain deeper insights, a heatmap and a precision matrix were generated.

### Heatmap and Precision Matrix
The heatmap analysis provided a detailed view of feature relationships.

![Heatmap](/photos_for_readme/PCA_Visualization_1_Heatmap.png)
![Precision Matrix](/photos_for_readme/Precision_Matrix.png)

### Principal Components

Principal components revealed how different components gathered signals from various features.

![Principal Component 1](/photos_for_readme/Princiapl_component.png)
![Principal Component 2](/photos_for_readme/Principal_Component_2.png)

## Model Testing

It was time to test various models for fraud detection. Several models were employed, each with its strengths and weaknesses.

### 1. Bayesian Neural Network (BNN)
- BNN extends traditional neural networks with probabilistic modeling, providing probability distributions over predictions. It quantifies uncertainty, essential in fraud detection.
- ![Image Alt Text](/photos_for_readme/Bayesian_MAtrix.png)

### 2. Logistic Regression (In NN)
- Logistic Regression is a classic model for binary classification. It models the probability of a binary outcome based on input features, providing interpretable coefficients.
-  ![Image Alt Text](/photos_for_readme/Logistic_Regression_Matrix.png) 

### 3. Decision Tree
- Decision Tree is a tree-based model that helps understand the decision-making process by creating a tree-like structure based on features.
- ![Image Alt Text](/photos_for_readme/Decision_Tree_Matrix.png)

### 4. Multi-Layer Perceptron (MLP)
- MLP is a neural network capable of capturing complex patterns and non-linear decision boundaries, well-suited for fraud detection.
![Image Alt Text](/photos_for_readme/MLP_Matrtix.png)

### 5. Hybrid Model (Decision Tree and MLP)
- The Hybrid Model combines the strengths of Decision Tree and MLP, boosting recall scores.
![Image Alt Text](/photos_for_readme/Hybrid_Matrix.png) 


In-depth testing revealed the best model for the fraud detection task.

## Model Deployment

The MLP model outperformed others, achieving a high recall score. This model was pickled and hosted in a Flask application.

### User Input

The Flask application offers a user-friendly interface where users can input data for fraud detection.

![App UI](/photos_for_readme/APP_UI.png)

### Model Prediction

JavaScript and Ajax are used to handle user requests, serving them to the Flask route `/predict`. The input is converted into a tensor for prediction by the pickled model.
```html
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
<script>
    function submitForm(event) {
        event.preventDefault(); // Prevent the form from submitting directly

        // Collect form data
        var formData = {
            type: $('#type').val(),
            amount: $('#amount').val(),
            oldbalanceOrg: $('#oldbalanceOrg').val(),
            newbalanceOrig: $('#newbalanceOrig').val(),
            oldbalanceDest: $('#oldbalanceDest').val(),
            newbalanceDest: $('#newbalanceDest').val()
        };

        // Make an AJAX request to the Flask route
        $.post("/predict", formData, function(response) {
            console.log('Response:', response);
            // Update the content of the modal with the response message
            $('#modalMessage').html('<p>' + response.message + '</p>');
            $('#resultModal').modal('show');
        });
    }
</script>

![Modal Prediction](/photos_for_readme/Modal_Prediciton.png)
```
Which converts the Json here and flashed the modal in index route:
```python
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

        scaler = MinMaxScaler()

        columns_to_normalize = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'Log_Transaction_Amount', 'Transaction_Amount_Difference',
            'Origin_Account_Balance_Difference', 'Destination_Account_Balance_Difference',
            'amount_mean_by_type', 'amount_median_by_type', 'amount_std_dev_by_type',
        ]


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
```

If you have any questions or need further information, feel free to reach out. We are happy to share more!

