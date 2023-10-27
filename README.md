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

### 2. Logistic Regression
- Logistic Regression is a classic model for binary classification. It models the probability of a binary outcome based on input features, providing interpretable coefficients.

### 3. Decision Tree
- Decision Tree is a tree-based model that helps understand the decision-making process by creating a tree-like structure based on features.

### 4. Multi-Layer Perceptron (MLP)
- MLP is a neural network capable of capturing complex patterns and non-linear decision boundaries, well-suited for fraud detection.

### 5. Hybrid Model (Decision Tree and MLP)
- The Hybrid Model combines the strengths of Decision Tree and MLP, boosting recall scores.

![Model Matrices](/photos_for_readme/Model_Matrices.png)

In-depth testing revealed the best model for the fraud detection task.

## Model Deployment

The MLP model outperformed others, achieving a high recall score. This model was pickled and hosted in a Flask application.

### User Input

The Flask application offers a user-friendly interface where users can input data for fraud detection.

![App UI](/photos_for_readme/APP_UI.png)

### Model Prediction

JavaScript and Ajax are used to handle user requests, serving them to the Flask route `/predict`. The input is converted into a tensor for prediction by the pickled model.

![Modal Prediction](/photos_for_readme/Modal_Prediction.png)

If you have any questions or need further information, feel free to reach out. We are happy to share more!

