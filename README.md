# Detecting_Fraud_APP
Deploying a Neural Network in a flask application to take in input and detect fraudulent activity. 

## Step 1 - Understanding the data
### Right from the start, there is a huge minority class in the data. The following led me to want to SMOTE.

### SMOTE:
**(Synthetic Minority Over-sampling Technique)**

SMOTE stands for "Synthetic Minority Over-sampling Technique." It is a technique used in the field of machine learning and data analysis, particularly in the context of imbalanced datasets. An imbalanced dataset is one in which the number of instances (or data points) in one class is significantly lower than the number of instances in another class. This can lead to problems in machine learning models, as they may be biased towards the majority class and perform poorly on the minority class.

**How SMOTE Works:**

![Image Alt Text](/photos_for_readme/smote.png)
1. For each minority class instance, SMOTE selects *k* nearest neighbors from the same class as seen above 👆 

2. It then selects a random neighbor from the *k* nearest neighbors and creates a synthetic example by interpolating between the selected instance and the randomly chosen neighbor. This interpolation is done in feature space, creating a new data point that lies along the line connecting the two selected instances.

3. This process is repeated for a specified number of synthetic examples to be generated, effectively increasing the number of minority class instances in the dataset.

SMOTE helps balance the class distribution in the dataset, which can improve the performance of machine learning models, especially when dealing with tasks like classification where class imbalance is a common problem. By generating synthetic examples, SMOTE ensures that the minority class is adequately represented in the training data, leading to better model generalization and accuracy.

### PCA 
** (Principal Component Analysis)**

PCA stands for "Principal Component Analysis." It is a technique used in the field of data analysis and dimensionality reduction. PCA is particularly valuable for reducing the complexity of high-dimensional datasets while preserving the most important information.

**How PCA Works:**

1. **Data Centering:** PCA begins by centering the data, which involves subtracting the mean from each feature. This step ensures that the first principal component (the direction of maximum variance) passes through the data's center.

2. **Covariance Matrix:** PCA then constructs the covariance matrix from the centered data. The elements of this matrix capture the relationships between different features.

3. **Eigenvalue Decomposition:** The next step involves finding the eigenvalues and eigenvectors of the covariance matrix. These eigenvalues represent the variances of the data along the corresponding eigenvectors.

4. **Principal Component Selection:** Principal components are selected based on their associated eigenvalues, with higher eigenvalues indicating more significant variance. These components capture the most critical information in the data.

5. **Data Projection:** Finally, data is projected onto the selected principal components, reducing the dimensionality while retaining as much information as possible.

PCA is a powerful tool for data analysis, dimensionality reduction, and feature extraction, making it valuable in various fields, including machine learning and statistics. See how helpful it was in feature selection:

![Image Alt Text](/photos_for_readme/PCA_Visualization_1.png)

## The above was a clear takeaway that the "step" feature in the data wasn't providing a strong signal. 

## To go more in depth I created a Heatmap and Precision Matrix:
### After seeing the below I was inspired to go a bit more in depth to analyze the feautres inner relationships and the results were fascinating! 
![Image Alt Text](/photos_for_readme/PCA_Visualization_1_Heatmap.png) ![Image Alt Text](/photos_for_readme/Precision_Matrix.png)

### Looking at how different features as the Principal Component rely on others opened up my mind to see howdeeply intertwined these features really are. 
![Image Alt Text](/photos_for_readme/Princiapl_component.png) ![Image Alt Text](/photos_for_readme/Principal_Component_2.png) 

## At this point, it was time to start testing. I was already excited. I used several models:

### 1. Bayesian Neural Network
- **Bayesian Neural Network (BNN)** is a type of neural network that extends traditional neural networks with probabilistic modeling. Unlike conventional neural networks that provide point estimates, BNNs provide probability distributions over their predictions. They are useful when you need to quantify uncertainty in your predictions. BNNs use Bayesian inference to update the model's beliefs as new data is observed.

### 2. Logistic Regression
- **Logistic Regression** is a classic statistical model used for binary classification tasks, such as fraud detection. It models the probability of a binary outcome (e.g., fraud or not fraud) based on input features. Logistic regression provides interpretable coefficients and is a good choice for understanding the importance of different features in your fraud detection model.

### 3. Decision Tree
- **Decision Tree** is a tree-based machine learning model that is often used for classification tasks. In the context of fraud detection, a decision tree can help you understand the decision-making process of the model, as it creates a tree-like structure where each node represents a feature and each leaf node represents a class label (fraud or not fraud).

### 4. MLP (Multi-Layer Perceptron)
- **Multi-Layer Perceptron (MLP)** is a type of neural network with multiple layers of interconnected nodes (neurons). It is a powerful model for capturing complex patterns in data. In fraud detection, MLP can learn intricate relationships between various features and is capable of modeling non-linear decision boundaries.

### 5. Hybrid Model (Decision Tree and MLP)
- The **Hybrid Model** you used combines the results from a Decision Tree and an MLP. This approach can boost the recall score by leveraging the strengths of both models. The Decision Tree may excel in explaining individual decisions, while the MLP can capture more complex patterns.

Each of these models has its own strengths and weaknesses, and testing multiple models to decide which ones I wanted to follow through with was my primary goal. 

See Below:
(Note that NN is our MLP model)

![Image Alt Text](/photos_for_readme/Bayesian_MAtrix.png) ![Image Alt Text](/photos_for_readme/Decision_Tree_Matrix.png) ![Image Alt Text](/photos_for_readme/Logistic_Regression_Matrix.png) 
![Image Alt Text](/photos_for_readme/MLP_Matrtix.png) ![Image Alt Text](/photos_for_readme/Hybrid_Matrix.png) 

## From this Point I was interested in our hybrid model but later I would be impressed that after feature engineering, the MLP was out performing and it had a high recall score over 99.5%

# When reviewing the final notebook "going in more depth" you will see that only arounf 10 "Fraud" instances Were categorized as "Not Fraud" With the rest being properly identified. I pickled the model and hosted it inside of a UI Built in flask. 

## Here Takes User Inputs:
![Image Alt Text](/photos_for_readme/APP_UI_2.png) 

## Then I use Javascript and Ajax to handle the user request and serve it into the flask route /perdict which converts the input into a tensor that our pickled model can interpret and predict. 

![Image Alt Text](/photos_for_readme/APP_UI.png) 

# If you have any questions please reach out anytime. I am happy to share more! 
