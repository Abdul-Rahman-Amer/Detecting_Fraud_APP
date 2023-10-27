# Detecting_Fraud_APP
Deploying a Neural Network in a flask application to take in input and detect fraudulent activity. 

## Step 1 - Understanding the data
### Right from the start, there is a huge minority class in the data. The following led me to want to SMOTE.

### SMOTE:
**SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE stands for "Synthetic Minority Over-sampling Technique." It is a technique used in the field of machine learning and data analysis, particularly in the context of imbalanced datasets. An imbalanced dataset is one in which the number of instances (or data points) in one class is significantly lower than the number of instances in another class. This can lead to problems in machine learning models, as they may be biased towards the majority class and perform poorly on the minority class.

**How SMOTE Works:**
![Image Alt Text](/photos_for_readme/smote.png)
1. For each minority class instance, SMOTE selects *k* nearest neighbors from the same class as seen above 👆 

2. It then selects a random neighbor from the *k* nearest neighbors and creates a synthetic example by interpolating between the selected instance and the randomly chosen neighbor. This interpolation is done in feature space, creating a new data point that lies along the line connecting the two selected instances.

3. This process is repeated for a specified number of synthetic examples to be generated, effectively increasing the number of minority class instances in the dataset.

SMOTE helps balance the class distribution in the dataset, which can improve the performance of machine learning models, especially when dealing with tasks like classification where class imbalance is a common problem. By generating synthetic examples, SMOTE ensures that the minority class is adequately represented in the training data, leading to better model generalization and accuracy.
