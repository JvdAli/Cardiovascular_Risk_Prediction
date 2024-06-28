# Cardiovascular Risk Prediction
The goal of the project is to develop a tool for the early detection and prevention of CHD, addressing a significant public health concern using machine learning techniques.

## Table of Content
  * [Problem Statement](#problem-statement)
  * [Dataset](#dataset)
  * [Data Pipeline](#data-pipeline)
  * [Project Structure](#project-structure)
  * [Conclusion](#conclusion)
  
  
## Problem Statement
  A group of conditions affecting the heart and blood vessels is known as cardiovascular diseases. They consist of heart disease, which affects the blood vessels that 
  supply the heart muscle.The issue of coronary heart disease is a significant public health concern and early prediction of CHD risk is crucial for preventative 
  measures. We have to build a classification model to predict the 10-year risk of future coronary heart disease (CHD) for patients.
  
 
## Dataset
  The dataset is from an ongoing cardiovascular study on residents of Flamingham, Massachusetts. The data set includes over 3390 records and 17 attributes, each of which is a potential risk factor, including demographic, behavioral, and medical risk factors. For more information on the dataset, please visit the Kaggle website at https://www.kaggle.com/datasets/christofel04/cardiovascular-study-dataset-predict-heart-disea
  
  
## Data Pipeline
  1. Analyze Data: 
      In this initial step, we attempted to comprehend the data and searched for various available features. We looked for things like the shape of the data, the data types of each feature, a statistical summary, etc. at this stage.
  2. EDA:
     EDA stands for Exploratory Data Analysis. It is a process of analyzing and understanding the data. The goal of EDA is to gain insights into the data, identify patterns, and discover relationships and trends. It helps to identify outliers, missing values, and any other issues that may affect the analysis and modeling of the data.
  4. Data Cleaning: 
      Data cleaning is the process of identifying and correcting or removing inaccuracies, inconsistencies, and missing values in a dataset. We inspected the dataset 
      for duplicate values. The null value and outlier detection and treatment followed. For the imputation of the null value we used the Mean, Median, and Mode 
      techniques, and for the outliers, we used the Clipping method to handle the outliers without any loss to the data.
  5. Feature Selection: 
      At this step, we did the encoding of categorical features. We used the correlation coefficient, chi-square test, information gain, and an extra tree classifier       to select the most relevant features. SMOTE is used to address the class imbalance in the target variable.
  6. Model Training and Implementation:  
      We scaled the features to bring down all of the values to a similar range. We pass the features to 8 different classification models. We also did 
      hyperparameter tuning using RandomSearchCV and GridSearchCV.
  7. Performance Evaluation: 
      After passing it to various classification models and calculating the metrics, we choose a final model that can make better predictions. We evaluated different 
      performance metrics but choose our final model using the f1 score and recall score.
      
      
## Project Structure
```
├── README.md
├── Dataset 
├── Problem Statement
│
├── Know Your Data
│
├── Understanding Your Variables
│
├── EDA
│   ├── Numeric & Categorical features
│   ├── Univariate Analysis
│   ├── Bivariate and Multivariate Analysis
│
├── Data Cleaning
│   ├── Duplicated values
│   ├── Missing values
│   ├── Skewness
│   ├── Treating Outliers
│
├── Feature Engineering
│   ├── Encoding
|   ├── Feature Selection
|   ├── Extra Trees Classifier
|   ├── Handling Class Imbalance
│
├── Model Building
│   ├── Train Test Split
|   ├── Scaling Data
|   ├── Model Training
│
├── Model Implementation
│   ├── Logistic Regression
|   ├── Naive Bayes Classifier
|   ├── SVM
|   ├── Random Forest
│   ├── XGBoost
|   ├── KNN
```

## Conclusion
In this project, we tackled a classification problem in which we had to classify and predict the 10-year risk of future coronary heart disease (CHD) for patients. The goal of the project was to develop a tool for the early detection and prevention of CHD, addressing a significant public health concern using machine learning techniques.

- There were approximately 3390 records and 16 attributes in the dataset.
    - We started by importing the dataset, and necessary libraries and conducted exploratory data analysis (EDA) to get a clear insight into each feature by separating the dataset into numeric and categoric features. We did Univariate, Bivariate, and even multivariate analyses.
    - After that, the outliers and null values were removed from the raw data and treated. Data were transformed to ensure that it was compatible with machine 
    learning models.
    - In feature engineering we transformed raw data into a more useful and informative form, by creating new features, encoding, and understanding important 
    features. We handled target class imbalance using SMOTE.
    - Then finally cleaned and scaled data was sent to various models, the metrics were made to evaluate the model, and we tuned the hyperparameters to make sure the right parameters were being passed to the model. To select the final model based on requirements, we checked model_result.
    - When developing a machine learning model, it is generally recommended to track multiple metrics because each one highlights distinct aspects of model performance. We are, however, focusing more on the Recall score and F1 score because we are dealing with healthcare data and our data is unbalanced.

If we want to completely avoid any situations where the patient has heart disease, a high recall is desired. Whereas if we want to avoid treating a patient with no heart diseases a high precision is desired.

Assuming that in our case the patients who were incorrectly classified as suffering from heart disease are equally important since they could be indicative of some other ailment, so we want a balance between precision and recall and a high f1 score is desired.

Since we have added synthetic datapoints to handle the huge class imbalance in training set, the data distribution in train and test are different so the high performance of models in the train set is due to the train-test data distribution mismatch and not due to overfitting.

Best performance of Models on test data based on evaluation metrics for class 1:
  1. Recall - SVC
  2. Precision - Naive Bayes Classifier
  3. F1 Score - Logistic Regression, XGBoost
  4. Accuracy - Naive Bayes Classifier



