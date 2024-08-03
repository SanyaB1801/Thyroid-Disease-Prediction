# Thyroid Disease Prediction

## Description

This project aims to develop and evaluate a machine learning model to predict thyroid disease using patient data. The dataset includes various features relevant to thyroid conditions, and the workflow covers data preprocessing, model training, evaluation, and visualization. The model helps in early diagnosis and effective treatment planning for thyroid disorders.

## Features

- **Data Loading and Preparation**: Load and prepare the dataset from a CSV file, including handling missing values and converting date columns (if applicable).
- **Data Preprocessing**: Standardize features, split data into training and testing sets, and perform feature scaling.
- **Model Training**: Train a HistGradientBoostingClassifier on the preprocessed data to classify thyroid conditions.
- **Performance Evaluation**: Assess the model using classification metrics and visualize performance with a confusion matrix heatmap.
- **Visualization**: Create visualizations such as confusion matrix heatmaps to understand model performance and diagnostic accuracy.

## Use Cases

- **Early Diagnosis**: Assist healthcare professionals in diagnosing thyroid disorders early based on patient data.
- **Risk Assessment**: Evaluate the likelihood of thyroid disease in patients with symptoms or relevant medical history.
- **Personalized Treatment**: Help in tailoring treatment plans based on predicted thyroid conditions.
- **Monitoring and Management**: Track disease progression and adjust treatment plans as needed for ongoing management.

## Benefits

- **Accurate Predictions**: Provides reliable predictions to support early diagnosis and effective treatment.
- **Efficient Workflow**: Streamlines the detection and classification of thyroid disease using machine learning.
- **Improved Patient Care**: Enhances decision-making with data-driven insights for better healthcare outcomes.
- **Ongoing Monitoring**: Facilitates regular assessment of patient data to manage thyroid conditions effectively.

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

## Project Components

### Dataset
- **thyroid-disease.data**: The dataset containing patient data used for training and evaluating the model.
- **Dataset Source**: [UCI Machine Learning Repository - Thyroid Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data)

### Notebook
- **thyroid_prediction.ipynb**: A Jupyter notebook containing the full workflow for data preprocessing, model training, evaluation, and saving the model.

### Models
- **thyroid_model.pkl**: The trained model for predicting thyroid disease.

### Documentation
- **README.md**: Project overview, setup instructions, and usage guide.

## Workflow

1. **Data Loading and Preparation**: Load data from `thyroid-disease.data` and preprocess it, including handling missing values and standardizing features.
2. **Data Preprocessing**: Split the dataset into training and testing sets and scale features.
3. **Model Training**: Train a HistGradientBoostingClassifier to classify thyroid conditions.
4. **Performance Evaluation**: Evaluate the model using classification metrics and visualize performance with a confusion matrix heatmap.
5. **Model Saving**: Save the trained model using Joblib for future use.

## Example Analysis

- **Dataset**: Thyroid disease dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data)
- **Model**: HistGradientBoostingClassifier
- **Performance Metrics**: Precision, recall, F1-score
- **Visualizations**: Confusion matrix heatmap

## Screenshots

![image](https://github.com/user-attachments/assets/42fbf800-f84e-4bb8-8c77-0ddb085782e7)


![image](https://github.com/user-attachments/assets/9d96e27e-165c-41ae-a131-3b602e8ee4b0)

