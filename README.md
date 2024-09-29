##  Heart Disease Prediction using Ensemble of Machine Learning Models
This project aims to predict the likelihood of an individual having heart disease using a sophisticated ensemble of machine learning models. The dataset includes patient information and various health parameters, and the goal is to build a model that can accurately predict whether or not a patient has heart disease.

##  Models Used in the Ensemble
- XGBoost: A powerful gradient-boosting algorithm that is particularly good for structured/tabular data.
- Random Forest: A robust ensemble method that averages multiple decision trees to improve accuracy and reduce overfitting.
- Logistic Regression: A simple, interpretable model that is well-suited for binary classification problems.
- LightGBM: Another gradient boosting model that is efficient and scalable, particularly for large datasets.

##  Project Workflow
1. Data Preprocessing
- Missing Values: Any missing values in the dataset are imputed with the median of each column.
- Feature Scaling: Features are standardized using StandardScaler to ensure that the models perform optimally.
- Class Imbalance Handling: The dataset is resampled using SMOTE (Synthetic Minority Over-sampling Technique) to handle any class imbalance in the target variable.
2. Ensemble Model
The ensemble model is created using VotingClassifier with four base models:

- XGBoost
- Random Forest
- Logistic Regression
- LightGBM
The ensemble uses soft voting, meaning it averages the predicted probabilities of each model to make the final prediction.

3. Model Evaluation
The model is evaluated on the validation set using the following metrics:
- Accuracy: Percentage of correctly classified instances.
- AUC-ROC: Measures the model's ability to distinguish between positive and negative classes.
- Classification Report: Provides precision, recall, and F1-score for both classes.

4. Test Predictions and Submission
The trained model is used to predict outcomes on the test dataset, and the predictions are saved in a CSV file for submission.

##  Project Structure
```heart-disease-prediction/
│
├── data/
│   ├── Train_Dataset.csv          # Training data
│   ├── Test_Dataset.csv           # Test data
│   ├── Sample_Submission.csv      # Sample submission format
│   ├── Variable_Definitions.csv   # Definitions of variables used in the dataset
│
├── notebooks/
│   └── heart_disease_prediction.ipynb    # Jupyter notebook containing the model code
│
├── Ensemble_Heart_Disease_Predictions_LightGBM.csv  # Final submission file with predictions
│
├── README.md  # Project documentation
