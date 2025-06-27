# Titanic Survival Prediction using Logistic Regression (From Scratch)

This project implements logistic regression from scratch in Python to predict whether a passenger survived the Titanic disaster, using a cleaned dataset with selected features.

## Project Overview

- Implements logistic regression manually using NumPy (no external ML libraries)
- Trains and evaluates on a Titanic dataset
- Includes preprocessing, training, evaluation, and prediction steps
- Predicts survival based on user-defined input

## Dataset

The dataset (`SVMtrain.csv`) is a simplified version of the Titanic dataset. It contains the following columns:

- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: male or female
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Fare paid
- `Embarked`: Port of embarkation (1, 2, or 3)

The target variable is `Survived` (0 = did not survive, 1 = survived).

## Features Used

The model uses the following input features:

- Pclass
- Sex (encoded as 0 or 1)
- Age
- SibSp
- Parch
- Fare
- Embarked (encoded as 0, 1, 2)

The target label is:
- Survived (0 or 1)

## How It Works

1. **Data Preprocessing**
   - Encodes categorical features (Sex and Embarked)
   - Handles missing values in Age and Embarked
   - Drops unnecessary columns like PassengerId

2. **Train/Test Split**
   - 80% training set, 20% testing set
   - Bias term is added to feature vectors

3. **Logistic Regression Implementation**
   - Sigmoid function for probability estimation
   - Gradient descent for cost minimization
   - Cost function is logged and plotted

4. **Model Evaluation**
   - Accuracy on the test set is reported

5. **User Prediction**
   - Hardcoded example simulates a passenger profile
   - Predicts survival probability and outputs class prediction

## How to Run

1. Make sure the following files are in the same folder:
   - `titanic.py` (or your script file)
   - `SVMtrain.csv`

2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
