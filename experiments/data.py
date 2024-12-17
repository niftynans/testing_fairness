import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo 

import argparse
import os
import sys

def preprocess_data(df, target_col, categorical_cols, scale_numerical=False):
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if scale_numerical:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, label_encoders

def naive_bayes_pipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def main(args):
    dataset = args.dataset
    if dataset == 'adult':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)  
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    elif dataset == 'student':
        student_performance = fetch_ucirepo(id=320) 
        
        df = student_performance.data.features
        df = pd.concat([df, student_performance.data.targets], axis=1)
        
        print(df) 

    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        columns = [
            "status", "duration", "credit history", "purpose", "amount", "savings", "employment", "installment rate", "personal status", 
            "debtors", "residence", "property", "age", "installment plans", "housing", "existing credits", "job", 
            "liable people", "telephone", "foreign worker", "credit risk"
        ]
        df = pd.read_csv(url, header=None, sep='\s+', names=columns)
        target_col="credit risk"
        categorical_cols=["status", "credit history", "purpose", "savings", "employment", "personal status", "debtors", "property", "installment plans", "housing", "job", "telephone", "foreign worker"]
        
    print(df.head())
    sys.exit()
    X, y, _ = preprocess_data(
        df, target_col=target_col,
        categorical_cols=categorical_cols,
        scale_numerical=True
    )
    print(f"{dataset} dataset:")
    naive_bayes_pipeline(X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, student, german", default='adult')
    args = parser.parse_args()

    main(args)