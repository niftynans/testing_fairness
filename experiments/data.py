import pandas as pd
import numpy as np 
from fairlearn.metrics import equalized_odds_difference

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo 

import sklearn 
import sklearn.cluster
from collections import Counter

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
        f_0 = 'unavailable'
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)  
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
        if f_0 == 'unavailable':
            strategy = ''
            if strategy == 'protected attribute bias':
                df['f_0'] = np.where(df['sex'] == 'Male', 1, 0)
                df['f_0'] = np.where(df['race'] == 'White', 1, 0)

            elif strategy == 'unsupervised clustering':
                X, y, _ = preprocess_data( df, target_col=target_col, categorical_cols=categorical_cols, scale_numerical=True)
                km = sklearn.cluster.KMeans(n_clusters=2)
                km.fit(X)
                labels = km.labels_
                df['f_0'] = labels
            
            elif strategy == 'random assignment':
                df['f_0'] = np.random.choice([0, 1], size=len(df))
            
            else:
                df['f_0'] = np.where(df['income'] == '>50K', 1, 0)
                print(df['f_0'])        
        eo_diff = equalized_odds_difference(
        y_true=df['Y'],
        y_pred=df['f_0'],
        sensitive_features=df['A']
        )
        print(f"Equalized Odds Difference: {eo_diff}")
    
    elif dataset == 'student':
        f_0 = 'unavailable'
        student_performance = fetch_ucirepo(id=320) 
        
        df = student_performance.data.features
        df = pd.concat([df, student_performance.data.targets], axis=1)
        
        print(df.columns) 
        
    elif dataset == 'law':
        f_0 = 'available'
        df = pd.read_csv('../data/law_school/law_school_clean.csv')
        print(df)

    else:
        f_0 = 'available'
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        columns = [
            "status", "duration", "credit history", "purpose", "amount", "savings", "employment", "installment rate", "personal status", 
            "debtors", "residence", "property", "age", "installment plans", "housing", "existing credits", "job", 
            "liable people", "telephone", "foreign worker", "credit risk"
        ]
        df = pd.read_csv(url, header=None, sep='\s+', names=columns)
        target_col="credit risk"
        categorical_cols=["status", "credit history", "purpose", "savings", "employment", "personal status", "debtors", "property", "installment plans", "housing", "job", "telephone", "foreign worker"]
        
    
    
    # X, y, _ = preprocess_data(
    #     df, target_col=target_col,
    #     categorical_cols=categorical_cols,
    #     scale_numerical=True
    # )
    # print(f"{dataset} dataset:")

    # naive_bayes_pipeline(X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, student, german, law", default='adult')
    args = parser.parse_args()

    main(args)