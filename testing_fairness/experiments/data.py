import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

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
        adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]
        adult_df = pd.read_csv(adult_url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        adult_df.dropna(inplace=True)  
        
        X_adult, y_adult, _ = preprocess_data(
            adult_df, target_col="income",
            categorical_cols=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"],
            scale_numerical=True
        )
        print("\nAdult Dataset:")
        naive_bayes_pipeline(X_adult, y_adult)

    elif dataset == 'census':

        census_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income/census-income.data"
        census_columns = [
            "age", "class of worker", "industry code", "occupation code", "education", "wage per hour", "enroll in edu inst", 
            "marital status", "major industry", "major occupation", "race", "hispanic origin", "sex", "member of union", 
            "reason for unemployment", "employment status", "capital gains", "capital losses", "stock dividends", 
            "tax filer status", "region", "state", "household family", "household summary", "instance weight", "migration", 
            "num person worked", "under 18", "birth country", "citizenship", "owner/renter", "veteran admin", 
            "veterans benefits", "weeks worked", "year", "income"
        ]
        census_df = pd.read_csv(census_url, header=None, names=census_columns, na_values=" ?", skipinitialspace=True)
        census_df.dropna(inplace=True)

        X_census, y_census, _ = preprocess_data(
            census_df, target_col="income",
            categorical_cols=["class of worker", "education", "marital status", "major industry", "major occupation", "race", "hispanic origin", "sex", "tax filer status", "region", "state", "birth country", "citizenship"],
            scale_numerical=True
        )
        print("\nCensus Dataset:")
        naive_bayes_pipeline(X_census, y_census)

    else:
        german_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        german_columns = [
            "status", "duration", "credit history", "purpose", "amount", "savings", "employment", "installment rate", "personal status", 
            "debtors", "residence", "property", "age", "installment plans", "housing", "existing credits", "job", 
            "liable people", "telephone", "foreign worker", "credit risk"
        ]
        german_df = pd.read_csv(german_url, header=None, delim_whitespace=True, names=german_columns)

        X_german, y_german, _ = preprocess_data(
            german_df, target_col="credit risk",
            categorical_cols=["status", "credit history", "purpose", "savings", "employment", "personal status", "debtors", "property", "installment plans", "housing", "job", "telephone", "foreign worker"],
            scale_numerical=True
        )
        print("\nGerman Credit Dataset:")
        naive_bayes_pipeline(X_german, y_german)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, census, german", default='adult')
    args = parser.parse_args()

    main(args)