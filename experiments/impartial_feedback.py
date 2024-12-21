import pandas as pd
import numpy as np
import random 
from fairlearn.metrics import equalized_odds_difference

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo 

import sklearn 
import sklearn.cluster
from collections import Counter

import argparse
import os
import sys



def preprocess_data(df, target_col, categorical_cols, scale_numerical=False, dataset = None, from_NB = None):
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df.drop(columns=[target_col])
    if from_NB:
        y = df['f_0']
    else:
        y = df[target_col]

    if scale_numerical:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y, label_encoders



def naive_bayes_pipeline(df, target_col, categorical_cols, dataset):
    X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model



def get_f_0(policy, dataset, df, target_col = None, categorical_cols = None): 
    X, y, _ = preprocess_data( df, target_col=target_col, categorical_cols=categorical_cols, scale_numerical=True)

    if policy == 'bias':
        if dataset == 'adult':
            feat_ind = np.array([8, 9, 13, 1])  # Protected Attribute Set
            f_0_str = ''                        # String to output the notational version of f_0
            thresh = random.randint(5, 50)
            coeffs, exps = [], []
            for i in range(len(feat_ind)):
                if i == len(feat_ind) - 1:
                    a = random.randint(0, 4)
                    x = random.randint(0, 10)
                    f_0_str += " " + str(x) + " * (" + str(df.columns[feat_ind[i]]) + ") ^ " + str(a) + "  "
                else:
                    a = random.randint(0, 4)
                    x = random.randint(0, 10)
                    f_0_str += " " + str(x) + " * (" + str(df.columns[feat_ind[i]]) + ") ^ " + str(a) + " + "
                exps.append(a)
                coeffs.append(x)
            f_0_str += ' >  ' + str(thresh)
            print(f_0_str)
        df['f_0'] = np.where(np.sum([coeffs[i] * df[df.columns[feat_ind[i]]] ** exps[i] for i in range(len(feat_ind))], axis=0) > thresh, 1,0)
        print(Counter(df['f_0']))

    elif policy == 'cluster':
        km = sklearn.cluster.KMeans(n_clusters=2)
        km.fit(X)
        labels = km.labels_
        df['f_0'] = labels
    
    elif policy == 'random':
        df['f_0'] = np.random.choice([0, 1], size=len(df))
    
    else:
        if dataset == 'adult':
            df['f_0'] = np.where(df[target_col] == '>50K', 1, 0)
        elif dataset == 'law':
            df['f_0'] = np.where(df[target_col] >= 10, 1, 0)            
    
    return df

def get_eo_diff(df, dataset, A, alpha):
    if dataset == 'adult':
        Y = np.where(df['income'] == '>50K', 1, 0)
    elif dataset == 'student':
        Y = np.where(df['G3'] >= 10, 1, 0)
    eo_diff = equalized_odds_difference(
        y_true=Y,
        y_pred=df['f_0'],
        sensitive_features=df[A]
        )

    print(f"H0 : eo_diff = 0  |  H1 : eo_diff > {alpha}")
    print(f"Equalized Odds Difference: {eo_diff}")
    
    if eo_diff == 0: # less than epsilon 1e-3, alpha = 0.1 or something
        print("H0 validated.")
    elif eo_diff > alpha:
        print("H1 validated.")


def main(args):
    dataset = args.dataset
    policy = args.policy
    alpha = float(args.alpha)
    
    
    if dataset == 'adult':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [ "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)  
        
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
        df = get_f_0(policy, dataset, df, target_col=target_col, categorical_cols=categorical_cols)        
        A = 'sex' # protected attribute
        get_eo_diff(df, dataset, A, alpha) # tells whether f_0 is fair or not.
        
        
    if dataset == 'law':
        df = pd.read_csv('../data/law_school/law_school_clean.csv')
        target_col = "pass_bar"
        categorical_cols = ["fam_inc", "tier", "race"]
        df['f_0'] = target_col
        model = naive_bayes_pipeline(df, "f_0", categorical_cols, dataset)
        print(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Testing Fairness.')
    parser.add_argument('--dataset', type=str, help="Name of dataset: adult, law", default='adult')
    parser.add_argument('--policy', type=str, help="Name of audit policy: bias, cluster, random, outcome", default='bias')
    parser.add_argument('--alpha', type=str, help="Alternate Hypothesis Constant, alpha", default='0.10')

    args = parser.parse_args()

    main(args)