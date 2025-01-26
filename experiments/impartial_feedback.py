import pandas as pd
import numpy as np
import random 
import time

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
import csv
import warnings

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



def naive_bayes_pipeline(df, target_col, categorical_cols, dataset, from_NB = True):
    X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=from_NB)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    return model

def get_f_0(policy, dataset, df, protected_attribute, target_col = None, categorical_cols = None): 

    if policy == 'biased':
        df_sub = df.sample(n=int(len(df)/10))
        model = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
        X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
        df['f_0'] = model.predict(X)
        df[protected_attribute] = np.where(df[protected_attribute] == 1, 'Male', 'Female')
    
    elif policy == 'unbiased':
        df_sub = df.sample(n=int(len(df)/10))
        df_sub = df_sub.drop(protected_attribute, axis=1)
        categorical_cols.remove(protected_attribute)
        model = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
        df_sub = df.drop(protected_attribute, axis=1)
        X, y, _ = preprocess_data(df_sub, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
        df['f_0'] = model.predict(X)

    elif policy == 'random':
        df['f_0'] = np.random.choice(['>50K', '<=50K'], size=len(df))

    else:
        df['f_0' ]= np.where(df[protected_attribute] == 'Male', '>50K', '<=50K')

    return df

def get_eo_diff(df, dataset, A, alpha):
    if dataset == 'adult':
        Y = np.where(df['income'] == '>50K', 1, 0)
        f_0 = np.where(df['f_0'] == '>50K', 1, 0)

        ind0 = np.where(Y == 0)[0]  
        ind1 = np.where(Y == 1)[0]  

        Y_ind0 = Y[ind0]             
        Y_ind1 = Y[ind1]             
        f_0_ind0 = f_0[ind0]         
        f_0_ind1 = f_0[ind1]         

        eo_diff_0 = equalized_odds_difference(
            y_true=Y_ind0,
            y_pred=f_0_ind0,
            sensitive_features=df[A].iloc[ind0]
        )
        eo_diff_1 = equalized_odds_difference(
            y_true=Y_ind1,
            y_pred=f_0_ind1,
            sensitive_features=df[A].iloc[ind1]
        )
        
        eo_diff = equalized_odds_difference(
            y_true=Y,
            y_pred=f_0,
            sensitive_features=df[A]
        )
    return eo_diff_0, eo_diff_1, eo_diff


def main(args):
    # Set the args
    dataset = args.dataset
    policy = args.policy
    alpha = float(args.alpha)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Load Datasets
    if dataset == 'adult':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [ "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
        df.dropna(inplace=True)  
        
        target_col = "income"
        categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
        A = 'sex' # protected attribute

        counts = Counter(df[A])
        count_male = counts['Male']
        count_female = counts['Female']
        ratio = count_female / count_male
        
        try:
            os.makedirs(f"../results/{dataset}/{policy}") 
        except:
            pass
        
        file_exists = os.path.exists(f"../results/{dataset}/{policy}/{timestr}.csv")
        with open(f"../results/{dataset}/{policy}/{timestr}.csv", mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists or os.path.getsize(f"../results/{dataset}/{policy}/{timestr}.csv") == 0:
                    writer.writerow(['dataset', 'f_0', 'm_val', 'eo_val_0', 'eo_val_1', 'overall_eo_val'])    
                df = get_f_0(policy, dataset, df, A, target_col=target_col, categorical_cols=categorical_cols)        
                for m in range(50, int(len(df)/10), 50):
                    condition = (df[A] == 'Male')
                    sampled_rows_male = df[condition].sample(n=int(m * (1-ratio)), random_state=42)
                    condition = (df[A] == 'Female')
                    sampled_rows_female = df[condition].sample(n=int(m  * ratio), random_state=42)
                    concatenated_samples = pd.concat([sampled_rows_male, sampled_rows_female], ignore_index=True)
                    eo_diff_0, eo_diff_1, eo_diff = get_eo_diff(concatenated_samples, dataset, A, alpha) 
                    writer.writerow([dataset, policy, m, eo_diff_0, eo_diff_1, eo_diff])
                eo_diff_0, eo_diff_1, eo_diff = get_eo_diff(df, dataset, A, alpha) 
                writer.writerow([dataset, policy, m, eo_diff_0, eo_diff_1, eo_diff])
        
        
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
    parser.add_argument('--policy', type=str, help="Name of audit policy: biased, unbiased, random, protected", default='biased')
    parser.add_argument('--alpha', type=str, help="Alternate Hypothesis Constant, alpha", default='0.10')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    main(args)