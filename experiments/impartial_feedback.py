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



def get_f_0(policy, dataset, df, target_col = None, categorical_cols = None): 
    f_0_str = ''
    X, y, _ = preprocess_data( df, target_col=target_col, categorical_cols=categorical_cols, scale_numerical=True)
    
    if policy == 'bias':
        if dataset == 'adult':
            feat_ind = np.array([8, 9, 13, 1])  
            f_0_str = ''                        
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
        df['f_0'] = np.where(np.sum([coeffs[i] * df[df.columns[feat_ind[i]]] ** exps[i] for i in range(len(feat_ind))], axis=0) > thresh, 1,0)

    elif policy == 'learn':
        if dataset == 'adult':
            # sample_type = 'rows'
            sample_type = 'columns'
            if sample_type == 'rows':
                f_0_str = 'Learnt from Subset of Rows'
                df_sub = df.sample(n=int(len(df)/10))
                model = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
                X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
                df['f_0'] = model.predict(X)
    
            elif sample_type == 'columns':
                f_0_str = 'Learnt from Subset of Columns'
                df_sub = df[['sex', target_col]]
                print(Counter(df_sub[target_col]))
                cat_cols = ['sex']
                model = naive_bayes_pipeline(df_sub, target_col, cat_cols, dataset, from_NB = False)
                X, y, _ = preprocess_data(df_sub, target_col=target_col, categorical_cols=cat_cols, dataset=dataset, from_NB=False)
                print(X.shape, y.shape)
                df['f_0'] = model.predict(X)
                print(Counter(df['f_0']))
                sys.exit()
    
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
    
    return df, f_0_str

def get_eo_diff(df, dataset, A, alpha):
    if dataset == 'adult':
        print(df.columns)
        sys.exit()
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
    # print(f"H0 : eo_diff = 0  |  H1 : eo_diff > {alpha}")
    # print(f"Equalized Odds Difference: {eo_diff_0}")
    # if eo_diff_0 == 0: # less than epsilon 1e-3, alpha = 0.1 or something
    #     print("H0 validated.")
    # elif eo_diff_0 > alpha:
    #     print("H1 validated.")
    return eo_diff_0, eo_diff_1


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
    
        A = 'sex' # protected attribute

        counts = Counter(df[A])
        count_male = counts['Male']
        count_female = counts['Female']
        ratio = count_female / count_male

        file_exists = os.path.exists('../results/adult_eo_vals.csv')
        with open("../results/adult_eo_vals.csv", mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists or os.path.getsize('../results/adult_eo_vals.csv') == 0:
                    writer.writerow(['f_0', 'm_val', 'eo_val_0', 'eo_val_1'])    
                for i in range(1):
                    df, f_0_str = get_f_0(policy, dataset, df, target_col=target_col, categorical_cols=categorical_cols)        
                    
                    for m in range(100, 3100, 20):
                        condition = (df[A] == 1)
                        sampled_rows_male = df[condition].sample(n=int(m * (1-ratio)), random_state=42)
                        condition = (df[A] == 0)
                        sampled_rows_female = df[condition].sample(n=int(m  * ratio), random_state=42)
                        concatenated_samples = pd.concat([sampled_rows_male, sampled_rows_female], ignore_index=True)
                        eo_diff_0, eo_diff_1 = get_eo_diff(concatenated_samples, dataset, A, alpha) 
                        writer.writerow([f_0_str, m, eo_diff_0, eo_diff_1])
                    eo_diff_0, eo_diff_1 = get_eo_diff(df, dataset, A, alpha) 
                    writer.writerow([f_0_str, m, eo_diff_0, eo_diff_1])
                    
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
    parser.add_argument('--policy', type=str, help="Name of audit policy: bias, cluster, random, outcome", default='learn')
    parser.add_argument('--alpha', type=str, help="Alternate Hypothesis Constant, alpha", default='0.10')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    main(args)