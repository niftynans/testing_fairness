# def get_f_0(policy, dataset, df, target_col = None, categorical_cols = None): 
#     f_0_str = ''
#     X, y, _ = preprocess_data( df, target_col=target_col, categorical_cols=categorical_cols, scale_numerical=True)
    
#     if policy == 'bias':
#         if dataset == 'adult':
#             feat_ind = np.array([8, 9])  
#             f_0_str = ''                        
#             thresh = random.randint(5, 50)
#             coeffs, exps = [], []
#             for i in range(len(feat_ind)):
#                 if i == len(feat_ind) - 1:
#                     a = random.randint(0, 4)
#                     x = random.randint(0, 10)
#                     f_0_str += " " + str(x) + " * (" + str(df.columns[feat_ind[i]]) + ") ^ " + str(a) + "  "
#                 else:
#                     a = random.randint(0, 4)
#                     x = random.randint(0, 10)
#                     f_0_str += " " + str(x) + " * (" + str(df.columns[feat_ind[i]]) + ") ^ " + str(a) + " + "
#                 exps.append(a)
#                 coeffs.append(x)
#             f_0_str += ' >  ' + str(thresh)
#         df['f_0'] = np.where(np.sum([coeffs[i] * df[df.columns[feat_ind[i]]] ** exps[i] for i in range(len(feat_ind))], axis=0) > thresh, 1,0)

#     elif policy == 'learn':
#         if dataset == 'adult':
#             # sample_type = 'rows'
#             sample_type = 'columns'
#             if sample_type == 'rows':
#                 f_0_str = 'Learnt from Subset of Rows'
#                 df_sub = df.sample(n=int(len(df)/10))
#                 model = naive_bayes_pipeline(df_sub, target_col, categorical_cols, dataset, from_NB = False)
#                 X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=categorical_cols, dataset=dataset, from_NB=False)
#                 df['f_0'] = model.predict(X)
    
#             elif sample_type == 'columns':
#                 f_0_str = 'Learnt from Subset of Columns'
#                 df_sub = df[['sex', target_col]]
#                 print(Counter(df[target_col]))
#                 cat_cols = ['sex']
#                 df_sub = df.sample(n=int(len(df)/10))
#                 model = naive_bayes_pipeline(df_sub, target_col, cat_cols, dataset, from_NB = False)
#                 X, y, _ = preprocess_data(df, target_col=target_col, categorical_cols=cat_cols, dataset=dataset, from_NB=False)
#                 df['f_0'] = model.predict(X)
#                 print(Counter(df['f_0']))
    
#     elif policy == 'cluster':
#         km = sklearn.cluster.KMeans(n_clusters=2)
#         km.fit(X)
#         labels = km.labels_
#         df['f_0'] = labels
    
#     elif policy == 'random':
#         df['f_0'] = np.random.choice([0, 1], size=len(df))
    
#     else:
#         if dataset == 'adult':
#             df['f_0'] = np.where(df[target_col] == '>50K', 1, 0)
#         elif dataset == 'law':
#             df['f_0'] = np.where(df[target_col] >= 10, 1, 0)            
    
#     return df, f_0_str



# print(f"H0 : eo_diff = 0  |  H1 : eo_diff > {alpha}")
# print(f"Equalized Odds Difference: {eo_diff_0}")
# if eo_diff_0 == 0: # less than epsilon 1e-3, alpha = 0.1 or something
#     print("H0 validated.")
# elif eo_diff_0 > alpha:
#     print("H1 validated.")
# print(eo_diff_0, eo_diff_1)