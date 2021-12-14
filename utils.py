import numpy as np
import pandas as pd
import pickle



columns = ['Ncodons', 'UUU', 'UUC', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG',
       'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA', 'GUG', 'GCU', 'GCC',
       'GCA', 'GCG', 'CCU', 'CCC', 'CCA', 'CCG', 'UGG', 'GGU', 'GGC', 'GGA',
       'GGG', 'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC', 'ACU', 'ACC', 'ACA',
       'ACG', 'UAU', 'UAC', 'CAA', 'CAG', 'AAU', 'AAC', 'UGU', 'UGC', 'CAU',
       'CAC', 'AAA', 'AAG', 'CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'GAU',
       'GAC', 'GAA', 'GAG', 'UAA', 'UAG', 'UGA']




def make_prediction_kingdom(X, clf_path, class_encoding_path, std_path, good_features_path ):   
    X = pd.DataFrame(X.values.reshape(1, -1), columns = X.index, dtype = float)
    
    #load class encoding
    with open(class_encoding_path, 'rb') as file:
        le = pickle.load(file)
    file.close()
    
    # load std_path file
    with open(std_path, 'rb') as file:
        std_ = pickle.load(file)
    file.close()
    
    # load classifier     
    with open(clf_path, 'rb') as file:
        clf = pickle.load(file)
    file.close()
    

    with open(good_features_path, 'rb') as file:
        good_features = pickle.load(file)
    file.close()
    
    
    # function for calculating double feature
    def get_XX_feature(val, f):
        temp = 0
        for col in val.columns:
            if col[0:2]==f or col[-1:-3]==f or (col[0]==f[0] and col[-1]==f[0]):
                try: 
                    temp = temp + float(val.iloc[0][col])
                except:
                    print(f'There has been a error while calculating {f}')
        return temp
    
    # function for calculating single occurance feature
    def get_X_feature(val, f):
        temp = 0
        for col in val.columns:
            if f in col and len(col)==3:
                try:
                    temp = temp + float(val.iloc[0][col])
                except:
                    print(f'There has been a error while calculating {f}')
        return temp
    X_copy = X.copy()
    kurt = X.kurtosis(axis = 1).values[0]
    med = X.median(axis = 1).values[0]
    mode = X.mode(axis = 1).values[0][0]
    var = X.var(axis = 1).values[0]
    max_ = X.max(axis = 1).values[0]
    min_ = X.min(axis = 1).values[0]
    q1 = X.quantile(0.25, axis = 1).values[0]
    q2 = X.quantile(0.50, axis = 1).values[0]
    q3 = X.quantile(0.75, axis = 1).values[0]
    std = X.std(axis = 1).values[0]
    sum_ = X.sum(axis = 1).values[0]
    UU = get_XX_feature(X, 'UU')
    AA = get_XX_feature(X, 'AA')
    CC = get_XX_feature(X, 'CC')
    GG = get_XX_feature(X, 'GG')
    sum_g = get_X_feature(X, 'G')
    sum_a = get_X_feature(X, 'A')
    sum_c = get_X_feature(X, 'C')
    sum_u = get_X_feature(X, 'U')
    
    X['kurt'] = kurt
    X['median'] = med
    X['mode'] = mode
    X['var'] = var
    X['max'] = max_
    X['min'] = min_
    X['q1'] = q1
    X['q2'] = q2
    X['q3'] = q3
    X['std'] = std
    X['sum'] = sum_
    X['UU'] = UU
    X['AA'] = AA
    X['CC'] = CC
    X['GG'] = GG
    X['sum_g'] = sum_g
    X['sum_a'] =sum_a
    X['sum_c'] = sum_c
    X['sum_u'] = sum_u
    
    X = pd.DataFrame(std_.transform(X, ), columns = X.columns)
    
    X = X[good_features]
    pred = clf.predict(X)
    return pred, X.values.tolist()[0]




def make_prediction_dnatype(X, clf_path ):   
    X = pd.DataFrame(X.values.reshape(1, -1), columns = X.index, dtype = float)
    
    # load classifier     
    with open(clf_path, 'rb') as file:
        clf = pickle.load(file)
    file.close()
    
    
    # function for calculating double feature
    def get_XX_feature(X, f):
        temp = 0
        for col in X.columns:
            if col[0:2]==f or col[-1:-3]==f or (col[0]==f[0] and col[-1]==f[0]):
                try: 
                    temp = temp + float(X.iloc[0][col])
                except:
                    print(f'There has been a error while calculating {f}')
        return temp
    
    # function for calculating single occurance feature
    def get_X_feature(X, f):
        temp = 0
        for col in X.columns:
            if f in col and len(col)==3:
                try:
                    temp = temp + float(X.iloc[0][col])
                except:
                    print(f'There has been a error while calculating {f}')
        return temp
    X_copy = X.copy()
    
    X['kurt'] = X_copy.kurtosis(axis = 1).values[0]
    X['median'] = X_copy.median(axis = 1).values[0]
    X['mode'] = X_copy.mode(axis = 1).values[0][0]
    X['var'] = X_copy.var(axis = 1).values[0]
    X['min'] = X_copy.min(axis = 1).values[0]
    X['max'] = X_copy.min(axis = 1).values[0]
    X['q1'] = X_copy.quantile(0.25, axis = 1).values[0]
    X['q2'] = X_copy.quantile(0.50, axis = 1).values[0]
    X['q3'] = X_copy.quantile(0.75, axis = 1).values[0]
    X['std'] = X_copy.std(axis = 1).values[0]
    X['sum'] = X_copy.sum(axis = 1).values[0]
    X['UU'] = get_XX_feature(X_copy, 'UU')
    X['AA'] = get_XX_feature(X_copy, 'AA')
    X['CC'] = get_XX_feature(X_copy, 'CC')
    X['GG'] = get_XX_feature(X_copy, 'GG')
    X['sum_g'] = get_X_feature(X_copy, 'G')
    X['sum_a'] = get_X_feature(X_copy, 'A')
    X['sum_c'] = get_X_feature(X_copy, 'C')
    X['sum_u'] = get_X_feature(X_copy, 'U')
    X = X[clf.feature_name_]
    pred = clf.predict(X)
    pred_proba  = clf.predict_proba(X)
    return pred




