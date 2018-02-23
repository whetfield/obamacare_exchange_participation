#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:37:04 2018

@author: whetfield

Analysis helper functions for Project McNulty

df is the main dataframe for the analysis

"""

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_testset_random_issuers(df, test_set_size):
    """
    Return random Series of insurers to be held out of training set for all years
    """

    np.random.seed(42)
    unique_issuer_list = df['IssuerId'].unique() 
    test_set_issuers = np.random.choice(unique_issuer_list, size=int(unique_issuer_list.shape[0] * .20))
    
    return test_set_issuers


    
def get_train_test_data (df,test_set_issuers,features):
    """
    Return X_train,X_test,y_train,y_test numpy arrays from main dataframe.
    Split based on random test_set_issuers from get_testset_random_issuers function
    features is list of string feautres representing columns in df
    """
    
    df_X_train, df_X_test = df[~df['IssuerId'].isin(test_set_issuers)], \
    df[df['IssuerId'].isin(test_set_issuers)]
    
    y_train, y_test = df_X_train['Exit_County_Next_Year'], df_X_test['Exit_County_Next_Year']
    
    X_train, X_test = df_X_train[features], df_X_test[features]
    
    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)
    
    
    return X_train,X_test,y_train,y_test


def naive_bayes_baseline (df, features_to_test):
    """ 
    Pass list of single features to test or can be list of lists of features
    to test.  Will print out accuracy, precision, recall and F1 for binomial
    naive bayes
    """
    validation_holdout = get_testset_random_issuers(df, .20)
    
    for feature in features_to_test: 
        X_train, X_test, y_train, y_test = get_train_test_data (df,validation_holdout,feature)
        
        #need to reshape arraty if only testing one feature
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
        
        
        bnb = BernoulliNB()
        # Fit the model to the training data
        bnb.fit(X_train, y_train)
        # Score the model against the test data
        accuracy = bnb.score(X_test, y_test)
        y_pred = bnb.predict(X_test)
        # Retrieve precision
        p = metrics.precision_score(y_test, y_pred)
        # Retrieve Recall
        r = metrics.recall_score(y_test, y_pred)
        
        #Print feature being baselined
        print(feature)
        #Print accuracy
        print ('Accuracy: ' + str(accuracy))
        # Print precision and recall
        print ('Precision: ' + str(p))
        print ('Recall: ' + str(r))
        # Retrieve F1 from sklearn and print
        print ('F1: ' + str(metrics.f1_score(y_test, y_pred)))
    
    return None
    


def log_reg_baseline (df, features_to_test):
    """ 
    Pass list of single features to test or can be list of lists of features
    to test.  Will print out accuracy, precision, recall and F1 for binomial
    naive bayes
    """
    validation_holdout = get_testset_random_issuers(df, .20)
    
    for feature in features_to_test: 
        X_train, X_test, y_train, y_test = get_train_test_data (df,validation_holdout,feature)
        
        #need to reshape arraty if only testing one feature
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
        
        scaler = StandardScaler()
        log_reg = LogisticRegression()

        pipe = Pipeline(steps = [('scaler',scaler),('log_reg',log_reg)])
        
        # Fit the model to the training data
        pipe.fit(X_train, y_train)
        # Score the model against the test data
        accuracy = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        # Retrieve precision
        p = metrics.precision_score(y_test, y_pred)
        # Retrieve Recall
        r = metrics.recall_score(y_test, y_pred)
        
        #Print feature being baselined
        print(feature)
        #Print accuracy
        print ('Accuracy: ' + str(accuracy))
        # Print precision and recall
        print ('Precision: ' + str(p))
        print ('Recall: ' + str(r))
        # Retrieve F1 from sklearn and print
        print ('F1: ' + str(metrics.f1_score(y_test, y_pred)))
        print ('Coefficients')
        print (pipe.named_steps['log_reg'].coef_)
    return None  
    