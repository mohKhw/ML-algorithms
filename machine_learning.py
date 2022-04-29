import numpy as np
import pandas as pd

def preprocess_classification_dataset():
    ##train##
    csv_reader_train = pd.read_csv('train.csv')
    train_feat_df = csv_reader_train.iloc[:,:-1] # grab all columns except the last one 
    train_output = csv_reader_train[['output']]
    X_train = train_feat_df.values
    y_train = train_output.values
    ##val##
    csv_reader_val = pd.read_csv('val.csv')
    val_feat_df = csv_reader_val.iloc[:,:-1] # grab all columns except the last one 
    val_output = csv_reader_val[['output']]
    X_val = val_feat_df.values
    y_val = val_output.values
    ##test##
    csv_reader_test = pd.read_csv('test.csv')
    test_feat_df = csv_reader_test.iloc[:,:-1] # grab all columns except the last one 
    test_output = csv_reader_test[['output']]
    X_test = test_feat_df.values
    y_test = test_output.values
    return X_train, y_train, X_val, y_val, X_test, y_test

def knn_classification(X_train, y_train, x_new, k=5):
    return 998

def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    return 997

def logistic_regression_prediction(X, weights, threshold=0.5):
    return 996

def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    return 995
