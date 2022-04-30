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
    print(X_train)
    print(y_train)
    return X_train, y_train, X_val, y_val, X_test, y_test

def knn_classification(X_train, y_train, x_new, k=5):
    #firstList=[]
    secondList=[]
    for x in X_train:
        firstList=[]
        temp = 0
        for r in x:
            value = (x[temp]-x_new[temp])**2
            firstList.append(value)
        euc=(sum(firstList))**(1/2)
        secondList.append(euc)
    minList=secondList
    kNearest=[]
    xsList=[]   #a list the will have the indecies of the nearest Ks (Xs)
    #smallest=0
    for i in range(0,k):
        #tempCount=0
        index=0
        minimum=9999999
        for j in range(len(minList)):
            if minList[j]<minimum:
                minimum=minList[j]
                index=secondList.index(minList[j])
                #xsList.append(tempCount)
                #smallest=tempCount
            #tempCount+=1
        #xsList.append(smallest)
        xsList.append(index)
        minList.remove(minimum)
        kNearest.append(minimum)
    yslist=[]
    for x in xsList:
        yslist.append(y_train[x])
    values, counts = np.unique(yslist, return_counts=True)
    if len(counts)==1:
        if values[0]==1:
            y_new_pred = 1
        elif values[0]==0:
            y_new_pred = 0
    else:
        if counts[0] > counts[1]:   #i am not sure
            y_new_pred = 0
        else:
            y_new_pred = 1
    return y_new_pred
    #ynew = sum(secondList)/len(secondList)


def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    return 997

def logistic_regression_prediction(X, weights, threshold=0.5):
    return 996

def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    return 995
