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
    euc_list=[]
    temp=0
    for x in X_train:
        euc_list.append(np.linalg.norm(x-x_new))
        temp +=1
    indecies_list=[] 
    gg=np.argsort(euc_list)
    indecies_list=np.array(euc_list)[gg]
    yslist=[]   #this will contain the y values in order of the indecies in the indecies_list
    for i in range(k):
        yslist.append(y_train[euc_list.index(indecies_list[i])])
    values, counts = np.unique(yslist, return_counts=True)  #This will count how many 1s and 0s is there
    if 0 not in values:
        y_new_pred=1
    else:
        y_new_pred =np.argmax(counts)                           #This will return the index of max of the counts.
    return y_new_pred

def logistic_regression_training(X_train, y_train, alpha=0.01, max_iters=5000, random_seed=1):
    copyX_train = np.array(X_train) #maybe there is no need for np.array
    rows, cols = copyX_train.shape
    oneArray = np.ones((rows,1))        # I guess this is its size
    finalXArray = np.hstack((oneArray,copyX_train))
    np.random.seed(random_seed) 
    num_of_features = cols+1      # I guess so, right?
    weights = np.random.normal(loc=0.0, scale=1.0, size=(num_of_features, 1))
    TransFXA = finalXArray.transpose()  #TransposedFinalXArray
    negativeX = -1*finalXArray
    for x in range(max_iters):
        ePart = np.exp(negativeX@weights)   #was negativeX*weights
        sigmoidFun = 1/(1+ePart) 
        weights = weights - alpha*TransFXA@(sigmoidFun-y_train) #was TransFXA*(sigmoidFun-y_train)
    return weights

def logistic_regression_prediction(X, weights, threshold=0.5):
    rows, cols = X.shape
    oneArray = np.ones((rows,1))
    finalXArray = np.hstack((oneArray,X))
    y_preds = 1/(1 + np.exp((-1*finalXArray)@weights))
    y_rows, y_cols = y_preds.shape
    for x in range(y_rows):
        if y_preds[x] < threshold:
            y_preds[x]=0
        else:
            y_preds[x]=1
    return np.array(y_preds)

def model_selection_and_evaluation(alpha=0.01, max_iters=5000, random_seed=1, threshold=0.5):
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_classification_dataset()

    pred_one_nn=knn_classification(X_train,y_train,X_val,1) 
    pred_three_nn=knn_classification(X_train,y_train,X_val,3) 
    pred_five_nn=knn_classification(X_train,y_train,X_val,5) 
    weights=logistic_regression_training(X_train,y_train,alpha,max_iters,random_seed)
    pred_logistic=logistic_regression_prediction(X_train,weights,threshold)
    
    true_one_nn=knn_classification(X_test,y_test,X_val,1) 
    true_three_nn=knn_classification(X_test,y_test,X_val,3) 
    true_five_nn=knn_classification(X_test,y_test,X_val,5) 
    t_weights=logistic_regression_training(X_test,y_test,alpha,max_iters,random_seed)
    true_logistic=logistic_regression_prediction(X_test,t_weights,threshold)

    acc1=(true_one_nn.flatten() == pred_one_nn.flatten()).sum() /true_one_nn.shape[0]
    acc2=(true_three_nn.flatten() == pred_three_nn.flatten()).sum() /true_three_nn.shape[0]
    acc3=(true_five_nn.flatten() == pred_five_nn.flatten()).sum() /true_five_nn.shape[0]
    acc4=(true_logistic.flatten() == pred_logistic.flatten()).sum() /true_logistic.shape[0]
    list_of_acc=[acc1,acc2,acc3,acc4]
    max_index = np.argmax(list_of_acc, axis=0)
    
    X_train_val_merge = np.vstack([X_train, X_val]) 
    y_train_val_merge = np.vstack([y_train, y_val])

    name=""
    if(max_index==0):
        winner=knn_classification(X_train_val_merge,y_train_val_merge,X_test,1)
        name="1nn"
        true=true_one_nn
    elif(max_index==1):
        winner=knn_classification(X_train_val_merge,y_train_val_merge,X_test,3)
        name="3nn"
        true=true_three_nn
    elif(max_index==2):
        winner=knn_classification(X_train_val_merge,y_train_val_merge,X_test,5)
        name="5nn"
        true=true_five_nn
    elif(max_index==3):
        weights=logistic_regression_training(X_train,y_train,alpha,max_iters,random_seed)
        winner=logistic_regression_prediction(X_train,weights,threshold)
        true=true_logistic
        name="logistic regression"
    
    test_acc=(true.flatten() == winner.flatten()).sum() /true.shape[0]
    
    return name,list_of_acc,test_acc
