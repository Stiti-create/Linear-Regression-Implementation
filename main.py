import argparse
import pandas as pd
import numpy as np
import csv
import matplotlib as plt
import math
import random
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel


#minimum square error function #debugged
def MSE(y_pred, y_true):
    n = len(y_pred)
    sum = 0
    for i in range(n):
        sum += (y_pred[i] - y_true[i])**2
    sum/=n
    return sum

def MAE(y_pred, y_true):
    n = len(y_pred)
    sum = 0
    for i in range(n):
        sum += abs(y_pred[i] - y_true[i])
    sum/=n
    return sum

#gradient for batch gradient descend
def batchGradient(y_pred, y_true, index, X):
    sum = 0
    for i in range(len(y_pred)):
       sum+=((y_pred[i]-y_true[i])*X[i][index])
    sum/=(len(y_pred))
    
    return sum

#gradient algorithm for stochastic gradient descend
def stoGradient(y_pred, y_true, index, X, sampnum):
    grad = 0
    grad = (y_pred[sampnum]-y_true[sampnum])*X[sampnum][index]
    return grad

def identfun(x,y):
    if(x==y):
        return 1
    else:
        return 0

def checklastlabel(x,y):
    if(x==9) and (y==9):
        return 1
    else:
        return 0

#linear regression algorithm
def linearRegression(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, stopcrit,maxit):
    #training algorithm
    n_sample = len(train_X)
    n_feature = len(train_X[0])+1
    # weights = np.random.rand(n_feature)
    weights = [1/math.sqrt(n_feature)]*(n_feature)
    # weights[0]=1;
    features = []
    valfeatures = []
    for i in range(n_sample):
        X = np.append([1],train_X[i])
        features.append(X)
        
    for i in range(val_X.shape[0]):
        X2 = np.append([1],val_X[i])
        valfeatures.append(X2)
    # print(len(features[0]))
    msestore = []
    mseval = []
    
    #batch gradient descend
    if(stopcrit==1): #maxit
        iter = 0
        while(iter<maxit):
            iter+=1
            y_pred = []
            for i in range(n_sample):
                y_pred.append(np.dot(weights, features[i]))  
            val_pred = []
            for i in range(val_X.shape[0]):
                val_pred.append(np.dot(weights, valfeatures[i]))
            for i in range(n_feature):               
                temp=weights[i]-2*(learning_rate*batchGradient(y_pred,train_y,i,features))
                weights[i]=temp
            msestore.append(MSE(y_pred,train_y)) 
            mseval.append(MSE(val_pred, val_y_true))
            # print(f'ITERTATION: {iter}, MSE: {MSE(y_pred,train_y)}')
            # print(f'ITERTATION: {iter}, MSE: {MSE(val_pred, val_y_true)} on val')
            if(iter>5):
                if(mseval[-1]>mseval[-2]):
                    break      
        # print(weights)
        # print(f'Final MSE: {MSE(y_pred,train_y)} on train')
        # print(f'Final MAE: {MAE(y_pred, train_y)} on train')
        # print(f'Final MSE: {MSE(val_pred, val_y_true)} on val')
        # print(f'Final MAE: {MAE(val_pred, val_y_true)} on val')
    else: #reltop
        pass
        threshold = 3e-6
        iter = 0
        rel_diff = float('inf')
        preverr = float('inf')
        minvalerr = float('inf')
        while(rel_diff>threshold):
            iter+=1
            y_pred = []
            for i in range(n_sample):
                y_pred.append(np.dot(weights, features[i]))  
            val_pred = []
            for i in range(val_X.shape[0]):
                val_pred.append(np.dot(weights, valfeatures[i]))
            # norm = np.dot(weights,weights)     
            # print(f'NORM: {norm}')
            for i in range(n_feature):               
                temp=weights[i]-2*(learning_rate*batchGradient(y_pred,train_y,i,features))
                weights[i]=temp
            currMSE = MSE(y_pred,train_y)
            msestore.append(MSE(y_pred,train_y)) 
            mseval.append(MSE(val_pred, val_y_true))
            rel_diff = abs(currMSE-preverr)
            preverr = currMSE
            minvalerr = min(minvalerr, currMSE)
            # print(f'relative diff, iterations({iter}): {rel_diff}')
            # print(f'minvalerr: {minvalerr}')
        # print(weights)
            # print(f'ITERTATION: {iter}, MSE: {currMSE}, stepsize: {stepsize}') 
        print(f'Number of iterations: {iter}')
        # print(f'Final MSE: {currMSE} ')  
        # print(f'Final MAE: {MAE(y_pred, train_y)}') 
        print(f'Final MSE: {MSE(val_pred, val_y_true)} with reltop = 3.10^-6 on val')
        print(f'Final MAE: {MAE(val_pred, val_y_true)} with reltop = 3.10^-6 on val')
    
    # msefile = open('./plots/section-1/msetrain.csv','a',newline='')
    # header = 'stopcrit learning_rate mse1 mse2 mse3 mse4 mse5 mse6 mse7 mse8 mse9 mse10 mse11 mse12 mse13 mse14 mse15 mse16 mse17 mse18 mse19 mse20'

    # with msefile:
    #     csv.writer(msefile).writerow(mseval.split())
    return [msestore, mseval, weights]

#cost/gradient for Ridge regression
#cost/gradient for Ridge regression
def costRidge(y_pred, y_true, X, idx, lamda, theta):
    n = len(y_pred)
    sum = 0
    for i in range(n):
        sum+=((y_pred[i]-y_true[i])*X[i][idx])
    sum*=(2)
    sum+=(2*lamda*theta)
    sum/=n
    return sum

#Ridge Regression Algorithm
def RidgeRegression(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, stopcrit, lamda, maxit):
    
    #training algorithm
    n_sample = len(train_X)
    n_feature = len(train_X[0])+1
    weights = [0]*(n_feature)
    weights[0]=1
    features = []
    for i in range(n_sample):
        X = np.append([1],train_X[i])
        features.append(X)
    valfeatures = []   
    for i in range(val_X.shape[0]):
        X2 = np.append([1],val_X[i])
        valfeatures.append(X2)
    
    # print(len(features[0]))
    msestore = []
    mseval = []
    if(stopcrit==1): #maxit
        iter = 0
        while(iter<maxit):
            iter+=1
            y_pred = []
            for i in range(n_sample):
                y_pred.append(np.dot(weights, features[i]))      
            val_pred = []
            for i in range(val_X.shape[0]):
                val_pred.append(np.dot(weights, valfeatures[i]))
            temp = weights[0]-learning_rate*(batchGradient(y_pred, train_y, 0, features))
            weights[0] = temp
            for i in range(1,n_feature):
                temp=weights[i]-learning_rate*(costRidge(y_pred, train_y, features, i, lamda, weights[i]))
                weights[i]=temp
        # print(weights)
            print(f'ITERTATION: {iter}, {MSE(val_pred, val_y_true)} on val')
            msestore.append(MSE(y_pred, train_y))
            mseval.append(MSE(val_pred, val_y_true))
            if(iter>5):
                if(mseval[iter-1]>mseval[iter-2]):
                    break
        
        print(f'Final MSE: {MSE(y_pred, train_y)} on train')
        print(f'Final MSE: {MSE(val_pred, val_y_true)} on val')
        # print(f'Final MAE: {MAE(y_pred, train_y)} on train')
        # print(f'Final MAE: {MAE(val_pred, val_y_true)} on val')
        
        
    else: #reltop
        iter = 0
        threshset = 7.5e-07
        while(threshold>=threshset):
            iter+=1
            y_pred = []
            for i in range(n_sample):
                y_pred.append(np.dot(weights, features[i]))      
            val_pred = []
            for i in range(val_X.shape[0]):
                val_pred.append(np.dot(weights, valfeatures[i]))
            temp = weights[0]-learning_rate*(batchGradient(y_pred, train_y, 0, features))
            weights[0] = temp
            for i in range(1,n_feature):
                temp=weights[i]-learning_rate*(costRidge(y_pred, train_y, features, i, lamda, weights[i]))
                weights[i]=temp
        # print(weights)
            print(f'ITERTATION: {iter}, {MSE(val_pred, val_y_true)} on val')
            msestore.append(MSE(y_pred, train_y))
            mseval.append(MSE(val_pred, val_y_true))
            currMse = MSE(val_pred, val_y_true)
            threshold = min(threshold, abs(currMse-prevMSE))
            prevMSE = currMse
            print(f'Threshold decrease: {threshold}')
        
        # print(f'Final MSE: {MSE(y_pred, train_y)} on train')
        print(f'Final MSE: {MSE(val_pred, val_y_true)} on val')
        # print(f'Final MAE: {MAE(y_pred, train_y)} on train')
        print(f'Final MAE: {MAE(val_pred, val_y_true)} on val')
        print(f'Threshold decrease: {threshold}')

    return [msestore, mseval, weights]

def softgradient(train_y, x, ind, probmat, label):
    n_samp = len(train_y)
    grad = 0
    n_lab = len(probmat[0])
    for i in range(n_samp):
        # denosum = 1
        # for k in range(n_lab):
        #     denosum += probmat[i][k]
        tmp = identfun(label,train_y[i]) - probmat[i][label-1]
        tmp *= x[i][ind]
        grad += tmp
    
    return grad

def multiclassifier(train_X, train_y, val_X, val_y_true, x_test, learning_rate, maxit, stopcrit):
    
    n_sample = len(train_X)
    n_feature = len(train_X[0])+1
    n_labels = 9
    wtmat = []
    for n in range(n_labels-1):
        # wts = np.random.rand(n_feature)
        wts = [0]*n_feature
        wts[0]=1
        wtmat.append(wts)
   
    features = []
    valfeatures = []
    for i in range(n_sample):
        X = np.append([1],train_X[i])
        features.append(X)
        
    for i in range(val_X.shape[0]):
        X2 = np.append([1],val_X[i])
        valfeatures.append(X2)
    # print(len(features[0]))
    msetrain = []
    mseval = []
    if(stopcrit==1):
        iter = 0
        while(iter<maxit):
            iter+=1
            y_pred = []
            probmat = []
            val_pred = []
        
            for i in range(n_sample):
                probvect = []
                sumprobs = 1
                for k in range(n_labels-1):
                    probvect.append(math.exp(np.dot(wtmat[k],features[i])))
                    sumprobs+=probvect[-1]
                for k in range(n_labels-1):
                    probvect[k]/=sumprobs
                probmat.append(probvect)

                y_pred.append(np.argmax(probvect)+1)
                
                sumfor9 = 0
                for k in range(n_labels-1):
                    sumfor9+=probvect[k]
                comp = 1-sumfor9
                
                if y_pred[-1] < comp:
                    y_pred[-1] = 9
            for i in range(val_X.shape[0]):
                probvect = []
                sumprobs = 1
                for k in range(n_labels-1):
                    probvect.append(math.exp(np.dot(wtmat[k],valfeatures[i])))
                    sumprobs+=probvect[-1]
                for k in range(n_labels-1):
                    probvect[k]/=sumprobs
                val_pred.append(np.argmax(probvect)+1)
                sumfor9 = 0
                for k in range(n_labels-1):
                    sumfor9+=probvect[k]
                comp = 1-sumfor9
                
                if val_pred[-1] < comp:
                    val_pred[-1] = 9
            
            for k in range(n_labels-1):
                for j in range(n_feature):
                    tmp = softgradient(train_y, features, j, probmat, k+1)
                    wtmat[k][j]=wtmat[k][j]+((learning_rate*tmp))
            msetrain.append(MSE(y_pred, train_y))
            mseval.append(MSE(val_pred, val_y_true))
            print("iteration: ", iter, "MSE for train: ", msetrain[-1])
            print("iteration: ", iter, "MSE for val: ", mseval[-1])
    return (msetrain, mseval, wtmat)   

def predMulticlass(test_X, wtmat):
    n_labels = len(wtmat)
    pred_test = []
    for i in range(len(test_X)):
        probvector = []
        sumprobs = 1
        for k in range(n_labels):
            probvector.append(math.exp(np.dot(wtmat[k],test_X[i])))
            sumprobs+=probvector[-1]
        for k in range(n_labels):
            probvector[k]/=sumprobs
        probvector.append(1-sum(probvector))
        pred_test.append(np.argmax(probvector)+1)
    return pred_test
             
def FeatureSelect(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, selectType, learning_rate, maxit):
    features = []
    n_sample = len(train_X)
    n_feature = len(train_X[0])+1
    
    if(selectType==0): #selectbestk
        selectedfeat = SelectKBest(f_classif, k=10)
        x_new = selectedfeat.fit_transform(train_X, train_y) 
        newfeats = selectedfeat.get_support()
        index = []
        for i in range(len(newfeats)):
            if(newfeats[i]==True):
                index.append(i)   
        newtrainx = []
        for i in range(len(train_X)):
            lis = []
            for j in index:
                lis.append(train_X[i][j])
            newtrainx.append(lis)
        newtrainx = np.array(newtrainx)
        newvalx = []
        for i in range(len(val_X)):
            lis = []
            for j in index:
                lis.append(val_X[i][j])
            newvalx.append(lis)
        newvalx = np.array(newvalx)
        msestore, mseval, weights = linearRegression(newtrainx, train_y, newvalx, val_y_true, test_X, test_samples, outpath, learning_rate, 1, maxit)
        return [msestore, mseval, weights, index]
    else:
        #using select from model code from scikit
        ridgeestimator = Ridge().fit(train_X, train_y)
        Selectmodel = SelectFromModel(ridgeestimator, threshold=0.1)
        X_new = Selectmodel.transform(train_X)  
        newfeats = Selectmodel.get_support()
        # print(newfeats.\) 
        index = []
        for i in range(len(newfeats)):
            if(newfeats[i]==True):
                index.append(i)   
        newtrainx = []
        for i in range(len(train_X)):
            lis = []
            for j in index:
                lis.append(train_X[i][j])
            newtrainx.append(lis)
        newtrainx = np.array(newtrainx)
        newvalx = []
        for i in range(len(val_X)):
            lis = []
            for j in index:
                lis.append(val_X[i][j])
            newvalx.append(lis)
        newvalx = np.array(newvalx)
        print(newtrainx.shape)
        msestore, mseval, weights = linearRegression(newtrainx, train_y, newvalx, val_y_true, test_X, test_samples, outpath, learning_rate, 1, maxit)
        return [msestore, mseval, weights, index]
         
#######BONUS PART#############
def predval(X, w):
    tmp = np.dot(X, w)
    y_pred = 1/(1+math.exp(-tmp))
    return y_pred

def LogRegresion(output, X, y, learning_rate, maxit, train_X):
    n_sample, n_feature = X.shape
    n_feature+=1
    weights = [0]*(n_feature)
    weights[0]=1
    features = []
    for i in range(n_sample):
        X = np.append([1],train_X[i])
        features.append(X)
    iter = 0
    msetrain = []
    while(iter < maxit):
        iter+=1
        y_pred = []
        for i in range(n_sample):
            y_pred.append(predval(features[i], weights))
        for j in range(n_feature):
            temp = weights[j]-learning_rate*batchGradient(y_pred, y, j, features)
            weights[j] = temp
        msetrain.append(MSE(y_pred, y))
        print(f'ITERATION: {iter} MSE: {msetrain[-1]}')
    print(f'Final MSE: {msetrain[-1]} for class {output}')
    return (weights, msetrain)

def OneVsRestClassifier(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, maxiter):
    weight2d = []
    for i in range(1,10):
        y = np.array([1 if x == i else 0 for x in train_y])
        weight = LogRegresion(i, train_X, y, learning_rate, maxiter, train_X)
        weight2d.append(weight[0])        
    return weight2d

def OnevsRest(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, maxit):
    maxiter = maxit 

    wtmat = OneVsRestClassifier(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, maxiter)

    valfeatures = []
    for i in range(val_X.shape[0]):
        X2 = np.append([1],val_X[i])
        valfeatures.append(X2)
    valpred = []
    print(len(wtmat))
    print(len(wtmat[0]))
    for i in range(val_X.shape[0]):
        predictions = []
        for j in range(9):
            predictions.append(predval(valfeatures[i], wtmat[j]))
        score = np.argmax(predictions)
        score+=1
        valpred.append(score)
    return wtmat
    

#Function to take input parameters from console input #debugged
def parseConsole():
    argp = argparse.ArgumentParser()
    argp.add_argument('-t', '--train_path', type=str, required=True, help='path to training data')
    argp.add_argument('-v', '--val_path', type=str, required=True, help='path to validation data')
    argp.add_argument('-p', '--test_path', type=str, required=True, help='path to test data')
    argp.add_argument('-o', '--out_path', type=str, required=True, help='path to generate output')
    argp.add_argument('-s', '--section', type=str, required=True, help='model name 1: linear regression, 2: ridge regression, 5: classification')

    args = vars(argp.parse_args())

    [trainpath, valpath, testpath, outpath, section] = [args['train_path'], args['val_path'], args['test_path'], args['out_path'], int(args['section'])]
    return [trainpath, valpath, testpath, outpath, section]

def outputoncsv(test_pred, outpath):
    file = open(outpath, 'w')
    with file:
        write = csv.writer(file)
        write.writerows(test_pred)
    return

def addbias(X):
    n_sample = X.shape[0]
    features = []
    for i in range(n_sample):
        Xnew = np.append([1],X[i])
        features.append(Xnew)
    return features

def main():
    
    [trainpath, valpath, testpath, outpath, section]=parseConsole()
        
    #read train data from csv file
    traindata = pd.read_csv(trainpath)
    train_samples = np.array(traindata.iloc[:, 0])
    train_y = np.array(traindata.iloc[:, 1])
    train_X = np.array(traindata.iloc[:, 2:])
         
    #read validation dataset from csv file
    valdata = pd.read_csv(valpath)
    val_samples = np.array(valdata.iloc[:, 0])
    val_y_true = np.array(valdata.iloc[:, 1])
    val_X = np.array(valdata.iloc[:, 2:])
    
    #read test dataset from csv file
    testdata = pd.read_csv(testpath)
    test_samples = np.array(testdata.iloc[:, 0])
    test_X = np.array(testdata.iloc[:, 1:])
    
    test_X = addbias(test_X)
    
    if section == 1:
        learning_rate = 0.001
        stopcrit = 1
        maxit = 500
        [msestore, mseval, weights] = linearRegression(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, stopcrit, maxit)
        
        n = len(test_samples)
        test_pred = []
        for i in range(n):
            namepair = [test_samples[i],np.dot(test_X[i], weights)]
            test_pred.append(namepair)
        outputoncsv(test_pred, outpath)
            
            
    elif section == 2:
        learning_rate = 0.001
        stopcrit = 1
        maxit = 400
        lamda = 25
        [msestore, mseval, weights]= RidgeRegression(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, stopcrit, lamda, maxit)
        n = len(test_samples)
        test_pred = []
        for i in range(n):
            namepair = [test_samples[i],np.dot(test_X[i], weights)]
            test_pred.append(namepair)
        outputoncsv(test_pred, outpath)
        
        
    elif section == 5:
        maxit = 50
        stopcrit = 1
        learning_rate = 0.0001
        [msestore, mseval, wtmat] = multiclassifier(train_X, train_y, val_X, val_y_true, test_X, learning_rate, maxit, stopcrit)
        predvalues = predMulticlass(test_X, wtmat)
        n = len(test_samples)
        test_pred = []
        for i in range(n):
            namepair = [test_samples[i],predvalues[i]]
            test_pred.append(namepair)
        outputoncsv(test_pred, outpath)
        
    else:
        #onevsrest
        learning_rate = 0.001
        stopcrit = 1
        maxit = 100
        lamda = 25
        weights= OnevsRest(train_X, train_y, val_X, val_y_true, test_X, test_samples, outpath, learning_rate, maxit)
        n = len(test_samples)
        test_pred = []
        for i in range(n):
            namepair = [test_samples[i],np.dot(test_X[i], weights)]
            test_pred.append(namepair)
        outputoncsv(test_pred, outpath)
        
    return

main()

