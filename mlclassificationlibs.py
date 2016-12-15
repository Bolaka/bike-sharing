# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:45:24 2015

@author: bolaka
"""

# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression as LinR
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence

#class mllibs:
    
# extract the names of the pandas data frame
def names(pandas_dataframe):
    return (pandas_dataframe.columns.values.tolist())

# Cross Validation
def runCVNumerical(x,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
#        print('Training Samples = ', len(train_index), '; Test Samples', len(test_index))
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        
    y_pred[y_pred < 0] = 0
    return y_pred     

# Cross Validation
def fitRandomForestRegressor(x,y):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = RFR(n_estimators=50)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        
#    y_pred[y_pred < 0] = 0
    return y_pred  

# Cross Validation
def runCV(x,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
#        print('Training Samples = ', len(train_index), '; Test Samples', len(test_index))
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(random_state=100)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        
#    y_pred[y_pred < 0] = 0
    return y_pred

# accuracy
def classificationAccuracy(Y_ori,Y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(Y_ori == Y_pred)
#    return (accuracy_score(Y_ori, Y_pred, normalize=False)/len(Y_ori) * 100)

# Confusion Matrix    
def drawConfusionMatrix(confusion_matrix,class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrix:
        classifier, cm, accu = cm[0], cm[1], cm[2]
        print(classifier)
        print(accu)
#        print(cm)
        print('\n') 
        
        norm_cm = []
        for i in cm:
                a = 0
                tmp_arr = []
                a = sum(i,0)
                for j in i:
                        tmp_arr.append(float(j)/float(a) * 100)
                norm_cm.append(tmp_arr)        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        cax = ax.matshow(cm)
        cax = ax.imshow(norm_cm, interpolation='nearest')
        for i, cas in enumerate(norm_cm):
            for j, c in enumerate(cas):
                if c>0:
                    plt.text(j-.2, i+.2, "%.1f" % c, fontsize=14)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def analyzeMetric(metric, training, excludeFeatures, max_features):
#    metric = 'Code 2'
    print('Analyzing', metric)
    
    X = training[training.columns - excludeFeatures]
    Y = training[metric]
    
    # To reproduce results, fix the random seed
    seed(1)
    
#    print(X.info(),'\n')
    features = X.columns

#     as an array
    x = np.asanyarray(X)
    y = np.asanyarray(Y)
    #est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    #est.fit(X,Y)
    #Y_HAT = est.predict(X)

    # classifier predictions
    linearP = runCV(x, y, LR)
    gbmP = runCV(x, y, GBC )
    svmP = runCV(x, y, SVC)
    forestP = runCV(x, y, RF)
    adaP = runCV(x, y, ABC )
    bagP = runCV(x, y, BC)
    extraP = runCV(x, y, ETC)
    #knnP = runCV(x, y, KNN)

    # accuracies
    linearAcc = classificationAccuracy(y, linearP) 
    gbmAcc = classificationAccuracy(y, gbmP) 
    svmAcc = classificationAccuracy(y, svmP) 
    forestAcc = classificationAccuracy(y, forestP) 
    adaAcc = classificationAccuracy(y, adaP) 
    bagAcc = classificationAccuracy(y, bagP) 
    extraAcc = classificationAccuracy(y, extraP) 
    #knnAcc = classificationAccuracy(y, knnP) 

    mappings = [
        { 'name' : "Logistic Regression", 
          'algo' : LR
        },
        { 'name' : "Gradient Boosting Classifier", 
          'algo' : GBC
        },
        { 'name' : "Support Vector Machines", 
          'algo' : SVC
        },
        { 'name' : "Random Forest", 
          'algo' : RF
        },
        { 'name' : "Ada Boost Classifier", 
          'algo' : ABC
        },
        { 'name' : "Bagging Classifier", 
          'algo' : BC
        },
        { 'name' : "Extra Trees Classifier", 
          'algo' : ETC
        }
    #    { 'name' : "K-Nearest-Neighbors", 
    #      'algo' : KNN
    #    }
    ]
    accuracies = [
        { 'name' : "Logistic Regression", 
          'accuracy' : linearAcc
        },
        { 'name' : "Gradient Boosting Classifier", 
          'accuracy' : gbmAcc
        },
        { 'name' : "Support Vector Machines", 
          'accuracy' : svmAcc
        },
        { 'name' : "Random Forest", 
          'accuracy' : forestAcc 
        },
        { 'name' : "Ada Boost Classifier", 
          'accuracy' : adaAcc 
        },
        { 'name' : "Bagging Classifier", 
          'accuracy' : bagAcc 
        },
        { 'name' : "Extra Trees Classifier", 
          'accuracy' : extraAcc 
        }
    #    ,{ 'name' : "K-Nearest-Neighbors", 
    #      'accuracy' : knnAcc
    #    }
    ]
    
    print(Y.value_counts(), '\n')

#    class_names = np.unique(y)
#    confusion_matrices = [
#        ( "Support Vector Machines", confusion_matrix( y, svmP ), svmAcc  ),
#        ( "Random Forest", confusion_matrix( y, forestP ), forestAcc ),
#    #    ( "K-Nearest-Neighbors", confusion_matrix( y, knnP ), knnAcc ),
#        ( "Gradient Boosting Classifier", confusion_matrix( y,gbmP ), gbmAcc ),
#        ( "Logistic Regression", confusion_matrix( y,linearP ), linearAcc ),
#        ( "Ada Boost Classifier", confusion_matrix( y, adaP ), adaAcc  ),
#        ( "Bagging Classifier", confusion_matrix( y, bagP ), bagAcc ),
#        ( "Extra Trees Classifier", confusion_matrix( y, extraP ), extraAcc )
#    ]

#    drawConfusionMatrix(confusion_matrices, class_names)
    #print(accuracies)
    theBest = max(accuracies, key=lambda x:x['accuracy'])['name']
    bestAlgo = next(d for (index, d) in enumerate(mappings) if d["name"] == theBest)['algo']
    bestAcc = max(accuracies, key=lambda x:x['accuracy'])['accuracy']
    print('Best performer:')
    print(theBest, bestAcc, bestAlgo)

    # get the best performing algorithm
    best = bestAlgo() # n_estimators=300, subsample=0.8, max_features=max_features
    fit = best.fit(X, Y)
    
#    y_hat = best.predict(X)    
#    training[metric + '_ML'] = y_hat    
    
#    # validation predictions    
#    y_hat_val = best.predict(x_val)
#    valAccuracy = classificationAccuracy(y_val,y_hat_val)
#    print('Accuracy on validation set for',metric,'is',valAccuracy)
#    validation[metric + '_ML'] = y_hat_val
    
    return ({ 'fit' : fit, 'features' : features, 'model' : best})

def analyzeMetricNumerical(metric, training, excludeFeatures, tuning=False):
#    metric = 'Code 2'
#    print('Analyzing', metric, 'with\n', training.columns)
    
    X = training[training.columns - excludeFeatures]
    Y = training[metric]
    
    # To reproduce results, fix the random seed
    seed(1)
    
    features = X.columns
#    print(X.info())

#     as an array
    x = np.asanyarray(X)
    y = np.asanyarray(Y)

    # regressor predictions
    linearP = runCVNumerical(x, y, LinR)
    gbrP = runCVNumerical(x, y, GBR )
#    forestP = fitRandomForestRegressor(x, y)
    forestP = runCVNumerical(x, y, RFR)
    treeP = runCVNumerical(x, y, DTR)

    # RMSLEs
    linearErr = rmsle(y, linearP) 
    gbrErr = rmsle(y, gbrP) 
    forestErr = rmsle(y, forestP) 
    treeErr = rmsle(y, treeP) 

    mappings = [
        { 'name' : "Linear Regression", 
          'algo' : LinR
        },
        { 'name' : "Gradient Boosting Regressor", 
          'algo' : GBR
        },
        { 'name' : "Random Forest Regressor", 
          'algo' : RFR
        },
        { 'name' : "Decision Tree Regressor", 
          'algo' : DTR
        }
    ]
    errors = [
        { 'name' : "Linear Regression", 
          'accuracy' : linearErr
        },
        { 'name' : "Gradient Boosting Regressor", 
          'accuracy' : gbrErr
        },
        { 'name' : "Random Forest Regressor", 
          'accuracy' : forestErr 
        },
        { 'name' : "Decision Tree Regressor", 
          'accuracy' : treeErr 
        }
    ]
    
    print(errors, '\n\n')
    theBest = min(errors, key=lambda x:x['accuracy'])['name']
    bestAlgo = next(d for (index, d) in enumerate(mappings) if d["name"] == theBest)['algo']
    bestErr = min(errors, key=lambda x:x['accuracy'])['accuracy']
    print('Best performer = ', theBest, bestErr, bestAlgo)

#    theBest = 'Gradient Boosting Regressor'
#    bestAlgo = GBR

    # get the best performing algorithm
    if theBest is "Decision Tree Regressor":
        best = bestAlgo()
    else:
        if tuning:
            print('plot_partial_dependence...')
            best = GBR(n_estimators=300) # , subsample=0.8, max_features=8
#            fit_partial = best.fit(training, y)
#            fig, axs = plot_partial_dependence(fit_partial, training,
#                                   features=range(len(training.columns)),
#                                   feature_names=training.columns,
#                                   n_cols=2)
#            fig.set_size_inches(13,25)
#            plt.subplots_adjust(top=1.5)
#            fig.show()
            
            fit = best.fit(X, Y)            
#            fig2, axs2 = plot_partial_dependence(fit, X,
#                                   features=range(len(X.columns)),
#                                   feature_names=X.columns,
#                                   n_cols=2)
#            fig2.set_size_inches(13,20)
#            plt.subplots_adjust(top=1.5)
#            fig2.show()
        else:
            print('no tuning...')
            best = bestAlgo(n_estimators=300)
    #        search = GridSearchCV(best, param_grid, verbose=2)
            fit = best.fit(X, Y)
    #        print(fit.best_params_)
#            y_hat = best.predict(X) 
        
#    best = GBR(n_estimators=200)
#    fit = best.fit(X, Y)

    return ({ 'fit' : fit, 'features' : features, 'model' : best})

def analyzeMetricNumericalRMSE(metric, training, excludeFeatures):
    print('\nModeling', metric)
    
    X = training[training.columns - excludeFeatures]
    Y = training[metric]
    
    # To reproduce results, fix the random seed
    seed(1)
    
    features = X.columns
    print(X.info())

#     as an array
    x = np.asanyarray(X)
    y = np.asanyarray(Y)

    # regressor predictions
    linearP = runCVNumerical(x, y, LinR)
    gbrP = runCVNumerical(x, y, GBR )
#    forestP = fitRandomForestRegressor(x, y)
    forestP = runCVNumerical(x, y, RFR)
    treeP = runCVNumerical(x, y, DTR)

    # RMSLEs
    linearErr = rmse(y, linearP) 
    gbrErr = rmse(y, gbrP) 
    forestErr = rmse(y, forestP) 
    treeErr = rmse(y, treeP) 

    mappings = [
        { 'name' : "Linear Regression", 
          'algo' : LinR
        },
        { 'name' : "Gradient Boosting Regressor", 
          'algo' : GBR
        },
        { 'name' : "Random Forest Regressor", 
          'algo' : RFR
        },
        { 'name' : "Decision Tree Regressor", 
          'algo' : DTR
        }
    ]
    errors = [
        { 'name' : "Linear Regression", 
          'accuracy' : linearErr
        },
        { 'name' : "Gradient Boosting Regressor", 
          'accuracy' : gbrErr
        },
        { 'name' : "Random Forest Regressor", 
          'accuracy' : forestErr 
        },
        { 'name' : "Decision Tree Regressor", 
          'accuracy' : treeErr 
        }
    ]

    errors = sorted(errors, key=lambda k: k['accuracy'])     
    
    for error in errors:
        print(error['name'], ' scores an RMSE value of ', error['accuracy'])        
        
        
    theBest = min(errors, key=lambda x:x['accuracy'])['name']
    bestAlgo = next(d for (index, d) in enumerate(mappings) if d["name"] == theBest)['algo']
    bestErr = min(errors, key=lambda x:x['accuracy'])['accuracy']
    print('\nBest performer:')
    print(theBest, ' with an RMSE of ', bestErr)

    best = GBR(n_estimators=1000, learning_rate = 0.09, loss = 'ls', random_state = 652100, max_depth=2, subsample=0.8)
    fit = best.fit(X, Y)
    gbrPreds = runCVNumerical(x, y, GBR, n_estimators=1000, learning_rate = 0.09, loss = 'ls', 
                              random_state = 652100, max_depth=2, subsample=0.8 )
    gbrFinalErr = rmse(y, gbrPreds)
    print('test rmse = ', gbrFinalErr)
    fi = best.feature_importances_ 
    sum_fi = {}
    for j in range( len( features ) ):
        if features[j] in sum_fi:
            sum_fi[features[j]] = sum_fi[features[j]] + fi[j] 
        else:
            sum_fi[features[j]] = fi[j] 
    
    sum_fi_list = [[key, sum_fi[key]] for key in sum_fi]
    sum_fi_list.sort( key = lambda x : x[1], reverse = True )
    
    return ({ 'features' : features, 'predictions' : gbrPreds, 'model' : best, 'importance' : sum_fi_list })

def analyzeMetricNumericalShell(metric, training, excludeFeatures):
    print('\nModeling', metric)
    
    X = training[training.columns - excludeFeatures]
    Y = training[metric]
    
    # To reproduce results, fix the random seed
    seed(1)
    
    features = X.columns
#    print(X.info())

#     as an array
    x = np.asanyarray(X)
    y = np.asanyarray(Y)

    # regressor predictions
    linearP = runCVNumerical(x, y, LinR)
    gbrP = runCVNumerical(x, y, GBR )
#    forestP = fitRandomForestRegressor(x, y)
    forestP = runCVNumerical(x, y, RFR)
    treeP = runCVNumerical(x, y, DTR)

    # RMSLEs
    linearErr = rmse(y, linearP) 
    gbrErr = rmse(y, gbrP) 
    forestErr = rmse(y, forestP) 
    treeErr = rmse(y, treeP) 

    mappings = [
        { 'name' : "Linear Regression", 
          'algo' : LinR
        },
        { 'name' : "Gradient Boosting Regressor", 
          'algo' : GBR
        },
        { 'name' : "Random Forest Regressor", 
          'algo' : RFR
        },
        { 'name' : "Decision Tree Regressor", 
          'algo' : DTR
        }
    ]
    errors = [
        { 'name' : "Linear Regression", 
          'accuracy' : linearErr
        },
        { 'name' : "Gradient Boosting Regressor", 
          'accuracy' : gbrErr
        },
        { 'name' : "Random Forest Regressor", 
          'accuracy' : forestErr 
        },
        { 'name' : "Decision Tree Regressor", 
          'accuracy' : treeErr 
        }
    ]

    errors = sorted(errors, key=lambda k: k['accuracy'])     
    
    for error in errors:
        print(error['name'], ' scores an RMSE value of ', error['accuracy'])        
        
        
    theBest = min(errors, key=lambda x:x['accuracy'])['name']
    bestAlgo = next(d for (index, d) in enumerate(mappings) if d["name"] == theBest)['algo']
    bestErr = min(errors, key=lambda x:x['accuracy'])['accuracy']
    print('\nBest performer:')
    print(theBest, ' with an RMSE of ', bestErr)
    
    if theBest == 'Gradient Boosting Regressor':
        best = GBR(n_estimators=1000, learning_rate = 0.09, loss = 'ls', random_state = 652100, max_depth=2, subsample=0.8)
        fit = best.fit(X, Y)
        gbrPreds = runCVNumerical(x, y, GBR, n_estimators=1000, learning_rate = 0.09, loss = 'ls', 
                              random_state = 652100, max_depth=2, subsample=0.8 )
        gbrFinalErr = rmse(y, gbrPreds)
        print('test rmse by GBR = ', gbrFinalErr)
    elif theBest == 'Random Forest Regressor':
        best = RFR(n_estimators=800, random_state = 652100, max_features=0.2)
        fit = best.fit(X, Y)
        gbrPreds = runCVNumerical(x, y, RFR, n_estimators=800, random_state = 652100, max_features=0.2 )
        gbrFinalErr = rmse(y, gbrPreds)
        print('test rmse by RFR = ', gbrFinalErr)
  
    return ({ 'features' : features, 'predictions' : gbrPreds, 'model' : best })

def predict(model, testing, metric):
    # predictions    
    testing[metric] = model.predict(testing)
    return testing
    
def showFeatureImportance(training, features, metric):
    # feature importance
    rf = RF(random_state=100)
    fit = rf.fit(training[features], training[metric])
    importances = fit.feature_importances_
#    print(importances)
#        indices = np.argsort(importances)[::-1]
#        # Print the feature ranking
#        print("Feature ranking:")
#        for f in range(len(features)):
#            print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
    
    importances = 100.0 * (importances / importances.max())
    indices = np.argsort(importances)
    pos = np.arange(indices.shape[0]) + 0.5
    plt.figure(figsize=(12,8))
    plt.subplot(1, 1, 1)
    plt.barh(pos, importances[indices], align='center')
    plt.yticks(pos, features[indices])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def showFeatureImportanceNumerical(training, features, metric):
    # feature importance
    rf = RFR()
    fit = rf.fit(training[features], training[metric])
    importances = fit.feature_importances_
#        indices = np.argsort(importances)[::-1]
#        # Print the feature ranking
#        print("Feature ranking:")
#        for f in range(len(features)):
#            print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
    
    importances = 100.0 * (importances / importances.max())
    indices = np.argsort(importances)
    pos = np.arange(indices.shape[0]) + 0.5
    plt.figure(figsize=(12,8))
    plt.subplot(1, 1, 1)
    plt.barh(pos, importances[indices], align='center')
    plt.yticks(pos, features[indices])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def rmse(actual, predicted):
    """
    Computes the root mean squared error.
    This function computes the root mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared error between actual and predicted
    """
    return np.sqrt(mse(actual, predicted))

def rmsle(actual, predicted):
    """
    Computes the root mean squared log error.
    This function computes the root mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The root mean squared log error between actual and predicted
    """
    return np.sqrt(msle(actual, predicted))

def mse(actual, predicted):
    """
    Computes the mean squared error.
    This function computes the mean squared error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared error between actual and predicted
    """
    return np.mean(se(actual, predicted))

def msle(actual, predicted):
    """
    Computes the mean squared log error.
    This function computes the mean squared log error between two lists
    of numbers.
    Parameters
    ----------
    actual : list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double
            The mean squared log error between actual and predicted
    """
    return np.mean(sle(actual, predicted))

def se(actual, predicted):
    """
    Computes the squared error.
    This function computes the squared error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared error between actual and predicted
    """
    return np.power(np.array(actual)-np.array(predicted), 2)

def sle(actual, predicted):
    """
    Computes the squared log error.
    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.
    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value
    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted
    """
    return (np.power(np.log(np.array(actual)+1) - 
            np.log(np.array(predicted)+1), 2))

def mape( actual, predicted ):
    return np.mean(np.abs((actual - predicted) / actual))