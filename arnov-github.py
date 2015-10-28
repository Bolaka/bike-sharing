# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:32:27 2015

@author: bolaka
"""

import sys
import math
from pprint import pprint
from itertools import chain, combinations
from datetime import datetime, timedelta
import random

import pandas as pd
import numpy as np
from sklearn import cross_validation, tree, svm, linear_model, preprocessing, \
    neighbors, ensemble
import matplotlib.pyplot as plt
from workalendar.usa import Maryland

def rmsle(actual_values, predicted_values):
    '''
        Implementation of Root Mean Squared Logarithmic Error
        See https://www.kaggle.com/c/bike-sharing-demand/details/evaluation
    '''
    assert len(actual_values) == len(predicted_values), \
            "Both input paramaters should have the same length"

    # Depending on the regression method, the input paramaters can be either
    # a numpy.ndarray or a list, we need to make sure it's a 1D iterable
    actual_values = np.ravel(actual_values)
    predicted_values = np.ravel(predicted_values)

    total = 0
    for a, p in zip(actual_values, predicted_values):
        total += math.pow(math.log(p+1) - math.log(a+1), 2)

    return math.sqrt(total/len(actual_values))

def pre_process_data(data, selected_columns):
    '''
        Does some pre-processing on the existing columns and only keeps
        columns present in [selected_columns].
        Returns a numpy array
    '''

    # Some 'magic' string to datatime function
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Since the hour of day is cyclical, e.g. 01:00 is equaly far from midnight
    # as 23:00 we need to represent this in a meaningful way. We use both sin
    # and cos, to make sure that 12:00 != 00:00 (which we cannot prevent if we only
    # use sin)
    data['hour_of_day'] = data['datetime'].apply(lambda i: i.hour)
    data['hour_of_day_sin'] = data['hour_of_day'].apply(lambda hour: math.sin(2*math.pi*hour/24))
    data['hour_of_day_cos'] = data['hour_of_day'].apply(lambda hour: math.cos(2*math.pi*hour/24))

    # Since it seems the service got more popular over time, we might need some
    # way of telling how far we are from the beginning
    first_day = datetime.strptime('2011-01-01', "%Y-%m-%d").date()
    data['day_since_begin'] = data['datetime'].apply(lambda i: (i.date()-first_day).days)

    # For some reason the dataset didn't indicate new year's day and christmas
    # day as holidays. Therefore we also use this external libraryto check if
    # a day is a holiday
    cal = Maryland()
    holidays = cal.holidays(2011)
    holidays += cal.holidays(2012)

    holidays = set([dt for (dt, name) in holidays])
    data['holiday_external'] = data['datetime'].apply(lambda i: int(i.date() in holidays))

    # Is it a holiday tomorrow or yesterday?
    data['almost_holiday'] = data['datetime'].apply(
        lambda i: int(i.date() - timedelta(days=1) in holidays or
            i.date() + timedelta(days=1) in holidays)
        )

    # Some simple model of rush hour
    data['rush_hour'] = data['datetime'].apply(
        lambda i: min([math.fabs(8-i.hour), math.fabs(18-i.hour)])
    )
    data.ix[data['workingday'] == 0,'rush_hour'] = \
        data['datetime'].apply(
            lambda i: math.fabs(14-i.hour)
        )
    data.ix[data['holiday_external'] == 1,'rush_hour'] = \
        data['datetime'].apply(
            lambda i: math.fabs(14-i.hour)
        )

    # Add the day of the week
    data['weekday'] = data['datetime'].apply(lambda i: i.weekday())

    # Some variables have no numerical value, they are categorical. E.g. the weather
    # variable has numerical values, but they cannot be interpreted as such.
    # In other words value 2 is not two times as small as value 4.
    # A method to deal with this is one-hot-enconding, which splits the existing
    # variable in n variables, where n equals the number of possible values.
    # See
    for column in ['season', 'weather', 'weekday']:
        dummies = pd.get_dummies(data[column])
        # Concat actual column name with index
        new_column_names = [column + str(i) for i in dummies.columns]
        data[new_column_names] = dummies
    
    data.to_csv('/home/bolaka/Bike Sharing/train-arnov.csv', index=False)
    data = data[selected_columns]

    return data.values

def print_feature_importance(data, features, labels_casual, labels_registered):
    '''
        Use a random forest regressor to print some info on how important the
        diffrent features are
    '''
    clf_c = ensemble.RandomForestRegressor(n_estimators=150)
    clf_r = ensemble.RandomForestRegressor(n_estimators=150)

    clf_c.fit(data,np.ravel(labels_casual))
    clf_r.fit(data,np.ravel(labels_registered))

    print( 'Registered features:' )
    pprint(sorted(zip(features, clf_r.feature_importances_),
        key=lambda i: i[1], reverse=True))
    print( 'Casual features:' )
    pprint(sorted(zip(features, clf_c.feature_importances_),
        key=lambda i: i[1], reverse=True))

def main(algorithm='random-forest'):
    # Read data from file and convert to dataframe
    data = pd.read_csv("/home/bolaka/Bike Sharing/train.csv")
    data = pd.DataFrame(data = data)

    # This list decides which features are going to be used, the rest is filtered out
    features = ['day_since_begin', 'hour_of_day_cos', 'hour_of_day_sin', 'workingday',
        'temp', 'weather1', 'weather3', 'holiday_external', 'almost_holiday',
        'weekday0', 'weekday1', 'weekday2', 'weekday3', 'weekday4', 'weekday5',
        'weekday6', 'humidity', 'windspeed', 'rush_hour', 'holiday']

    # Extract and select features
    train_data = pre_process_data(data, features)    
    
    # Get target values
    train_labels_casual = data[['casual']].values.astype(float)
    train_labels_registered = data[['registered']].values.astype(float)
    train_labels_count = data[['count']].values.astype(float)

    # Inspect feature importance
    print_feature_importance(
        train_data, features, train_labels_casual, train_labels_registered
    )

    # Do cross validation by leaving out specific weeks
    weeks = data['datetime'].apply(lambda i: str(i.year) + '-' + str(i.week))

    # Take out 10 weeks to test on, but don't do ALL permutations
    kf = []
    for fold in cross_validation.LeavePLabelOut(weeks, p=10):
        kf.append(fold)
        if len(kf) == 6:
            break

    scores = []
    for fold, (train_index, test_index) in enumerate(kf):
        print( "Computing fold %d" % fold )

        # Train the model

        if algorithm == 'ridge':
            clf_c = linear_model.Ridge(alpha = 1.5)
            clf_r = linear_model.Ridge(alpha = 1.5)

        elif algorithm == 'decision-tree':
            clf_c = tree.DecisionTreeRegressor(max_depth=15)
            clf_r = tree.DecisionTreeRegressor(max_depth=15)

        elif algorithm == 'knn':
            clf_c = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')
            clf_r = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance')

        elif algorithm == 'svr':
            clf_c = svm.SVR(kernel='rbf', C=10000, epsilon=5)
            clf_r = svm.SVR(kernel='rbf', C=10000, epsilon=5)

        elif algorithm == 'random-forest':
            clf_c = ensemble.RandomForestRegressor(n_estimators=150)
            clf_r = ensemble.RandomForestRegressor(n_estimators=150)
        else:
            raise Exception("Unkown algorithm type, choices are: 'ridge'," +
                " 'decision-tree', 'knn', 'svr', and 'random-forest' (default)")

        clf_c.fit(train_data[train_index],np.ravel(train_labels_casual[train_index]))
        clf_r.fit(train_data[train_index],np.ravel(train_labels_registered[train_index]))

        # Test it
        predicted_c = clf_c.predict(train_data[test_index])
        predicted_r = clf_r.predict(train_data[test_index])

        # Some methods can predict negative values
        predicted_c = [p if p > 0 else 0 for p in predicted_c]
        predicted_r = [p if p > 0 else 0 for p in predicted_r]

        # Plot predicted vs true values for a random week
        df = pd.DataFrame({'datetime': data['datetime'].values[test_index],
            'true_c': np.ravel(train_labels_casual[test_index]),
            'predicted_c': np.ravel(predicted_c),
            'true_r': np.ravel(train_labels_registered[test_index]),
            'predicted_r': np.ravel(predicted_r)})

        index = random.randint(0,len(df))
        df[index:index+24*7].plot(x='datetime')
        plt.show()

        # Add casual and registered prediction
        predicted = [c+r for (c,r) in zip(predicted_c, predicted_r)]

        scores.append(rmsle(train_labels_count[test_index], predicted))

    # Print average cross-validation score
    avg = sum(scores) / len(scores)
    print( "Average RMSLE:", avg )

    # Train classifier on all data
    clf_c.fit(train_data,np.ravel(train_labels_casual))
    clf_r.fit(train_data,np.ravel(train_labels_registered))

    # Predict test data
    test_data = pd.read_csv("/home/bolaka/Bike Sharing/test.csv")
    test_data = pd.DataFrame(data = test_data)
    transformed_test_data = pre_process_data(test_data, features)

    # Predict all test data
    predicted_c = clf_c.predict(transformed_test_data)
    predicted_r = clf_r.predict(transformed_test_data)

    # Add up casual and registered prediction
    predicted = [c+r for (c,r) in zip(predicted_c, predicted_r)]

    # Some methods can predict negative values
    predicted = [p if p > 0 else 0 for p in predicted]

    # Write the output to a csv file
    output = pd.DataFrame()
    output['datetime'] = test_data['datetime']
    output['count'] = predicted

    # Don't write row numbers, using index=False
    output.to_csv('/home/bolaka/Bike Sharing/predicted.csv', index=False)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print( "Unknown number of parameters.\n")
        print( "Usage 'python prediction.py' or 'python prediction.py [algorithm]'" )
        exit(1)