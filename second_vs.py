# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:47:29 2015

@author: bolaka

submission1.csv - first pass @ 1.25643166239
submission2.csv - first pass
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
import time
import datetime
from cvxtextproject import *
from mlclassificationlibs import *

setPath('/home/bolaka/Bike Sharing')

trainfilename = 'train.csv'
testfilename = 'test.csv'
actualsfilename = 'actuals.csv'

# read data from CSV files
idCol = 'datetime'
training = pd.read_csv(trainfilename, index_col=idCol)
testing = pd.read_csv(testfilename, index_col=idCol)
validation = pd.read_csv(actualsfilename, index_col=idCol)
validation = validation[['casual', 'registered', 'count']]

# add metrics dummy columns to test set
testing['count'] = 0
testing['casual'] = 0
testing['registered'] = 0

# merge the training and test sets
trainingLen = len(training)
pieces = [ training, testing ]
combined = pd.concat(pieces)

# extract the date-timestamps from training and testing files
#combined['time_index'] = range(1, len(combined) + 1 )
combined['timestamp'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in combined.index]

combined['year'] = [x.year for x in combined['timestamp'] ]
combined['month'] = [x.month for x in combined['timestamp'] ]
#combined['day'] = [x.day for x in combined['timestamp'] ]
combined['hour'] = [x.hour for x in combined['timestamp'] ]
combined['weekday'] = [x.weekday() for x in combined['timestamp'] ]

combined[ 'weekend'] = 0
combined.loc[ (combined['holiday'] == 0) & (combined['workingday'] == 0) ,'weekend'] = 1

# binning of continuous features
#tempbox = training.boxplot('temp')
#temphist = training.temp.hist()
#combined['atemp_cat'] = pd.qcut(combined.atemp.values, [0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
#combined['windspeed_cat'] = pd.cut(combined.windspeed.values, 5, labels=[1, 2, 3, 4, 5])
#combined['daylight'] = pd.cut(combined.hour.values, [0, 8, 24], labels=[1,2], include_lowest=True)

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

featuresUnused1 = [ 'casual','registered','count', 'timestamp', 'atemp' ] # ,, 'year'
results1 = analyzeMetricNumerical('registered',training, featuresUnused1)
showFeatureImportanceNumerical(training, results1['features'], 'registered')

featuresUnused2 = [ 'casual','registered','count', 'timestamp', 'temp', 'holiday' ] #  , 'atemp', 'windspeed',
results2 = analyzeMetricNumerical('casual',training, featuresUnused2)
showFeatureImportanceNumerical(training, results2['features'], 'casual')

temp1 = predict(results1['model'], testing[results1['features']], 'registered')
testing['registered'] = temp1['registered']

temp2 = predict(results2['model'], testing[results2['features']], 'casual')
testing['casual'] = temp2['casual']

testing['count'] = testing['registered'] + testing['casual']

print('rmsle = ', rmsle(validation['count'].values, testing['count'].values) )

testing = testing[['count']]
testing.to_csv('submission7.csv', sep=',', encoding='utf-8')