# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:18:31 2015

@author: bolaka

submission1.csv - first pass @ 1.25643166239
submission2.csv - first pass
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
#import math
import time
import datetime
from cvxtextproject import *
from mlclassificationlibs import *

setPath('/home/bolaka/Bike Sharing')

trainfilename = 'train.csv'
testfilename = 'test.csv'

idCol = 'datetime'
training = pd.read_csv(trainfilename, index_col=idCol)
testing = pd.read_csv(testfilename, index_col=idCol)

# extract the date-timestamps from training and testing files
training['time_index'] = range(1, len(training) + 1 )
training['timestamp'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in training.index]

testing['time_index'] = range(len(training) + 1, len(training) + len(testing) + 1)
testing['timestamp'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in testing.index]

training['year'] = [x.year for x in training['timestamp'] ]
training['month'] = [x.month for x in training['timestamp'] ]
training['day'] = [x.day for x in training['timestamp'] ]
training['hour'] = [x.hour for x in training['timestamp'] ]
training['weekday'] = [x.weekday() for x in training['timestamp'] ]
#training['yearday'] = [x.timetuple().tm_yday for x in training['timestamp'] ]

testing['year'] = [x.year for x in testing['timestamp'] ]
testing['month'] = [x.month for x in testing['timestamp'] ]
testing['day'] = [x.day for x in testing['timestamp'] ]
testing['hour'] = [x.hour for x in testing['timestamp'] ]
testing['weekday'] = [x.weekday() for x in testing['timestamp'] ]
#testing['yearday'] = [x.timetuple().tm_yday for x in testing['timestamp'] ]

training[ 'weekend'] = 0
training.loc[ (training['holiday'] == 0) & (training['workingday'] == 0) ,'weekend'] = 1

testing[ 'weekend'] = 0
testing.loc[ (testing['holiday'] == 0) & (testing['workingday'] == 0) ,'weekend'] = 1

# handle skewed counts
#training['registered_sqrt'] = np.sqrt(training['registered'].values) 
#training['casual_sqrt'] = np.sqrt(training['casual'].values)  

#training['registered_log10'] = np.log10(training['registered'].values + 1) 
#training['casual_log10'] = np.log10(training['casual'].values + 1)  

training.to_csv('training-newfeatures.csv')

featuresUnused1 = [ 'casual','registered','count', 'timestamp', 'atemp', 'holiday', 'year' ] #'registered_sqrt', 'casual_sqrt', 'registered_log10', 'casual_log10' ]
results1 = analyzeMetricNumerical('registered',training, featuresUnused1)
showFeatureImportanceNumerical(training, results1['features'], 'registered')

featuresUnused2 = [ 'casual','registered','count', 'timestamp', 'atemp', 'holiday' ] #'registered_sqrt', 'casual_sqrt', 'registered_log10', 'casual_log10'
results2 = analyzeMetricNumerical('casual',training, featuresUnused2)
showFeatureImportanceNumerical(training, results2['features'], 'casual')

temp1 = predict(results1['model'], testing[results1['features']], 'registered')
testing['registered'] = temp1['registered']
#testing['registered'] = np.square( temp1['registered_sqrt'].values )
#testing['registered'] = np.power( 10, temp1['registered_log10'].values ) - 1

temp2 = predict(results2['model'], testing[results2['features']], 'casual')
testing['casual'] = temp2['casual']
#testing['casual'] = np.square( temp2['casual_sqrt'].values ) 
#testing['casual'] = np.power( 10, temp2['casual_log10'].values )  - 1

testing['count'] = testing['registered'] + testing['casual']

testing = testing[['count']]
testing.to_csv('submission6.csv', sep=',', encoding='utf-8')