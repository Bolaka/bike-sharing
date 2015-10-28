# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:36:35 2015

@author: bolaka
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
#testfilename = 'test.csv'
#actualsfilename = 'actuals.csv'

# read data from CSV files
idCol = 'datetime'
training = pd.read_csv(trainfilename, index_col=idCol)

# extract the date-timestamps
training['timestamp'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in training.index]

training['year'] = [x.year for x in training['timestamp'] ]
training['month'] = [x.month for x in training['timestamp'] ]
training['hour'] = [x.hour for x in training['timestamp'] ]
training['weekday'] = [x.weekday() for x in training['timestamp'] ]

# handle skewed counts with log10
training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 

# separate working from non-working for training, testing & validation sets!
training_nonworking = training.loc[ (training['workingday'] == 0) ]
training_working = training.loc[ (training['workingday'] == 1) ]

d = training_nonworking
metric = 'registered_log10'

data = prepareSeries1(d, 'hour', 'weekday', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'weekday', 0, 0, 'density')
#
#data = prepareSeries1(d, 'month', 'year', metric, False, 5)
#res = plotDensityMatrix(data, 'month', 'year', 0, 0, 'density')

data = prepareSeries1(d, 'season', 'year', metric, False, 5)
res = plotDensityMatrix(data, 'season', 'year', 0, 0, 'density')

#data = prepareSeries1(d, 'weekday', 'season', metric, False, 5)
#res = plotDensityMatrix(data, 'weekday', 'season', 0, 0, 'density')
#
#data = prepareSeries1(d, 'weekday', 'weather', metric, False, 5)
#res = plotDensityMatrix(data, 'weekday', 'weather', 0, 0, 'density')
#
#data = prepareSeries1(d, 'weekday', 'humidity', metric, False, 5)
#res = plotDensityMatrix(data, 'weekday', 'humidity', 0, 20, 'density')
#
#data = prepareSeries1(d, 'weekday', 'temp', metric, False, 5)
#res = plotDensityMatrix(data, 'weekday', 'temp', 0, 5, 'density')
#
#data = prepareSeries1(d, 'weekday', 'windspeed', metric, False, 5)
#res = plotDensityMatrix(data, 'weekday', 'windspeed', 0, 10, 'density')
#
#
#
#data = prepareSeries1(d, 'hour', 'weekday', metric, False, 5)
#res = plotDensityMatrix(data, 'hour', 'weekday', 0, 0, 'density')
#
data = prepareSeries1(d, 'hour', 'season', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'season', 0, 0, 'density')

data = prepareSeries1(d, 'hour', 'weather', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'weather', 0, 0, 'density')
#
data = prepareSeries1(d, 'hour', 'humidity', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'humidity', 0, 25, 'density')
#
data = prepareSeries1(d, 'hour', 'temp', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'temp', 0, 10, 'density')
#
data = prepareSeries1(d, 'hour', 'windspeed', metric, False, 5)
res = plotDensityMatrix(data, 'hour', 'windspeed', 0, 20, 'density')

#data = prepareSeries1(d, 'temp', 'humidity', metric, False, 5)
#res = plotDensityMatrix(data, 'temp', 'humidity', 5, 10, 'density')

#data = prepareSeries1(d, 'season', 'weather', metric, False, 5)
#res = plotDensityMatrix(data, 'season', 'weather', 0, 0, 'density')

#d = training_nonworking
#metric = 'casual_log10'
#
#data = prepareSeries1(d, 'season', 'year', metric, False, 5)
#res = plotDensityMatrix(data, 'season', 'year', 0, 0, 'density')