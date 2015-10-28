# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:05:16 2015

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

# extract the date-timestamps
combined['timestamp'] = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in combined.index]

combined['year'] = [x.year for x in combined['timestamp'] ]
combined['month'] = [x.month for x in combined['timestamp'] ]
combined['hour'] = [x.hour for x in combined['timestamp'] ]
combined['weekday'] = [x.weekday() for x in combined['timestamp'] ]

#combined[ 'weekend'] = 0
#combined.loc[ (combined['holiday'] == 0) & (combined['workingday'] == 0) ,'weekend'] = 1

#combined['daypart'] = 4
#combined.loc[ (combined['hour'] >= 4) & (combined['hour'] < 10), 'daypart' ] = 1
#combined.loc[ (combined['hour'] >= 10) & (combined['hour'] < 15), 'daypart' ] = 2
#combined.loc[ (combined['hour'] >= 15) & (combined['hour'] < 21), 'daypart' ] = 3

combined['peak_hrs'] = 0
combined.loc[ (combined['hour'] >= 7) & (combined['hour'] <= 8), 'peak_hrs' ] = 1
combined.loc[ (combined['hour'] >= 17) & (combined['hour'] <= 18), 'peak_hrs' ] = 1

# combine hour with season
combined['working_hr'] = (combined['workingday'] * 100) + combined['hour']
combined['weather_hr'] = (combined['weather'] + combined['hour']/100) * 100
combined['temp_hr'] = (combined['temp'] / (combined['hour'] + 1))
combined['season_hr'] = (combined['season'] + combined['hour']/100) * 100
combined['season_weather'] = (combined['season'] + combined['weather']/10) * 10
combined['season_temp'] = (combined['season'] * combined['temp'])

#combined['temp_hr'] = (combined['temp'] / (combined['hour'] + 1))

# binning of continuous features
#combined['atemp_cat'] = pd.qcut(combined.atemp.values, [0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
#combined['temp_cat'] = pd.cut(combined.temp.values, [combined.temp.min(), combined.temp.mean(), combined.temp.max()], labels=[1, 2]
#                        , include_lowest=True)
#combined['temp_cat'] = [int(x) for x in combined['temp_cat']]
#combined['temp_cat'] = pd.cut(combined.temp.values, 4, labels=[1, 2, 3, 4])
#combined['daylight'] = pd.cut(combined.hour.values, [0, 8, 24], labels=[1,2], include_lowest=True)

combined['windspeed_cat'] = pd.cut(combined.windspeed.values, 5, labels=[1, 2, 3, 4, 5])
combined['temp2'] = combined['temp']**2
combined['humidity2'] = combined['humidity']**2

# weather outlier!
#combined.loc[ (combined['weather'] == 4), 'weather' ] = 3

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

# handle skewed counts
training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')
testing.to_csv('testing-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

#training_nonworking = training.loc[ (training['workingday'] == 0) ]

featuresUnused1 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'working_hr' ] # , 'season'
results1 = analyzeMetricNumerical('registered_log10',training, featuresUnused1)
showFeatureImportanceNumerical(training, results1['features'], 'registered_log10')

featuresUnused2 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 'season_hr',
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'temp_hr', 'weather_hr',
                   'peak_hrs', 'workingday'] # , 'season', 'workingday'
#featuresUnused2 = [ 'casual','registered','count', 'timestamp', 'atemp', 'humidity', 'temp',
#                   'registered_log10', 'casual_log10', 'workingday'] # , 'season', 'workingday'
results2 = analyzeMetricNumerical('casual_log10', training, featuresUnused2)
showFeatureImportanceNumerical(training, results2['features'], 'casual_log10')

temp1 = predict(results1['model'], testing[results1['features']], 'registered_log10')
testing['registered'] = np.power( 10, temp1['registered_log10'].values ) - 1

temp2 = predict(results2['model'], testing[results2['features']], 'casual_log10')
testing['casual'] = np.power( 10, temp2['casual_log10'].values )  - 1

testing['count'] = testing['registered'] + testing['casual']

print('rmsle of casual = ', rmsle(validation['casual'].values, testing['casual'].values) )
print('rmsle of registered = ', rmsle(validation['registered'].values, testing['registered'].values) )
print('rmsle of count = ', rmsle(validation['count'].values, testing['count'].values) )

#testing = testing[['count']]
#testing.to_csv('submission13.csv', sep=',', encoding='utf-8')

def plotByGrp(groups, x, y):
    # Plot
    plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
    colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')
    
    fig, ax = plt.subplots()
    fig.set_size_inches(11,8)
    ax.set_color_cycle(colors)
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group[x], group[y], marker='o', linestyle='', ms=5, label=name)
    ax.legend(numpoints=1, loc='upper right')
#    ax.set_xlim(-102.4, -101.8)
#    ax.set_ylim(31.2, 31.8)
    plt.show()

#groups = training_nonworking.groupby('temp_hr')
#plotByGrp(groups, 'hour', 'casual')
