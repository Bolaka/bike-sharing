# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:16:41 2015

@author: bolaka
rmsle = 0.3843410
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
import time
from datetime import datetime, timedelta
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
validation = validation[['casual', 'registered', 'count', 'workingday', 'hr']]

# add metrics dummy columns to test set
testing['count'] = 0
testing['casual'] = 0
testing['registered'] = 0

# merge the training and test sets
trainingLen = len(training)
pieces = [ training, testing ]
combined = pd.concat(pieces)

# extract the date-timestamps
combined['timestamp'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in combined.index]
validation['timestamp'] = [datetime.strptime(x, "%m/%d/%Y") for x in validation.index]

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

# binning of continuous features
#combined['atemp_cat'] = pd.qcut(combined.atemp.values, [0, .25, .5, .75, 1], labels=[1, 2, 3, 4])
#combined['temp_cat'] = pd.cut(combined.temp.values, [combined.temp.min(), combined.temp.mean(), combined.temp.max()], labels=[1, 2]
#                        , include_lowest=True)
#combined['temp_cat'] = [int(x) for x in combined['temp_cat']]
#combined['temp_cat'] = pd.cut(combined.temp.values, 4, labels=[1, 2, 3, 4])
combined['windspeed_cat'] = pd.cut(combined.windspeed.values, 5, labels=[1, 2, 3, 4, 5])
#combined['daylight'] = pd.cut(combined.hour.values, [0, 8, 24], labels=[1,2], include_lowest=True)
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

training_nonworking = training.loc[ (training['workingday'] == 0) ]
training_working = training.loc[ (training['workingday'] == 1) ]
testing_nonworking = testing.loc[ (testing['workingday'] == 0) ]
testing_working = testing.loc[ (testing['workingday'] == 1) ]
validation_nonworking = validation.loc[ (validation['workingday'] == 0) ]
validation_working = validation.loc[ (validation['workingday'] == 1) ]

featuresUnused10 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 'workingday',
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'working_hr' ] # , 'season'
results10 = analyzeMetricNumerical('registered_log10',training_nonworking, featuresUnused10)
showFeatureImportanceNumerical(training_nonworking, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_nonworking[results10['features']], 'registered_log10')
testing_nonworking['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1

featuresUnused11 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 'workingday',
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'working_hr' ] # , 'season'
results11 = analyzeMetricNumerical('registered_log10',training_working, featuresUnused11)
showFeatureImportanceNumerical(training_working, results11['features'], 'registered_log10')
temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_log10')
testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
print('rmsle of registered non working = ', rmsle(validation_nonworking['registered'].values, testing_nonworking['registered'].values) )
print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )




featuresUnused20 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 'season_hr',
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'temp_hr', 'weather_hr',
                   'peak_hrs', 'workingday'] # , 'season', 'workingday'
results20 = analyzeMetricNumerical('casual_log10',training_nonworking, featuresUnused20)
showFeatureImportanceNumerical(training_nonworking, results20['features'], 'casual_log10')
temp20 = predict(results20['model'], testing_nonworking[results20['features']], 'casual_log10')
testing_nonworking['casual'] = np.power( 10, temp20['casual_log10'].values )  - 1

featuresUnused21 = [ 'casual','registered','count', 'timestamp', 'atemp', 'windspeed', 'season_hr',
                   'registered_log10', 'casual_log10', 'temp', 'humidity', 'temp_hr', 'weather_hr',
                   'peak_hrs', 'workingday'] # , 'season', 'workingday'
results21 = analyzeMetricNumerical('casual_log10',training_working, featuresUnused21)
showFeatureImportanceNumerical(training_working, results21['features'], 'casual_log10')
temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_log10')
testing_working['casual'] = np.power( 10, temp21['casual_log10'].values )  - 1
print('rmsle of casual non working = ', rmsle(validation_nonworking['casual'].values, testing_nonworking['casual'].values) )
print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )



testing_regrouped = pd.concat([ testing_nonworking, testing_working ])
testing_regrouped.sort_index(inplace=True)
testing_regrouped['count'] = testing_regrouped['registered'] + testing_regrouped['casual']
testing_regrouped.to_csv('testing-merged.csv')
valid_regrouped = pd.concat([ validation_nonworking, validation_working ])
valid_regrouped.sort_index(by=['timestamp', 'hr'],inplace=True)
valid_regrouped.to_csv('valid-merged.csv')

print('rmsle of count = ', rmsle(valid_regrouped['count'].values, testing_regrouped['count'].values) )

#testing = testing[['count']]
#testing.to_csv('submission13.csv', sep=',', encoding='utf-8')

#def plotByGrp(groups, x, y):
#    # Plot
#    plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
#    colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')
#    
#    fig, ax = plt.subplots()
#    fig.set_size_inches(11,8)
#    ax.set_color_cycle(colors)
#    ax.margins(0.05)
#    for name, group in groups:
#        ax.plot(group[x], group[y], marker='o', linestyle='', ms=5, label=name)
#    ax.legend(numpoints=1, loc='upper right')
#    plt.show()

#groups = training.groupby('workingday')
#plotByGrp(groups, 'hour', 'casual')
