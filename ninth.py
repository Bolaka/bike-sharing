# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:47:59 2015

@author: bolaka

starts with
rmsle of registered non working =  0.442794393199
rmsle of casual non working =  0.579054484082
rmsle of registered working =  0.341574279415
rmsle of casual working =  0.53680104828
rmsle of count =  0.366273580611
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
import time
from datetime import datetime, timedelta
from cvxtextproject import *
from mlclassificationlibs import *
from scipy.stats import boxcox

setPath('/home/bolaka/Bike Sharing')

# turn off pandas warning on data frame operations
pd.options.mode.chained_assignment = None  # default='warn'

trainfilename = 'train.csv'
testfilename = 'test.csv'
actualsfilename = 'actuals.csv'

# read data from CSV files
idCol = 'datetime'
training = pd.read_csv(trainfilename, index_col=idCol, parse_dates = True)
testing = pd.read_csv(testfilename, index_col=idCol, parse_dates = True)
validation = pd.read_csv(actualsfilename, index_col=[0, 5], parse_dates = True)
validation = validation[['casual', 'registered', 'count', 'workingday', 'holiday']]

# add metrics dummy columns to test set
testing['count'] = 0
testing['casual'] = 0
testing['registered'] = 0

# merge the training and test sets
trainingLen = len(training)
pieces = [ training, testing ]
combined = pd.concat(pieces)

# extract the date-timestamps
#combined['timestamp'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in combined.index]
combined['Date']= combined.index.date
#validation['timestamp'] = [datetime.strptime(x, "%m/%d/%Y") for x in validation.index]

# weather dummies
combined.loc[ (combined['weather'] == 1), 'weather_name' ] = 'clear_weather'
combined.loc[ (combined['weather'] == 2), 'weather_name' ] = 'mist_cloudy'
combined.loc[ (combined['weather'] == 3), 'weather_name' ] = 'light_rain'
combined.loc[ (combined['weather'] == 4), 'weather_name' ] = 'heavy_rain'
dummies = pd.get_dummies(combined['weather_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop('weather_name', axis=1, inplace=True)

# season dummies
combined.loc[ (combined['season'] == 1), 'season_name' ] = 'spring'
combined.loc[ (combined['season'] == 2), 'season_name' ] = 'summer'
combined.loc[ (combined['season'] == 3), 'season_name' ] = 'winter'
combined.loc[ (combined['season'] == 4), 'season_name' ] = 'fall'
dummies = pd.get_dummies(combined['season_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop('season_name', axis=1, inplace=True)


combined['year'] = [str(x) for x in combined.index.year ]
dummies = pd.get_dummies(combined['year'])
combined = pd.concat([combined, dummies], axis=1)
combined['year'] = combined.index.year

combined['month'] = combined.index.month

combined['hour'] = combined.index.hour

combined['weekday'] = combined.index.weekday
combined.loc[ (combined['weekday'] == 0), 'weekday' ] = 'monday'
combined.loc[ (combined['weekday'] == 1), 'weekday' ] = 'tuesday'
combined.loc[ (combined['weekday'] == 2), 'weekday' ] = 'wednesday'
combined.loc[ (combined['weekday'] == 3), 'weekday' ] = 'thursday'
combined.loc[ (combined['weekday'] == 4), 'weekday' ] = 'friday'
combined.loc[ (combined['weekday'] == 5), 'weekday' ] = 'saturday'
combined.loc[ (combined['weekday'] == 6), 'weekday' ] = 'sunday'
dummies = pd.get_dummies(combined['weekday'])
combined = pd.concat([combined, dummies], axis=1)
combined['weekday'] = combined.index.weekday

combined['temperature'] = combined['temp'] * combined['atemp']

# peak hours
#combined['peak_hrs'] = 0
#combined.loc[ (combined['hour'] >= 7) & (combined['hour'] <= 8), 'peak_hrs' ] = 1
#combined.loc[ (combined['hour'] >= 17) & (combined['hour'] <= 18), 'peak_hrs' ] = 1

## midday hours
#combined['midday_hrs'] = 0
#combined.loc[ (combined['hour'] >= 11) & (combined['hour'] <= 18), 'midday_hrs' ] = 1

# factorize hours
#combined['sleeptime'] = 0
#combined.loc[ (combined['hour'] == 23), 'sleeptime' ] = 1
#combined.loc[ (combined['hour'] >= 0) & (combined['hour'] <= 7), 'sleeptime' ] = 1
#
#combined['morning'] = 0
#combined.loc[ (combined['hour'] == 8) | (combined['hour'] == 10), 'morning' ] = 1

#combined['late_morn'] = 0
#combined.loc[ (combined['hour'] >= 9) & (combined['hour'] <= 10), 'late_morn' ] = 1

combined['afternoon'] = 0
combined.loc[ (combined['hour'] >= 11) & (combined['hour'] <= 15), 'afternoon' ] = 1

#combined['evening'] = 0
#combined.loc[ (combined['hour'] >= 16) & (combined['hour'] <= 20), 'evening' ] = 1
#
#combined['night'] = 0
#combined.loc[ (combined['hour'] >= 21) & (combined['hour'] <= 22), 'night' ] = 1

## feature engineering
#combined[ 'weekend' ] = 0
#combined.loc[ (combined['holiday'] == 0) & (combined['workingday'] == 0) ,'weekend'] = 1
#combined.loc[ (combined['weekend'] == 1), 'holiday'] = 1
#combined['weekday_holiday'] = combined.holiday * (combined.weekday+1)
#combined['atemp_cat'] = pd.cut(combined.atemp.values, 6, labels=[1, 2, 3, 4, 5, 6])
#combined['temp_cat'] = pd.cut(combined.atemp.values, 6, labels=[1, 2, 3, 4, 5, 6])
#combined['hum_cat'] = pd.cut(combined.humidity.values, 4, labels=[1, 2, 3, 4 ])
#combined['windspeed_cat'] = pd.cut(combined.windspeed.values, 4, labels=[ 1, 2, 3, 4 ])
#dummies = pd.get_dummies(combined['windspeed_cat'], prefix='wind')
#combined = pd.concat([combined, dummies], axis=1)


def morningWeather(group):
    morning = group.loc[ (group['hour'] >= 7) & (group['hour'] <= 10) ]
    group['morning_weather'] = morning['weather'].mean()
    
    if (morning['weather'].empty):
        group['morning_weather'] = group['weather'].mean()
    return group
   
# deduce type of day from weather i.e. rainy day, sunny day, etc...
combined = combined.groupby('Date').apply(morningWeather)  
combined.drop('Date', axis=1, inplace=True)

combined['weather_hr'] = (combined['weather'] + combined['hour']/100)
#combined['weather_season'] = (combined['weather'] + combined['season']/10)
#combined['season_hr'] = (combined['season'] + combined['hour']/100) 
#combined['year_season'] = combined['year'] + (combined['season']/10)
#combined['atemp_hr'] = (combined['hour'] + combined['atemp_cat']/10) 
#combined['season_temp'] = (combined['season'] * combined['temp'])
#combined['humidity_temp'] = (combined['temp_cat'] + combined['hum_cat']/10) - nopes, no luck!
combined['weekday_hr'] = (combined['weekday'] + combined['hour']/100) 

## binning of continuous features
#wind_box, wind_lambda = boxcox(combined['windspeed'].values + 1) # Add 1 to be able to transform 0 values
#combined['windspeed_box'] = wind_box
#
##combined['temp2'] = combined['temp']**2
#combined['humidity2'] = combined['humidity']**2
#combined['atemp2'] = combined['atemp']**2

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

# handle skewed counts with log10
#training['registered_ln'] = np.log(training['registered'].values + 1) 
#training['casual_ln'] = np.log(training['casual'].values + 1) 

training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 

reg_box, reg_lambda = boxcox(training['registered'].values + 1) # Add 1 to be able to transform 0 values
training['registered_box'] = reg_box

cas_box, cas_lambda = boxcox(training['casual'].values + 1) # Add 1 to be able to transform 0 values
training['casual_box'] = cas_box

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')
testing.to_csv('testing-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

# separate working from non-working for training, testing & validation sets!
training_nonworking = training.loc[ (training['workingday'] == 0) & (training['holiday'] == 0) ] # weekends
training_working = training.loc[ (training['workingday'] == 1) | (training['holiday'] == 1) ] # weekdays
training_nonworking.to_csv('training-nonworking-features.csv', sep=',', encoding='utf-8', header=True)
training_working.to_csv('training-working-features.csv', sep=',', encoding='utf-8', header=True)

testing_nonworking = testing.loc[ (testing['workingday'] == 0) & (testing['holiday'] == 0) ]
testing_working = testing.loc[ (testing['workingday'] == 1) | (testing['holiday'] == 1) ]
validation_nonworking = validation.loc[ (validation['workingday'] == 0) & (validation['holiday'] == 0) ]
validation_working = validation.loc[ (validation['workingday'] == 1) | (validation['holiday'] == 1) ]


# registered non working
#featuresUnused10 = [ 'casual','registered','count', 'timestamp', 'workingday', 'registered_log10', 'casual_log10',
#                    'season_refrac','holiday', 'temp', 'atemp', 'season', 'hour', 'humidity2', 'weather', 'atemp_cat', 
#                    'year', 'month', 'weekend','peak_hrs', 'windspeed_cat', 'temp_cat', 'windspeed_box', 'hum_cat', 
#                    'weather_season', '2011', 'weekday', 'tuesday', 'thursday', 'wednesday', 'morning_weather', 
#                    'morning_temp', 'registered_box', 'casual_box' ]


featuresUnused10 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
                    'month', 'year', 'season', 
                    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'spring', 'clear_weather', '2012', 
                    'morning_weather']

results10 = analyzeMetricNumerical('registered_log10',training_nonworking, featuresUnused10)
showFeatureImportanceNumerical(training_nonworking, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_nonworking[results10['features']], 'registered_log10')
testing_nonworking['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
print('rmsle of registered non working = ', rmsle(validation_nonworking['registered'].values, testing_nonworking['registered'].values) )
#results10 = analyzeMetricNumerical('registered_box',training_nonworking, featuresUnused10)
#showFeatureImportanceNumerical(training_nonworking, results10['features'], 'registered_box')
#temp10 = predict(results10['model'], testing_nonworking[results10['features']], 'registered_box')
##testing_nonworking['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
#testing_nonworking['registered'] = np.power((temp10['registered_box'].values * reg_lambda) + 1, 1 / reg_lambda) - 1
#print('rmsle of registered non working = ', rmsle(validation_nonworking['registered'].values, testing_nonworking['registered'].values) )



# casual non working
#featuresUnused20 = [ 'casual','registered','count', 'timestamp', 'registered_log10', 'casual_log10', 'workingday','peak_hrs',
#                    'atemp_cat', 'weather_hr', 'year_season', 'atemp_hr', 'holiday', 'weekend', 'season', 'month',
#                    'season_temp', 'humidity2', 'atemp2','atemp', 'windspeed_cat', 'temp_cat', 'windspeed_box', 'hum_cat',
#                    'tuesday', 'thursday', 'wednesday', '2012', 'midday_hrs', 'weather_season', 'registered_box', 'casual_box' ] 
featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 
                    'month', 
                    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'spring', 'clear_weather', '2012', 
                    'weather_hr', 'weekday_hr' ]
#results20 = analyzeMetricNumerical('casual_log10',training_nonworking, featuresUnused20)
#showFeatureImportanceNumerical(training_nonworking, results20['features'], 'casual_log10')
#temp20 = predict(results20['model'], testing_nonworking[results20['features']], 'casual_log10')
#testing_nonworking['casual'] = np.power( 10, temp20['casual_log10'].values )  - 1
#print('rmsle of casual non working = ', rmsle(validation_nonworking['casual'].values, testing_nonworking['casual'].values) )

results20 = analyzeMetricNumerical('casual_box',training_nonworking, featuresUnused20)
showFeatureImportanceNumerical(training_nonworking, results20['features'], 'casual_box')
temp20 = predict(results20['model'], testing_nonworking[results20['features']], 'casual_box')
#testing_nonworking['casual'] = np.power( 10, temp20['casual_log10'].values )  - 1
testing_nonworking['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
print('rmsle of casual non working = ', rmsle(validation_nonworking['casual'].values, testing_nonworking['casual'].values) )


# registered working
#featuresUnused11 = [ 'casual','registered','count', 'timestamp', 'workingday', 'registered_log10', 'casual_log10',
#                    'atemp_hr', 'season_temp', 'humidity2','weather_hr', 'weekday_holiday', 'windspeed_cat',
#                    'atemp_cat','holiday','weekend', 'temp', 'season', 'year', 'month','atemp', 'temp_cat',
#                    'windspeed_box', 'hum_cat', '2012', 'midday_hrs', 'weekday', 'weather_season', 'morning_weather',
#                    'registered_box', 'casual_box']
featuresUnused11 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'holiday', 'atemp', 'temp', 'weather',
                    'month', 'season', 'weekday',
                    'sunday', 'saturday', 'spring', 'clear_weather', '2012', 
                    'morning_weather', 'afternoon' ]
results11 = analyzeMetricNumerical('registered_log10',training_working, featuresUnused11)
showFeatureImportanceNumerical(training_working, results11['features'], 'registered_log10')
temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_log10')
testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )

#results11 = analyzeMetricNumerical('registered_box',training_working, featuresUnused11)
#showFeatureImportanceNumerical(training_working, results11['features'], 'registered_box')
#temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_box')
##testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
#testing_working['registered'] = np.power((temp11['registered_box'].values * reg_lambda) + 1, 1 / reg_lambda) - 1
#print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )


# casual working
#featuresUnused21 = [ 'casual','registered','count', 'timestamp', 'registered_log10', 'casual_log10', 'workingday',
#                    'peak_hrs', 'weekend', 'weekday_holiday', 'year_season', 'windspeed_cat', 'atemp_hr', 'season_temp', 
#                    'humidity2', 'atemp', 'atemp2', 'holiday', 'atemp_cat', 'windspeed', 'temp_cat', 'hum_cat', 'season', 
#                    'weather', '2011', 'sunday', 'saturday','tuesday', 'wednesday', 'weekday_hr', 'registered_box', 'casual_box' ]
featuresUnused21 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 
#                    'month', 
                    'sunday', 'saturday', 'spring', 'clear_weather', '2012', '2011' ]
#results21 = analyzeMetricNumerical('casual_log10',training_working, featuresUnused21)
#showFeatureImportanceNumerical(training_working, results21['features'], 'casual_log10')
#temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_log10')
#testing_working['casual'] = np.power( 10, temp21['casual_log10'].values )  - 1
#print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )
results21 = analyzeMetricNumerical('casual_box',training_working, featuresUnused21)
showFeatureImportanceNumerical(training_working, results21['features'], 'casual_box')
temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_box')
#testing_working['casual'] = np.power( 10, temp21['casual_log10'].values )  - 1
testing_working['casual'] = np.power((temp21['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )



testing_regrouped = pd.concat([ testing_nonworking, testing_working ])
testing_regrouped.sort_index(inplace=True)
testing_regrouped['count'] = testing_regrouped['registered'] + testing_regrouped['casual']
testing_regrouped.to_csv('testing-merged.csv')
valid_regrouped = pd.concat([ validation_nonworking, validation_working ])
valid_regrouped.sort_index(inplace=True) #by=['timestamp', 'hr'],
valid_regrouped.to_csv('valid-merged.csv')

print('rmsle of count = ', rmsle(valid_regrouped['count'].values, testing_regrouped['count'].values) )

#testing_regrouped = testing_regrouped[['count']]
#testing_regrouped.to_csv('submission16.csv', sep=',', encoding='utf-8')
#
##def plotByGrp(groups, x, y):
##    # Plot
##    plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
##    colors = pd.tools.plotting._get_standard_colors(len(groups), color_type='random')
##    
##    fig, ax = plt.subplots()
##    fig.set_size_inches(11,8)
##    ax.set_color_cycle(colors)
##    ax.margins(0.05)
##    for name, group in groups:
##        ax.plot(group[x], group[y], marker='o', linestyle='', ms=5, label=name)
##    ax.legend(numpoints=1, loc='upper right')
##    plt.show()
##
##groups = training_working.groupby('season')
##plotByGrp(groups, 'hour', 'casual')
#
##training_nonworking = prepareSeries1(training_nonworking, 'hour', 'season', 'holiday', False, 5)
##res = plotDensityMatrix1(training_nonworking, 1, 'season', 0, 'weekday-hour')