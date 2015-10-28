# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:56:25 2015

@author: bolaka
"""

import os
os.chdir('/home/bolaka/python-workspace/CVX-timelines/')

# imports
import time
from datetime import datetime, timedelta, date
from cvxtextproject import *
from mlclassificationlibs import *
from scipy.stats import boxcox
from workalendar.usa import Maryland

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
validation = validation[['casual', 'registered', 'count', 'weekday', 'holiday']]

## some plots
#training.plot(y='count')
#training.plot(y='registered')
#training.plot(y='casual')
#
## Plot one week (24 * 7 = 168 datapoints)
#training[7000:7168].plot(y='count')

# add metrics dummy columns to test set
testing['count'] = 0
testing['casual'] = 0
testing['registered'] = 0

# merge the training and test sets
trainingLen = len(training)
pieces = [ training, testing ]
combined = pd.concat(pieces)

## extract the date-timestamps
combined['Date']= combined.index.date
firstDate = combined.index.date[0]
#combined['time_index'] = [ (x - firstDate).days for x in combined.index.date ]

# weather dummies
combined.loc[ (combined['weather'] == 1), 'weather_name' ] = 'clear_weather'
combined.loc[ (combined['weather'] == 2), 'weather_name' ] = 'mist_cloudy'
combined.loc[ (combined['weather'] == 3), 'weather_name' ] = 'light_rain'
combined.loc[ (combined['weather'] == 4), 'weather_name' ] = 'heavy_rain'
dummies = pd.get_dummies(combined['weather_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop(['weather_name', 'heavy_rain', 'mist_cloudy' ], axis=1, inplace=True) # 

## season dummies
#combined.loc[ (combined['season'] == 1), 'season_name' ] = 'spring'
#combined.loc[ (combined['season'] == 2), 'season_name' ] = 'summer'
#combined.loc[ (combined['season'] == 3), 'season_name' ] = 'fall'
#combined.loc[ (combined['season'] == 4), 'season_name' ] = 'winter'
#dummies = pd.get_dummies(combined['season_name'])
#combined = pd.concat([combined, dummies], axis=1)
#combined.drop([ 'season_name' ], axis=1, inplace=True) # , 'spring', 'summer', 'winter', 'spring'

#combined['weekofyear'] = [int(x.strftime('%V')) for x in combined.index ] 
combined['dayofyear'] = [int(x.strftime('%j')) for x in combined.index ] 
combined['year'] = [str(x) for x in combined.index.year ]
dummies = pd.get_dummies(combined['year'])
combined = pd.concat([combined, dummies], axis=1)
combined['year'] = combined.index.year
combined.drop(['2012'], axis=1, inplace=True) # 

combined['month'] = combined.index.month

combined['hour'] = combined.index.hour
## Since the hour of day is cyclical, e.g. 01:00 is equaly far from midnight
## as 23:00 we need to represent this in a meaningful way. We use both sin
## and cos, to make sure that 12:00 != 00:00 (which we cannot prevent if we only
## use sin)
combined['hour_sin'] = combined['hour'].apply(lambda hour: math.sin(2*math.pi*hour/24))
combined['hour_cos'] = combined['hour'].apply(lambda hour: math.cos(2*math.pi*hour/24))

# Some simple model of rush hour
combined['rush_hour'] = combined['hour'].apply(
    lambda hr: min([math.fabs(8-hr), math.fabs(18-hr)])
)
#combined.ix[combined['holiday'] == 1,'rush_hour'] = \
#    combined['hour'].apply(
#        lambda hr: math.fabs(14-hr)
#    )
#data.ix[data['workingday'] == 0,'rush_hour'] = \
#    data['datetime'].apply(
#        lambda i: math.fabs(14-i.hour)
#    )

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
combined['weekday_sin'] = combined['weekday'].apply(lambda weekday: math.sin(2*math.pi*weekday/7))
combined['weekday_cos'] = combined['weekday'].apply(lambda weekday: math.cos(2*math.pi*weekday/7))

# does daylight savings have an impact
start2011 = date(2011, 3, 13)
end2011 = date(2011, 11, 6)
start2012 = date(2012, 3, 11)
end2012 = date(2012, 11, 4)
combined['daylight_savings'] = combined['Date'].apply( \
    lambda d: int( (d >= start2011 and d < end2011) or (d >= start2012 and d < end2012) ) )


combined['temperature'] = combined['temp'] * combined['atemp']

# For some reason the dataset didn't indicate new year's day and christmas
# day as holidays. Therefore we also use this external libraryto check if
# a day is a holiday
cal = Maryland()
holidays = cal.holidays(2011)
holidays += cal.holidays(2012)

holidays = set([dt for (dt, name) in holidays])
combined['holiday'] = combined['Date'].apply(lambda i: int(i in holidays))
validation['holiday'] = [ int(date.date() in holidays) for (date, hour) in validation.index ]

combined.loc[ (combined.holiday == 0) & (combined.weekday < 5) & (combined.workingday == 0), 'workingday' ] = 1

# Was it a holiday yesterday?
combined['holiday_lag'] = combined['Date'].apply(
    lambda i: int(i - timedelta(days=1) in holidays)
    )

# Is it a holiday tomorrow?
combined['holiday_lead'] = combined['Date'].apply(
    lambda i: int(i + timedelta(days=1) in holidays)
    )
combined['almost_holiday'] = combined['Date'].apply(
    lambda i: int(i - timedelta(days=1) in holidays or 
        i + timedelta(days=1) in holidays)
    )

# weekend dates
weekends = set(combined.loc[ (combined['weekday'] >= 5), 'Date' ])

# Was it a weekend yesterday?
combined['weekend_lag'] = combined['Date'].apply(
    lambda i: int(i - timedelta(days=1) in weekends)
    )

# Is it a weekend tomorrow?
combined['weekend_lead'] = combined['Date'].apply(
    lambda i: int(i + timedelta(days=1) in weekends)
    )
combined['almost_weekend'] = combined['Date'].apply(
    lambda i: int(i + timedelta(days=1) in weekends or
        i - timedelta(days=1) in weekends)
    )

def morningWeather(group):
    morning = group.loc[ (group['hour'] >= 7) & (group['hour'] <= 10) ]
    group['morning_weather'] = morning['weather'].mean()
    
    if (morning['weather'].empty):
        group['morning_weather'] = group['weather'].mean()
    return group
   
# deduce type of day from weather i.e. rainy day, sunny day, etc...
combined = combined.groupby('Date').apply(morningWeather)  

combined.drop('Date', axis=1, inplace=True)

combined['weather_lag'] = combined.weather.shift(1)
combined['weather_lag'][0] = combined['weather_lag'][1]
combined['temp_lag'] = combined.temperature.shift(1)
combined['temp_lag'][0] = combined['temp_lag'][1]

combined['season_hr'] = (combined['season'] + combined['hour']/100)
combined['weather_hr'] = (combined['weather_lag'] + combined['hour']/100)
combined['weekday_hr'] = (combined['weekday'] + combined['hour']/100) 

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 

reg_box, reg_lambda = boxcox(training['registered'].values + 1) # Add 1 to be able to transform 0 values
training['registered_box'] = reg_box

cas_box, cas_lambda = boxcox(training['casual'].values + 1) # Add 1 to be able to transform 0 values
training['casual_box'] = cas_box

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')
testing.to_csv('testing-features.csv')
validation.to_csv('validation-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

# separate working from non-working for training, testing & validation sets!
training_holidays = training.loc[ (training['holiday'] == 1) ]
training_weekends = training.loc[ (training['holiday'] == 0) & (training['weekday'] >= 5) ]
training_working = training.loc[ (training['holiday'] == 0) & (training['weekday'] < 5) ]

training_holidays.to_csv('training-holidays-features.csv', sep=',', encoding='utf-8', header=True)
training_weekends.to_csv('training-weekends-features.csv', sep=',', encoding='utf-8', header=True)
training_working.to_csv('training-working-features.csv', sep=',', encoding='utf-8', header=True)

testing_holidays = testing.loc[ (testing['holiday'] == 1) ]
testing_weekends = testing.loc[ (testing['holiday'] == 0) & (testing['weekday'] >= 5) ]
testing_working = testing.loc[ (testing['holiday'] == 0) & (testing['weekday'] < 5) ]

testing_holidays.to_csv('testing-holidays-features.csv', sep=',', encoding='utf-8', header=True)
testing_weekends.to_csv('testing-weekends-features.csv', sep=',', encoding='utf-8', header=True)
testing_working.to_csv('testing-working-features.csv', sep=',', encoding='utf-8', header=True)

validation_holidays = validation.loc[ (validation['holiday'] == 1) ]
validation_weekends = validation.loc[ (validation['holiday'] == 0) & ((validation['weekday'] == 0) | (validation['weekday'] == 6)) ]
validation_working = validation.loc[ (validation['holiday'] == 0) & ((validation['weekday'] >= 1) & (validation['weekday'] <= 5)) ]


# registered holidays
featuresUnused10 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
                    'month', 'weekday', 
                    'saturday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'spring', 'clear_weather', '2012', 
                    '2011', 'heavy_rain', 'summer',
                    'weather_hr', 'afternoon', 'weather_lag', 'dayofyear', 'holiday_lag', 'holiday_lead', 'hour_sin' ] # 

results10 = analyzeMetricNumerical('registered_log10',training_holidays, featuresUnused10)
showFeatureImportanceNumerical(training_holidays, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_holidays[results10['features']], 'registered_log10')
testing_holidays['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
print('rmsle of registered holidays = ', rmsle(validation_holidays['registered'].values, testing_holidays['registered'].values) )

## registered weekends
#featuresUnused10 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
#                    'month', 'year', 'season', 'weekday',
#                    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'spring', 'clear_weather', 
#                    '2012', 
#                    'morning_weather', 'weather_lag', 'dayofyear', 'almost_holiday', 'afternoon', 'daylight_savings' ]
#results10 = analyzeMetricNumerical('registered_log10',training_weekends, featuresUnused10)
#showFeatureImportanceNumerical(training_weekends, results10['features'], 'registered_log10')
#temp10 = predict(results10['model'], testing_weekends[results10['features']], 'registered_log10')
#testing_weekends['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
#print('rmsle of registered weekends = ', rmsle(validation_weekends['registered'].values, testing_weekends['registered'].values) )
#
#
## registered working
#featuresUnused11 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'workingday', 'atemp', 'temp', 'holiday', 'weather',
#                    'month', 'hour_cos', 'weekday', 'season', 'year', 
#                    'spring', 'sunday', 'saturday', 'tuesday', 'fall', #'monday', #'thursday', 'wednesday', #'clear_weather', '2012', 'heavy_rain',
#                    'dayofyear', 'daylight_savings', 'temp_lag', 'weather_lag', 'weekday_sin', 'weekday_cos', 'rush_hour' #, 'morning_weather', 'holiday_lag', 'holiday_lead',
##                    'afternoon', 'weekend_lag'
#                    ]
#results11 = analyzeMetricNumerical('registered_log10',training_working, featuresUnused11)
#showFeatureImportanceNumerical(training_working, results11['features'], 'registered_log10')
#temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_log10')
#testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
#print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )

## registered overall
#featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
#                    'timestamp', 'atemp', 'temp', #'weather', 'season', 
#                    'month', # 'weekday',
#                    'hour_sin', # 'hour_cos', 
#                    'dayofyear', # 'rush_hour', 'weather_lag', 'daylight_savings', 'temp_lag', 
#                    'morning_weather', '2011', 'weekday_cos', # 'weekday_sin', 
#                    'weekday_hr', 'weather_hr', 'season_hr', 
#                    'holiday_lag', 'holiday_lead', 'almost_weekend', 'weekend_lag', 'weekend_lead', # 'almost_holiday', 
#                    ]
#results20 = analyzeMetricNumerical('registered_log10',training, featuresUnused20)
#showFeatureImportanceNumerical(training, results20['features'], 'registered_log10')
#temp20 = predict(results20['model'], testing[results20['features']], 'registered_log10')
##testing['registered'] = np.power((temp20['registered_log10'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#testing['registered'] = np.power( 10, temp20['registered_log10'].values ) - 1
#print('rmsle of registered overall = ', rmsle(validation['registered'].values, testing['registered'].values) )

#testing_regrouped = pd.concat([ testing_holidays, testing_weekends, testing_working ])
#testing_regrouped.sort_index(inplace=True)
#testing_regrouped['count'] = testing_regrouped['registered'] + testing_regrouped['casual']
#testing_regrouped.to_csv('testing-merged.csv')
#valid_regrouped = pd.concat([ validation_holidays, validation_weekends, validation_working ])
#valid_regrouped.sort_index(inplace=True) #by=['timestamp', 'hr'],
#valid_regrouped.to_csv('valid-merged.csv')
#
#print('rmsle of count = ', rmsle(valid_regrouped['count'].values, testing_regrouped['count'].values) )

#testing_regrouped = testing_regrouped[['count']]
#testing_regrouped.to_csv('submission22.csv', sep=',', encoding='utf-8')

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