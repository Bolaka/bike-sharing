# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:28:39 2015

@author: bolaka

@author: bolaka
starts with
rmsle of registered non working =  0.450681332253
rmsle of registered working =  0.343219495225
rmsle of casual non working =  0.573892309621
rmsle of casual working =  0.53562665465
rmsle of count =  0.368635688343
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

# extract the date-timestamps
#combined['timestamp'] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in combined.index]
#validation['timestamp'] = [datetime.strptime(x, "%m/%d/%Y") for x in validation.index]
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
combined.drop(['weather_name'], axis=1, inplace=True) # , 'mist_cloudy', 'light_rain'

# season dummies
combined.loc[ (combined['season'] == 1), 'season_name' ] = 'spring'
combined.loc[ (combined['season'] == 2), 'season_name' ] = 'summer'
combined.loc[ (combined['season'] == 3), 'season_name' ] = 'fall'
combined.loc[ (combined['season'] == 4), 'season_name' ] = 'winter'
dummies = pd.get_dummies(combined['season_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop(['season_name'], axis=1, inplace=True) # , 'spring', 'summer'

combined['dayofyear'] = [int(x.strftime('%j')) for x in combined.index ] 
combined['year'] = [str(x) for x in combined.index.year ]
dummies = pd.get_dummies(combined['year'])
combined = pd.concat([combined, dummies], axis=1)
combined['year'] = combined.index.year

combined['month'] = combined.index.month

combined['hour'] = combined.index.hour
# Since the hour of day is cyclical, e.g. 01:00 is equaly far from midnight
# as 23:00 we need to represent this in a meaningful way. We use both sin
# and cos, to make sure that 12:00 != 00:00 (which we cannot prevent if we only
# use sin)
combined['hour_sin'] = combined['hour'].apply(lambda hour: math.sin(2*math.pi*hour/24))
combined['hour_cos'] = combined['hour'].apply(lambda hour: math.cos(2*math.pi*hour/24))

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

# does daylight savings have an impact
#combined['daylight_savings']  = 0
start2011 = date(2011, 3, 13)
end2011 = date(2011, 11, 6)
start2012 = date(2012, 3, 11)
end2012 = date(2012, 11, 4)
combined['daylight_savings'] = combined['Date'].apply( \
    lambda d: int( (d >= start2011 and d < end2011) or (d >= start2012 and d < end2012) ) )


combined['temperature'] = combined['temp'] * combined['atemp']

## weekday cycle
#combined.loc[ (combined['weekday'] >= 5), 'weekday_cycle' ] = 1 # sundays & saturdays
#combined.loc[ (combined['weekday'] == 0) | (combined['weekday'] == 4), 'weekday_cycle' ] = 2 # mondays & fridays
#combined.loc[ (combined['weekday'] == 1) | (combined['weekday'] == 3), 'weekday_cycle' ] = 3 # tuesdays & thursdays
#combined.loc[ (combined['weekday'] == 2), 'weekday_cycle' ] = 4 # wednesdays


## Some simple model of rush hour
#combined['rush_hour'] = combined.hour.apply(
#    lambda i: min([math.fabs(8-i), math.fabs(18-i)])
#)
#combined.ix[ combined['holiday'] == 1, 'rush_hour'] = \
#    combined.hour.apply(
#        lambda i: math.fabs(14-i)
#    )
#combined.ix[ (combined['holiday'] == 0 & (combined['weekday'] >= 5) ), 'rush_hour'] = \
#    combined.hour.apply(
#        lambda i: math.fabs(14-i)
#    )


## peak hours
#combined['peak_hrs'] = 0
#combined.loc[ (combined['hour'] >= 7) & (combined['hour'] <= 8), 'peak_hrs' ] = 1
#combined.loc[ (combined['hour'] >= 17) & (combined['hour'] <= 18), 'peak_hrs' ] = 1

## midday hours
#combined['midday_hrs'] = 0
#combined.loc[ (combined['hour'] >= 11) & (combined['hour'] <= 18), 'midday_hrs' ] = 1

combined['afternoon'] = 0
combined.loc[ (combined['hour'] >= 11) & (combined['hour'] <= 15), 'afternoon' ] = 1

#combined['daylight'] = combined.hour.apply(lambda h: int(h >= 7 and h <= ) )


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

# For some reason the dataset didn't indicate new year's day and christmas
# day as holidays. Therefore we also use this external libraryto check if
# a day is a holiday
cal = Maryland()
holidays = cal.holidays(2011)
holidays += cal.holidays(2012)

holidays = set([dt for (dt, name) in holidays])
combined['holiday'] = combined['Date'].apply(lambda i: int(i in holidays))
validation['holiday'] = [ int(date.date() in holidays) for (date, hour) in validation.index ]
#print(validation['holiday'].sum())

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

## workingday dates
#workingdays = set(combined.loc[ (combined['holiday'] == 0) & (combined['weekday'] < 5), 'Date' ])
#
## Was it a workingday yesterday?
#combined['working_lag'] = combined['Date'].apply(
#    lambda i: int(i - timedelta(days=1) in workingdays)
#    )
#
## Is it a workingday tomorrow?
#combined['working_lead'] = combined['Date'].apply(
#    lambda i: int(i + timedelta(days=1) in workingdays)
#    )
#combined['almost_working'] = combined['Date'].apply(
#    lambda i: int(i + timedelta(days=1) in workingdays or
#        i - timedelta(days=1) in workingdays)
#    )

def morningWeather(group):
    morning = group.loc[ (group['hour'] >= 7) & (group['hour'] <= 10) ]
    group['morning_weather'] = morning['weather'].mean()
    
    if (morning['weather'].empty):
        group['morning_weather'] = group['weather'].mean()
    return group
   
# deduce type of day from weather i.e. rainy day, sunny day, etc...
combined = combined.groupby('Date').apply(morningWeather)  
combined.drop('Date', axis=1, inplace=True)

#combined[ 'weekend'] = 0
#combined.loc[ (combined['holiday'] == 0) & (combined['workingday'] == 0) ,'weekend'] = 1
#combined['weekend_lag'] = combined.weekend.shift(24)
#combined['weekend_lag'][:24] = 0
#combined.drop('weekend', axis=1, inplace=True)

combined['weather_lag'] = combined.weather.shift(1)
combined['weather_lag'][0] = combined['weather_lag'][1]

#month_lag = (combined.month - 1) % 3
#combined['season_variation'] = (combined['season'] + month_lag/3.0)
combined['weather_hr'] = (combined['weather_lag'] + combined['hour']/100)
#combined['weather_season'] = (combined['weather'] + combined['season']/10)
#combined['season_hr'] = (combined['season'] + combined['hour']/100) 
#combined['year_season'] = combined['year'] + (combined['season']/10)
#combined['atemp_hr'] = (combined['hour'] + combined['atemp_cat']/10) 
#combined['season_temp'] = (combined['season'] * combined['temp'])
#combined['humidity_temp'] = (combined['temp_cat'] + combined['hum_cat']/10) - nopes, no luck!
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

results10 = analyzeMetricNumerical('registered_log10',training_holidays, featuresUnused10, False)
showFeatureImportanceNumerical(training_holidays, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_holidays[results10['features']], 'registered_log10')
testing_holidays['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
print('rmsle of registered holidays = ', rmsle(validation_holidays['registered'].values, testing_holidays['registered'].values) )



## casual holidays
#featuresUnused20 = [ 'casual','registered','count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
#                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
#                    'month', 'weekday', 
#                    'saturday', 'spring', 'clear_weather', '2012', '2011', 'heavy_rain', 'tuesday', 'thursday', 'summer',
##                    'monday', 'wednesday', 'friday',
#                    'weather_lag', 'dayofyear', 'holiday_lead', 'almost_holiday', 'hour_sin', 'hour_cos', 
#                    'weekend_lag', 'weekend_lead', 'daylight_savings' ]
#results20 = analyzeMetricNumerical('casual_box',training_holidays, featuresUnused20)
#showFeatureImportanceNumerical(training_holidays, results20['features'], 'casual_box')
#temp20 = predict(results20['model'], testing_holidays[results20['features']], 'casual_box')
#testing_holidays['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual holidays = ', rmsle(validation_holidays['casual'].values, testing_holidays['casual'].values) )



# registered weekends
featuresUnused10 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
                    'month', 'year', 'season', 'weekday',
                    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'spring', 'clear_weather', 
                    '2012', 
                    'morning_weather', 'weather_lag', 'dayofyear', 'almost_holiday', 'afternoon', 'daylight_savings' ]
results10 = analyzeMetricNumerical('registered_log10',training_weekends, featuresUnused10, False)
showFeatureImportanceNumerical(training_weekends, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_weekends[results10['features']], 'registered_log10')
testing_weekends['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
print('rmsle of registered weekends = ', rmsle(validation_weekends['registered'].values, testing_weekends['registered'].values) )



## casual weekends
#featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'timestamp', 'workingday', 'atemp', 'temp', 'weather', 'holiday',
#                    'month', 'year',
#                    'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'spring', 'clear_weather', 
#                    '2012', 
#                    'weather_hr', 'weekday', 'morning_weather', 'holiday_lag', 'holiday_lead', 'almost_holiday',
#                    'hour_sin', 'hour_cos', 'weekend_lag', 'weekend_lead' ]
#results20 = analyzeMetricNumerical('casual_box',training_weekends, featuresUnused20)
#showFeatureImportanceNumerical(training_weekends, results20['features'], 'casual_box')
#temp20 = predict(results20['model'], testing_weekends[results20['features']], 'casual_box')
#testing_weekends['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual weekends = ', rmsle(validation_weekends['casual'].values, testing_weekends['casual'].values) )


# registered working
featuresUnused11 = [ 'casual','registered','count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
                    'timestamp', 'workingday', 'atemp', 'temp', 'holiday', 'weather',
                    'month', 'year', 'season', 'weekday',
                    'sunday', 'saturday', 'monday', 'spring', 'clear_weather', '2012', 'heavy_rain',
                    #'wednesday', 'thursday', 'tuesday',
                    'morning_weather', 'weather_lag', 'dayofyear', 'holiday_lag', 'holiday_lead', 'hour_cos',
                    'afternoon', 'weekend_lag', 'daylight_savings']
results11 = analyzeMetricNumerical('registered_log10',training_working, featuresUnused11, False)
showFeatureImportanceNumerical(training_working, results11['features'], 'registered_log10')
temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_log10')
testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )

## casual working
#featuresUnused21 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'timestamp', 'workingday', 'atemp', 'temp', 'holiday', 'weather',
#                    'weekday', 
#                    'sunday', 'saturday', 'spring', 'clear_weather', '2012', 
#                    'morning_weather', 'weather_lag', 'holiday_lag', 'holiday_lead', 'weekend_lag', 'weekend_lead', 'daylight_savings' ]
#results21 = analyzeMetricNumerical('casual_box',training_working, featuresUnused21)
#showFeatureImportanceNumerical(training_working, results21['features'], 'casual_box')
#temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_box')
#testing_working['casual'] = np.power((temp21['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )


## casual overall
#featuresUnused20 = [ 'casual','registered','count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
#                    'timestamp', 'atemp', 'temp', 'weather', 
#                    'month', 'weekday', 
#                    'saturday', 'spring', 'clear_weather', '2012', '2011', 'heavy_rain', 'tuesday', 'thursday', 'summer',
##                    'monday', 'wednesday', 'friday',
#                    'weather_lag', 'dayofyear', 'holiday_lead', 'almost_holiday', 'hour_sin', 'hour_cos', 
#                    'weekend_lag', 'weekend_lead' ]
#results20 = analyzeMetricNumerical('casual_box',training, featuresUnused20)
#showFeatureImportanceNumerical(training, results20['features'], 'casual_box')
#temp20 = predict(results20['model'], testing[results20['features']], 'casual_box')
#testing['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual overall = ', rmsle(validation['casual'].values, testing['casual'].values) )



testing_regrouped = pd.concat([ testing_holidays, testing_weekends, testing_working ])
testing_regrouped.sort_index(inplace=True)
print('rmsle of registered overall = ', rmsle(validation['registered'].values, testing_regrouped['registered'].values) )

predictions['registered_pred'] = testing_regrouped['registered']
predictions['count'] = predictions['registered_pred'] + predictions['casual_pred']

print('rmsle of count = ', rmsle(validation['count'].values, predictions['count'].values) )

#testing_regrouped['count'] = testing_regrouped['registered'] + testing_regrouped['casual']
#testing_regrouped.to_csv('testing-merged.csv')
#valid_regrouped = pd.concat([ validation_holidays, validation_weekends, validation_working ])
#valid_regrouped.sort_index(inplace=True) #by=['timestamp', 'hr'],
#valid_regrouped.to_csv('valid-merged.csv')
#
#print('rmsle of count = ', rmsle(valid_regrouped['count'].values, testing_regrouped['count'].values) )

predictions = predictions[['count']]
predictions.to_csv('submission23.csv', sep=',', encoding='utf-8')

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