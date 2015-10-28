# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:02:15 2015

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
from sklearn.preprocessing import MinMaxScaler

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

# some plots
#training.plot(y='count')
#training.plot(y='registered')
#c = training.plot(y='casual', figsize=(20, 4))

# Plot one week (24 * 7 = 168 datapoints)
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
combined.loc[ (combined['weather'] == 1), 'weather_name' ] = 'dry'
combined.loc[ (combined['weather'] == 2), 'weather_name' ] = 'moist'
combined.loc[ (combined['weather'] == 3), 'weather_name' ] = 'wet'
combined.loc[ (combined['weather'] == 4), 'weather_name' ] = 'vwet'
dummies = pd.get_dummies(combined['weather_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop(['weather_name' ], axis=1, inplace=True) # , 'heavy_rain', 'light_rain', 'mist_cloudy'

# season dummies
combined.loc[ (combined['season'] == 1), 'season_name' ] = 'spring'
combined.loc[ (combined['season'] == 2), 'season_name' ] = 'summer'
combined.loc[ (combined['season'] == 3), 'season_name' ] = 'fall'
combined.loc[ (combined['season'] == 4), 'season_name' ] = 'winter'
dummies = pd.get_dummies(combined['season_name'])
combined = pd.concat([combined, dummies], axis=1)
combined.drop([ 'season_name' ], axis=1, inplace=True) # , 'spring', 'winter', 'fall'

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
    lambda hr: math.fabs(14-hr)
)
#combined.ix[combined['holiday'] == 1,'rush_hour'] = \
#    combined['hour'].apply(
#        lambda hr: math.fabs(14-hr)
#    )
#data.ix[data['workingday'] == 0,'rush_hour'] = \
#    data['datetime'].apply(
#        lambda i: math.fabs(14-i.hour)
#    )

#combined['weekday'] = combined.index.weekday
#combined.loc[ (combined['weekday'] == 0), 'weekday' ] = 'monday'
#combined.loc[ (combined['weekday'] == 1), 'weekday' ] = 'tuesday'
#combined.loc[ (combined['weekday'] == 2), 'weekday' ] = 'wednesday'
#combined.loc[ (combined['weekday'] == 3), 'weekday' ] = 'thursday'
#combined.loc[ (combined['weekday'] == 4), 'weekday' ] = 'friday'
#combined.loc[ (combined['weekday'] == 5), 'weekday' ] = 'saturday'
#combined.loc[ (combined['weekday'] == 6), 'weekday' ] = 'sunday'
#dummies = pd.get_dummies(combined['weekday'])
#combined = pd.concat([combined, dummies], axis=1)

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
#combined['temperature'] = combined['temp'] * combined['temp']

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

## Was it a holiday yesterday?
#combined['holiday_lag'] = combined['Date'].apply(
#    lambda i: int(i - timedelta(days=1) in holidays)
#    )
#
## Is it a holiday tomorrow?
#combined['holiday_lead'] = combined['Date'].apply(
#    lambda i: int(i + timedelta(days=1) in holidays)
#    )
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

## Is it a weekend tomorrow?
#combined['weekend_lead'] = combined['Date'].apply(
#    lambda i: int(i + timedelta(days=1) in weekends)
#    )
#combined['almost_weekend'] = combined['Date'].apply(
#    lambda i: int(i + timedelta(days=1) in weekends or
#        i - timedelta(days=1) in weekends)
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

combined['weather_lead'] = combined.weather.shift(-1)
combined.loc[ combined.tail(1).index, 'weather_lead' ] = combined.tail(2)['weather_lead'].values[0]

combined['weather_lag'] = combined.weather.shift(1)
combined['weather_lag'][0] = combined['weather_lag'][1]
combined['temp_lag'] = combined.temperature.shift(1)
combined['temp_lag'][0] = combined['temp_lag'][1]
combined['temp_lead'] = combined.temperature.shift(-1)
combined.loc[ combined.tail(1).index, 'temp_lead' ] = combined.tail(2)['temp_lead'].values[0]

#month_lag = (combined.month - 1) % 3
#combined['season_variation'] = (combined['season'] + month_lag/3.0)
combined['season_hr'] = (combined['season'] + combined['hour']/100)
combined['weather_hr'] = (combined['weather_lag'] + combined['hour']/100)
combined['weekday_hr'] = (combined['weekday'] + combined['hour']/100) 

combined['summer_weather'] = combined.summer * combined.weather
combined['fall_weather'] = combined.fall * combined.weather
combined['spring_weather'] = combined.spring * combined.weather
combined['winter_weather'] = combined.winter * combined.weather
#combined['summer_temp'] = combined.summer * combined.temp
#combined['fall_temp'] = combined.fall * combined.temp
#combined['spring_temp'] = combined.spring * combined.temp
#combined['winter_temp'] = combined.winter * combined.temp

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

## lag plot
#from pandas.tools.plotting import lag_plot
#names = combined.columns
#for i in names:
#    print(i)
#    plt.figure()
#    plt.title(i)
#    lag_plot(combined[i])

training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 
#training['count_log10'] = np.log10(training['count'].values + 1) 

reg_box, reg_lambda = boxcox(training['registered'].values + 1) # Add 1 to be able to transform 0 values
training['registered_box'] = reg_box

cas_box, cas_lambda = boxcox(training['casual'].values + 1) # Add 1 to be able to transform 0 values
training['casual_box'] = cas_box

#c_box, c_lambda = boxcox(training['count'].values + 1) # Add 1 to be able to transform 0 values
#training['count_box'] = c_box

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')
testing.to_csv('testing-features.csv')
validation.to_csv('validation-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

#dataNorm = training
#scaler = MinMaxScaler()
##dataNorm['casual'] = scaler.fit_transform(dataNorm['casual'])
##dataNorm['temp'] = scaler.fit_transform(dataNorm['temp'])
##dataNorm['humidity'] = scaler.fit_transform(dataNorm['humidity'])
##dataNorm['windspeed'] = scaler.fit_transform(dataNorm['windspeed'])
#months = pd.groupby(dataNorm,by=[dataNorm.season, dataNorm.index.year, dataNorm.index.month, dataNorm.index.week]) # 
#
#for index, group in months:
##    print(index)
#    if index[1] == 2011 and index[2] == 4: #index[0] == 1 and 
##        print(group)   
##        group.plot(y=[ 'casual', 'temp' ], figsize=(21, 4)) # 'humidity', , 'windspeed'
#        group.casual.plot(figsize=(21, 4), label='Casual', legend=True)
#        group.weather.plot(secondary_y=True, figsize=(21, 4), label='Weather', legend=True)
##        group.temp.plot(secondary_y=True, figsize=(21, 4), label='Temp', legend=True)
##        group['rush_hour'].plot(secondary_y=True, figsize=(21, 4), label='Hour', legend=True)


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


## casual holidays
#featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
#                    'timestamp', 'workingday', 'atemp', 'temp', 'holiday', #'weather',
#                    'month', 'hour', # 'weekday', 
#                    'rush_hour', 'weekday_sin', 'weekday_cos', '2011', 'morning_weather',
#                    'temp_lag', 'dayofyear', 'holiday_lead', 'almost_holiday', #'hour_sin', 'hour_cos', 'weather_lag', 
#                    'weekend_lag', 'weekend_lead', 'almost_weekend', 'daylight_savings',
#                    'weekday_hr', 'season_hr', 'weather_hr'
#                    ]
#results20 = analyzeMetricNumerical('casual_box',training_holidays, featuresUnused20)
#showFeatureImportanceNumerical(training_holidays, results20['features'], 'casual_box')
#temp20 = predict(results20['model'], testing_holidays[results20['features']], 'casual_box')
#testing_holidays['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual holidays = ', rmsle(validation_holidays['casual'].values, testing_holidays['casual'].values) )
#
#
## casual weekends
#featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'timestamp', 'workingday', 'atemp', 'temp', 'holiday', 'weather',
#                    'month', 'weekday', 'year', 
#                    'rush_hour', 'morning_weather', # 'weekday_cos', 'weekday_sin', '2011', 
#                    'hour_sin', 'hour_cos', 'holiday_lead', #'weather_lag', 'dayofyear', 'almost_holiday', 
#                    'weekend_lag', 'weekend_lead', 'daylight_savings', # 'almost_weekend', 
#                    'weekday_hr' #, 'season_hr', 'weather_hr'
#                    ]
#results20 = analyzeMetricNumerical('casual_box',training_weekends, featuresUnused20)
#showFeatureImportanceNumerical(training_weekends, results20['features'], 'casual_box')
#temp20 = predict(results20['model'], testing_weekends[results20['features']], 'casual_box')
#testing_weekends['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual weekends = ', rmsle(validation_weekends['casual'].values, testing_weekends['casual'].values) )
#
## casual working
#featuresUnused21 = [ 'casual', 'registered', 'count', 'registered_log10', 'casual_log10', 'registered_box', 'casual_box', 
#                    'timestamp', 'workingday', 'atemp', 'temp', 'holiday', 'weather',
#                    'hour', 'year', # 'month', 'weekday', 
#                    'rush_hour', 'morning_weather', 'weekday_cos', 'weekday_sin', # '2011', 
#                    'holiday_lead', 'temp_lag', 'almost_holiday', # 'hour_sin', 'hour_cos', 'weather_lag', 'dayofyear', 
#                    'weekend_lag', 'weekend_lead', 'daylight_savings', 'almost_weekend', 
#                    'weekday_hr', 'season_hr', 'weather_hr' 
#                    ]
#results21 = analyzeMetricNumerical('casual_box',training_working, featuresUnused21)
#showFeatureImportanceNumerical(training_working, results21['features'], 'casual_box')
#temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_box')
#testing_working['casual'] = np.power((temp21['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
#print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )


# casual overall
featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
                    'timestamp', 'atemp', 'temp', 'temperature', 'weather', 
                    'hour', 'month', # 'hour_sin', 'hour_cos', 
                    'dayofyear', 'season', 'season_hr', #'weather_lag', 'temp_lag', 'daylight_savings', 'rush_hour', 
                    'morning_weather', 'weekday_sin', 'weekday_cos', '2011',
                    'dry', 'moist', 'wet', 'vwet', 'spring', 'summer', 'fall', 'winter'
                    ]
results20 = analyzeMetricNumerical('casual_box',training, featuresUnused20, True)
showFeatureImportanceNumerical(training, results20['features'], 'casual_box')
temp20 = predict(results20['model'], testing[results20['features']], 'casual_box')
testing['casual'] = np.power((temp20['casual_box'].values * cas_lambda) + 1, 1 / cas_lambda) - 1
print('rmsle of casual overall = ', rmsle(validation['casual'].values, testing['casual'].values) )


## count overall
#featuresUnused20 = [ 'casual', 'registered', 'count', 'registered_box', 'casual_box', 'registered_log10', 'casual_log10',
#                    'count_box', 'count_log10',
#                    'timestamp', 'atemp', 'temp', 'temperature', 'weather', 
#                    'hour', 'month', # 'hour_sin', 'hour_cos', 
#                    'dayofyear', 'season', 'season_hr', #'weather_lag', 'temp_lag', 'daylight_savings', 'rush_hour', 
#                    'morning_weather', 'weekday_sin', 'weekday_cos', '2011'
#                    ]
#results20 = analyzeMetricNumerical('count_box',training, featuresUnused20, True)
#showFeatureImportanceNumerical(training, results20['features'], 'count_box')
#temp20 = predict(results20['model'], testing[results20['features']], 'count_box')
#testing['count'] = np.power((temp20['count_box'].values * c_lambda) + 1, 1 / c_lambda) - 1
#print('rmsle of count overall = ', rmsle(validation['count'].values, testing['count'].values) )



#testing_regrouped = pd.concat([ testing_holidays, testing_weekends, testing_working ])
#testing_regrouped.sort_index(inplace=True)
#print('rmsle of casual overall merged = ', rmsle(validation['casual'].values, testing_regrouped['casual'].values) )

p = { 'datetime' : pd.Series(testing.index),
#        'casual_actual' : pd.Series(validation['casual'].values), 
        'casual_pred' : pd.Series(testing['casual'].values) }
predictions = pd.DataFrame(p)
predictions = predictions.set_index('datetime')



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