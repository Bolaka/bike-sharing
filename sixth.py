# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:02:37 2015

@author: bolaka
starts with
rmsle of registered non working =  0.484413769423
rmsle of registered working =  0.351604349497
rmsle of casual non working =  0.590263194079
rmsle of casual working =  0.543205187673
rmsle of count =  0.384341016695
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
combined['Date']= combined['timestamp'].apply(lambda x:x.date())
validation['timestamp'] = [datetime.strptime(x, "%m/%d/%Y") for x in validation.index]

combined['year'] = [str(x.year) for x in combined['timestamp'] ]
dummies = pd.get_dummies(combined['year'])
combined = pd.concat([combined, dummies], axis=1)
combined['year'] = [x.year for x in combined['timestamp'] ]

combined['month'] = [x.month for x in combined['timestamp'] ]
combined['hour'] = [x.hour for x in combined['timestamp'] ]

combined['weekday'] = [x.weekday() for x in combined['timestamp'] ]
combined.loc[ (combined['weekday'] == 0), 'weekday' ] = 'monday'
combined.loc[ (combined['weekday'] == 1), 'weekday' ] = 'tuesday'
combined.loc[ (combined['weekday'] == 2), 'weekday' ] = 'wednesday'
combined.loc[ (combined['weekday'] == 3), 'weekday' ] = 'thursday'
combined.loc[ (combined['weekday'] == 4), 'weekday' ] = 'friday'
combined.loc[ (combined['weekday'] == 5), 'weekday' ] = 'saturday'
combined.loc[ (combined['weekday'] == 6), 'weekday' ] = 'sunday'
dummies = pd.get_dummies(combined['weekday'])
combined = pd.concat([combined, dummies], axis=1)
combined['weekday'] = [x.weekday() for x in combined['timestamp'] ]

# peak hours
combined['peak_hrs'] = 0
combined.loc[ (combined['hour'] >= 7) & (combined['hour'] <= 8), 'peak_hrs' ] = 1
combined.loc[ (combined['hour'] >= 17) & (combined['hour'] <= 18), 'peak_hrs' ] = 1

# midday hours
combined['midday_hrs'] = 0
combined.loc[ (combined['hour'] >= 11) & (combined['hour'] <= 18), 'midday_hrs' ] = 1

# feature engineering
combined[ 'weekend' ] = 0
combined.loc[ (combined['holiday'] == 0) & (combined['workingday'] == 0) ,'weekend'] = 1
combined.loc[ (combined['weekend'] == 1), 'holiday'] = 1
combined['weekday_holiday'] = combined.holiday * (combined.weekday+1)
combined['atemp_cat'] = pd.cut(combined.atemp.values, 6, labels=[1, 2, 3, 4, 5, 6])
combined['temp_cat'] = pd.cut(combined.atemp.values, 6, labels=[1, 2, 3, 4, 5, 6])
combined['hum_cat'] = pd.cut(combined.humidity.values, 4, labels=[1, 2, 3, 4 ])
combined['windspeed_cat'] = pd.cut(combined.windspeed.values, 4, labels=[ 1, 2, 3, 4 ])

## combinations
#combined['working_hr'] = (combined['workingday'] * 100) + combined['hour']
#combined['temp_hr'] = (combined['temp'] / (combined['hour'] + 1))
#combined['season_temp'] = (combined['season'] * combined['temp'])
#combined['season_weather'] = (combined['season'] + combined['weather']/10)

def morningWeather(group):
    morning = group.loc[ (group['hour'] >= 7) & (group['hour'] <= 10) ]
    group['morning_weather'] = morning['weather'].mean()
    
    if (morning['weather'].empty):
        group['morning_weather'] = group['weather'].mean()
    return group
   
# deduce type of day from weather i.e. rainy day, sunny day, etc...
combined = combined.groupby('Date').apply(morningWeather)  
combined.drop('Date', axis=1, inplace=True)

#daywise = days.agg({'weather':'mean', 'casual':'sum', 'registered':'sum', 'workingday':'mean'})
#daywise.to_csv('day_aggregates.csv')

combined['weather_hr'] = (combined['weather'] + combined['hour']/100)
combined['weather_season'] = (combined['weather'] + combined['season']/10)
combined['season_hr'] = (combined['season'] + combined['hour']/100) 
combined['year_season'] = combined['year'] + (combined['season']/10)
combined['atemp_hr'] = (combined['hour'] + combined['atemp_cat']/10) 
combined['season_temp'] = (combined['season'] * combined['temp'])
combined['humidity_temp'] = (combined['temp_cat'] + combined['hum_cat']/10)
combined['weekday_hr'] = (combined['weekday'] + combined['hour']/100) 

# binning of continuous features
#combined['temp2'] = combined['temp']**2
#combined['windspeed_log'] = np.log10(combined['windspeed'].values + 1)
#combined['windspeed2'] = combined['windspeed']**2
#combined['windspeed_sqrt'] = np.sqrt(combined['windspeed'].values)
wind_box, wind_lambda = boxcox(combined['windspeed'].values + 1) # Add 1 to be able to transform 0 values
combined['windspeed_box'] = wind_box

combined['humidity2'] = combined['humidity']**2
combined['atemp2'] = combined['atemp']**2

# separate into training and test sets
training = combined.head(trainingLen)
testing = combined.drop(training.index)

# handle skewed counts with log10
training['registered_log10'] = np.log10(training['registered'].values + 1) 
training['casual_log10'] = np.log10(training['casual'].values + 1) 

combined.to_csv('combined-features.csv')
training.to_csv('training-features.csv')
testing.to_csv('testing-features.csv')

# drop metrics from the testing set
testing.drop(['count','registered','casual'], axis=1, inplace=True)

# separate working from non-working for training, testing & validation sets!
training_nonworking = training.loc[ (training['workingday'] == 0) ]
training_working = training.loc[ (training['workingday'] == 1) ]
training_nonworking.to_csv('training-nonworking-features.csv', sep=',', encoding='utf-8', header=True)
training_working.to_csv('training-working-features.csv', sep=',', encoding='utf-8', header=True)

testing_nonworking = testing.loc[ (testing['workingday'] == 0) ]
testing_working = testing.loc[ (testing['workingday'] == 1) ]
validation_nonworking = validation.loc[ (validation['workingday'] == 0) ]
validation_working = validation.loc[ (validation['workingday'] == 1) ]

# registered non working
featuresUnused10 = [ 'casual','registered','count', 'timestamp', 'workingday', 'registered_log10', 'casual_log10',
                    'season_refrac','holiday', 'temp', 'atemp', 'season', 'hour', 'humidity2', 'weather', 'atemp_cat', 
                    'year', 'month', 'weekend','peak_hrs', 'windspeed_cat', 'temp_cat', 'windspeed_box', 'hum_cat', 
                    'weather_season', '2011', 'weekday', 'tuesday', 'thursday', 'wednesday', 'morning_weather', 
                    'morning_temp' ]
results10 = analyzeMetricNumerical('registered_log10',training_nonworking, featuresUnused10)
showFeatureImportanceNumerical(training_nonworking, results10['features'], 'registered_log10')
temp10 = predict(results10['model'], testing_nonworking[results10['features']], 'registered_log10')
testing_nonworking['registered'] = np.power( 10, temp10['registered_log10'].values ) - 1
print('rmsle of registered non working = ', rmsle(validation_nonworking['registered'].values, testing_nonworking['registered'].values) )

# casual non working
featuresUnused20 = [ 'casual','registered','count', 'timestamp', 'registered_log10', 'casual_log10', 'workingday','peak_hrs',
                    'atemp_cat', 'weather_hr', 'year_season', 'atemp_hr', 'holiday', 'weekend', 'season', 'month',
                    'season_temp', 'humidity2', 'atemp2','atemp', 'windspeed_cat', 'temp_cat', 'windspeed_box', 'hum_cat',
                    'tuesday', 'thursday', 'wednesday', '2012', 'midday_hrs', 'weather_season' ] 
results20 = analyzeMetricNumerical('casual_log10',training_nonworking, featuresUnused20)
showFeatureImportanceNumerical(training_nonworking, results20['features'], 'casual_log10')
temp20 = predict(results20['model'], testing_nonworking[results20['features']], 'casual_log10')
testing_nonworking['casual'] = np.power( 10, temp20['casual_log10'].values )  - 1
print('rmsle of casual non working = ', rmsle(validation_nonworking['casual'].values, testing_nonworking['casual'].values) )

# registered working
featuresUnused11 = [ 'casual','registered','count', 'timestamp', 'workingday', 'registered_log10', 'casual_log10',
                    'atemp_hr', 'season_temp', 'humidity2','weather_hr', 'weekday_holiday', 'windspeed_cat',
                    'atemp_cat','holiday','weekend', 'temp', 'season', 'year', 'month','atemp', 'temp_cat',
                    'windspeed_box', 'hum_cat', '2012', 'midday_hrs', 'weekday', 'weather_season', 'morning_weather' ] 
results11 = analyzeMetricNumerical('registered_log10',training_working, featuresUnused11)
showFeatureImportanceNumerical(training_working, results11['features'], 'registered_log10')
temp11 = predict(results11['model'], testing_working[results11['features']], 'registered_log10')
testing_working['registered'] = np.power( 10, temp11['registered_log10'].values ) - 1
print('rmsle of registered working = ', rmsle(validation_working['registered'].values, testing_working['registered'].values) )


# casual working
featuresUnused21 = [ 'casual','registered','count', 'timestamp', 'registered_log10', 'casual_log10', 'workingday',
                    'peak_hrs', 'weekend', 'weekday_holiday', 'year_season', 'windspeed_cat', 'atemp_hr', 'season_temp', 
                    'humidity2', 'atemp', 'atemp2', 'holiday', 'atemp_cat', 'windspeed', 'temp_cat', 'hum_cat', 'season', 
                    'weather', '2011', 'sunday', 'saturday','tuesday', 'wednesday', 'weekday_hr' ]
results21 = analyzeMetricNumerical('casual_log10',training_working, featuresUnused21)
showFeatureImportanceNumerical(training_working, results21['features'], 'casual_log10')
temp21 = predict(results21['model'], testing_working[results21['features']], 'casual_log10')
testing_working['casual'] = np.power( 10, temp21['casual_log10'].values )  - 1
print('rmsle of casual working = ', rmsle(validation_working['casual'].values, testing_working['casual'].values) )



testing_regrouped = pd.concat([ testing_nonworking, testing_working ])
testing_regrouped.sort_index(inplace=True)
testing_regrouped['count'] = testing_regrouped['registered'] + testing_regrouped['casual']
testing_regrouped.to_csv('testing-merged.csv')
valid_regrouped = pd.concat([ validation_nonworking, validation_working ])
valid_regrouped.sort_index(by=['timestamp', 'hr'],inplace=True)
valid_regrouped.to_csv('valid-merged.csv')

print('rmsle of count = ', rmsle(valid_regrouped['count'].values, testing_regrouped['count'].values) )

#testing_regrouped = testing_regrouped[['count']]
#testing_regrouped.to_csv('submission16.csv', sep=',', encoding='utf-8')

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
#
#groups = training_working.groupby('season')
#plotByGrp(groups, 'hour', 'casual')

#training_nonworking = prepareSeries1(training_nonworking, 'hour', 'season', 'holiday', False, 5)
#res = plotDensityMatrix1(training_nonworking, 1, 'season', 0, 'weekday-hour')