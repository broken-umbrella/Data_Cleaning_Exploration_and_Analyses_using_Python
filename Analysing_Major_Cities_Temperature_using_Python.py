# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 00:57:02 2022

@author: Maruf Ahmed
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

!pip install plotly
!pip install chart_studio

import plotly.tools as tls
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from chart_studio import plotly as py
from plotly.offline import iplot

os.getcwd()
os.chdir('C:/Users\Maruf Ahmed/Desktop/SummerMats/3 June - 10 June')


# Cleaning the dataset
city_temperature = pd.read_csv('C:/Users\Maruf Ahmed/Desktop/SummerMats/3 June - 10 June/city_temperature_dataset.csv')
city_temperature.info()

city_temperature.drop('State', axis = 1, inplace = True)

# Ways to check if there are null values
city_temperature.isnull().values.any() #Returns boolean. False if no nulls, True if nulls exist
city_temperature.isnull().sum() #Returns the number of null rows in each column

null_check = city_temperature.isnull()
tr = 'True'
if tr in null_check:
    print('Nulls exist')
else:
    print('No nulls')

# Finding Uniqe Values
for uniq in city_temperature:
    print(uniq, ':', city_temperature[uniq].nunique()) #32 days and 28 years, means outliers exist

# Finding outliers
city_temperature['Day'].unique() #There are 0s, means outliers
city_temp = city_temperature.drop(city_temperature[city_temperature.Day == 0].index)

city_temperature['Year'].unique() #200, 201 years exist, means outliers
city_temp = city_temp.drop(city_temp[(city_temp.Year == 200) | (city_temp.Year == 201)].index)

city_temp['AvgTemperature'].sort_values()
city_temp['AvgTemperature'].describe()
city_temp.groupby('AvgTemperature').count()

city_temp = city_temp.drop(city_temp[city_temp.AvgTemperature == -99].index)

#Finding and dropping Duplicates
len(city_temp[city_temp.duplicated()])
duplicates = city_temp[city_temp.duplicated()]
city_temp = city_temp.drop_duplicates()

#Converting Farenheit to Celsius
city_temp['AvgTemperature'] = (city_temp['AvgTemperature'] - 32) * (5/9)

#Checking if rows are wrongly named using RegEx
sp_ch_region = city_temp.Region[city_temp.Region.str.contains("[^a-zA-Z]")]
city_temp['Region'].unique()
city_temp['Region'].replace('Australia/South Pacific', 'Oceania', inplace = True)
city_temp['Region'].replace('South/Central America & Carribean', 'South America', inplace = True)

sp_ch_country = city_temp.Country[city_temp.Country.str.contains('[^a-zA-Z -]')]
city_temp['Country'].replace('Myanmar (Burma)', 'Myanmar', inplace = True)

sp_ch_city = city_temp.City[city_temp.City.str.contains('[^a-zA-Z -.]')]
city_temp['City'].replace('Bombay (Mumbai)', 'Mumbai', inplace = True)
city_temp['City'].replace('Chennai (Madras)', 'Chennai', inplace = True)


# Analyses of City Temperature
city_temp['City'].nunique() #321 Cities

city_temp.groupby('Region')['City'].nunique() #Africa 29, Asia 35, Europe 45, Middle East 14, North America 167, Oceania 6, South America 25

city_temp.groupby('Country')['City'].nunique().sort_values() #USA has 154 cities (the most), Canada 10 (second most)

city_temp['Year'].nunique() #Years range 1995-2020

avg_years = np.round(city_temp.groupby('Year')['AvgTemperature'].mean().sort_values(), decimals = 2) #From 15.24 in 1995 to 16.18 in 2019. Peaked in 2016

plt.figure(figsize = (15, 8))
plt.bar(avg_years.index, avg_years.values)
plt.xticks(rotation = 10, size = 15)
plt.yticks(size = 15)
plt.ylabel("Average_Temperature", size = 15)
plt.title("Average Temperature every year",size = 20)
plt.show()


np.round(city_temp.groupby('City')['AvgTemperature'].mean().sort_values(), decimals = 2) #Fairbanks and Ulan-Bator were the coldest,  Port au Prince and Niamey were the hottest.

avg_regions = np.round(city_temp.groupby('Region')['AvgTemperature'].mean().sort_values(), decimals = 2) #Europe and North America was the coldest, Middle East and Africa were the hottest.

plt.figure(figsize = (15, 8))
plt.bar(avg_regions.index, avg_regions.values)
plt.ylabel('Average Temperature', size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.title('Average Temperature of every region', size = 20)
plt.show()

avg_countries = np.round(city_temp.groupby('Country')['AvgTemperature'].mean().sort_values(), decimals = 2) #Mongolia and Iceland coldest, Nigeria and Haiti hottest 

temp_1995 = city_temp[city_temp['Year'] == 1995]
avg_temp_1995 = temp_1995.groupby('City')['AvgTemperature'].mean()

temp_2019 = city_temp[city_temp['Year'] == 2019]
avg_temp_2019 = temp_2019.groupby('City')['AvgTemperature'].mean()

joining_1995_and_2019 = pd.merge(avg_temp_1995, avg_temp_2019, on=['City'], how='inner')
joining_1995_and_2019.rename(columns = {joining_1995_and_2019.columns[0]: 'AvgTemp1995', joining_1995_and_2019.columns[1]: 'AvgTemp2019'}, inplace = True)
joining_1995_and_2019 = joining_1995_and_2019.assign(change_rate = joining_1995_and_2019['AvgTemp2019'] - joining_1995_and_2019['AvgTemp1995']) #Dusanbe experienced the highest rise, Muscat and islamabad witnessed the highest decline 
len(joining_1995_and_2019[joining_1995_and_2019['change_rate'] >= 0]) #225 cities had an increase in temperature, rest had a decrease

avg_months = city_temp.groupby('Month')['AvgTemperature'].mean() #December and January were the coldest months, July and August were the hottest 

plt.figure(figsize = (15, 8))
plt.bar(avg_months.index, avg_months.values)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.ylabel('Average Temperature', size = 20)
plt.title('Average Temperature of every region', size = 20)
plt.show()

region_month_temp = city_temp.groupby(['Region', 'Month'])['AvgTemperature'].mean().sort_values(ascending = True)
#The coldest months in Africa, Asia, Europe, Middle East, and North America are January and December
#The coldest months in Oceania and South America are June and July
#The hottest months in Asia, Europe, Middle East, and North America are July and August
#The hottest months in Africa are September and October

us_temp = city_temp[city_temp['Country'] == 'US']
us_temp['City'].nunique() #154 Different cities
us_temp.groupby('City')['AvgTemperature'].mean().sort_values() #Fairbanks and Anchorage were the coldest, Honolulu and San Juan Puerto Rico were the hottest
us_temp.groupby('Year')['AvgTemperature'].mean() #Avg Temperature increased from 13.43 in 1995 to 13.74 in 2019, 
us_temp.groupby('Month')['AvgTemperature'].mean() #December and january coldest, July and August hottest

ustemp_1995 = us_temp[us_temp['Year'] == 1995]
avg_ustemp_1995 = ustemp_1995.groupby('City')['AvgTemperature'].mean()

ustemp_2019 = us_temp[us_temp['Year'] == 2019]
avg_ustemp_2019 = ustemp_2019.groupby('City')['AvgTemperature'].mean()

join_us = pd.merge(avg_ustemp_1995, avg_ustemp_2019, on = ['City'], how = 'inner')
join_us.rename(columns = {join_us.columns[0]: 'AvgTemp1995', join_us.columns[1]: 'AvgTemp2019'}, inplace = True)
join_us = join_us.assign(change_rate = join_us['AvgTemp2019'] - join_us['AvgTemp1995']) #Fairbanks and Anchorage witnessed highest increase, Rapid City and Denver witnessed highest decrease
len(join_us[join_us['change_rate'] >= 0]) #111 cities had an increase in temperature, rest had a decrease


hottest_cities = city_temp.groupby("City").mean().sort_values(by = "AvgTemperature")[-1:-11:-1]
plt.figure(figsize = (15, 8))
plt.barh(hottest_cities.index, hottest_cities.AvgTemperature)

hottest_countries = city_temp.groupby("Country").mean().sort_values(by = "AvgTemperature")

plt.figure(figsize = (20,8))
plt.bar(hottest_countries.index[-1:-20:-1], hottest_countries.AvgTemperature[-1:-20:-1])
plt.yticks(size = 15)
plt.ylabel("Avgerage Temperature",size = 15)
plt.xticks(rotation = 90,size = 12)
plt.title("The hottest Countries in The world",size = 20)
plt.show()


plt.figure(figsize=(30, 55))
i= 1 # this is for the subplot
for region in city_temp.Region.unique(): # this for loop make it easy to visualize every region with less code
    
    region_data =city_temp[city_temp['Region'] == region]
    final_data= region_data.groupby(city_temp['Month']).mean()['AvgTemperature'].sort_values(ascending=False)

    final_data = pd.DataFrame(final_data)
    final_data = final_data.sort_index()

    final_data = final_data.rename(index = {1:"January",2:"February" ,3:"March" ,4:"April" ,5:"May" ,6:"June" ,7:"July" ,8:"August" ,9:"September" ,10:"October" ,11:"November" ,12:"December" })
    plt.subplot(4,2,i)
    sns.barplot(x=final_data.index,y='AvgTemperature',data=final_data,palette='Paired')
    plt.title(region,size = 20)
    plt.xlabel(None)
    plt.xticks(rotation = 90,size = 18)
    plt.ylabel("Mean Temperature",size = 15)
    i+=1






