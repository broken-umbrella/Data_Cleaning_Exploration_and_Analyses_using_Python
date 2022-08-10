# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:07:49 2022

@author: Maruf Ahmed
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.getcwd()
os.chdir('C:/Users\Maruf Ahmed/Desktop/SummerMats/26 May - 2 June/')
os.getcwd()

#Glimpsing, Formatting and Cleaning dataset
adult_man_obesity = pd.read_excel('C:/Users\Maruf Ahmed/Desktop/SummerMats/26 May - 2 June/adult_men_overweight_or_obese_vs_calories.xlsx')
adult_man_obesity.info()

adult_man_obesity['Code'].fillna(0, inplace=True)
adult_man_obesity = adult_man_obesity.drop(adult_man_obesity[adult_man_obesity.Code == 0].index)
adult_man_obesity = adult_man_obesity.drop(adult_man_obesity[adult_man_obesity['Code'].str.contains('OWID')].index)
adult_man_obesity = adult_man_obesity.drop(adult_man_obesity[adult_man_obesity.Year < 2000].index)

adult_man_obesity = adult_man_obesity.drop('Continent', 1)
adult_man_obesity['overweight_obese'].fillna(0, inplace=True)
adult_man_obesity['caloric_supply'].fillna(0, inplace=True)
adult_man_obesity = adult_man_obesity.drop(adult_man_obesity[adult_man_obesity.overweight_obese == 0].index)
adult_man_obesity = adult_man_obesity.drop(adult_man_obesity[adult_man_obesity.caloric_supply == 0].index)

#Analysing adult overweight/obese
adult_man_obesity.groupby('Year').agg({'overweight_obese': ['mean', 'min', 'max']}) #worldwide avg, min and max increased throughout the years
caloric_supply_years = adult_man_obesity.groupby('Year')['caloric_supply'].mean()
caloric_supply_years.sort_values(ascending=False) # Caloric Supply increased throughout the years, except 2014 when decreased a little

mean_obesity = adult_man_obesity.groupby('Entity')['overweight_obese'].mean()
mean_obesity.sort_values(ascending=False) #Samoa, Kuwait and US had highest rate of obesity, Ethiopia and Rwnada had the lowest

caloric_supply = adult_man_obesity.groupby('Entity')['caloric_supply'].mean()
caloric_supply.sort_values(ascending=False) #US, Belgium and Austria had highest caloric supply, Afghanistan and Zambia had the lowest

#Consumption of calory vs obsity rate
merge_obesity_calory = pd.merge(mean_obesity, caloric_supply, on = 'Entity')

max_obesity = adult_man_obesity.groupby('Entity')['overweight_obese'].max()
max_obesity_year = pd.merge(max_obesity, adult_man_obesity, on=['overweight_obese', 'Entity'], how='inner') #Every country witnessed an upward trend except Brunei

#Visualizing adult overweight/obese
calory_sum = adult_man_obesity.groupby('Year')['caloric_supply'].sum()
#calory_years_tab = pd.crosstab(index = adult_man_obesity['Year'], columns = 'CaloryInYears')
calory_sum.plot(kind = 'bar', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))

obesity_sum = adult_man_obesity.groupby('Year')['overweight_obese'].sum()
obesity_sum.plot(kind = 'bar', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))
obesity_sum.plot(kind = 'line', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))
obesity_sum.plot(kind = 'pie', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))

sns.relplot(x = 'caloric_supply', y='overweight_obese', sizes = (1, 100), size='overweight_obese', hue='Year', palette='viridis',data = adult_man_obesity)
plt.ticklabel_format(style='plain', axis='y')



#Cleaning and Analysing the Obesity Death Rate table
obesity_death_rate = pd.read_excel('C:/Users\Maruf Ahmed/Desktop/SummerMats/26 May - 2 June/obesity_death_rate.xlsx')
obesity_death_rate.info()
obesity_death_rate.dropna(how='any', axis=0)

obect_dr = obesity_death_rate.groupby('Entity')['death_rate'].mean()
obect_dr.sort_values(ascending=False) #On average, Fiji and Kiribati had the highest death rate, Timor and Japan had the lowest
max_death_rate = obesity_death_rate.groupby('Entity')['death_rate'].max()
max_death_rate_year = pd.merge(max_death_rate, obesity_death_rate, on=['death_rate', 'Entity'], how='inner')
max_death_rate_year.groupby('Year').count() #55 countries had the highest death rate recorded in 2019, latest year on this dataset

#Visualizing Obesity death rate
odr_sum = obesity_death_rate.groupby('Year')['death_rate'].sum()
odr_sum.plot(kind = 'line', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))

#Cleaning and Analysing the overweight children table
overweight_children = pd.read_excel('C:/Users\Maruf Ahmed/Desktop/SummerMats/26 May - 2 June/overweight_children.xlsx')
overweight_children.info()
overweight_children.dropna(how='any', axis=0)

overweight_ch_avg = overweight_children.groupby('Entity')['overweight_prevalence'].mean()
overweight_ch_avg.sort_values(ascending=False) # Ukraine and Libya had the highest overweight children, Sri Lanka and Niger had the lowest
max_oc = overweight_children.groupby('Entity')['overweight_prevalence'].max()
max_oc_year = pd.merge(max_oc, overweight_children, on=['overweight_prevalence', 'Entity'], how='inner')
max_oc_year.groupby('Year').count() #67 countries had the highest overweight prevalence recorded in 2019, latest year on this dataset

op_sum = overweight_children.groupby('Year')['overweight_prevalence'].sum()
op_sum.plot(kind = 'line', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))

#Cleaning and Analysing the share of deaths table
share_of_deaths_obesity = pd.read_excel('C:/Users\Maruf Ahmed/Desktop/SummerMats/26 May - 2 June/share_of_deaths_obesity.xlsx')
share_of_deaths_obesity.info()
share_of_deaths_obesity.fillna(0, inplace=True)
share_of_deaths_obesity = share_of_deaths_obesity.drop(share_of_deaths_obesity[share_of_deaths_obesity.Code == 0].index)
share_of_deaths_obesity = share_of_deaths_obesity.drop(share_of_deaths_obesity[share_of_deaths_obesity['Code'].str.contains('OWID')].index)
share_of_deaths_obesity = share_of_deaths_obesity.drop(share_of_deaths_obesity[share_of_deaths_obesity['Year'] < 2000].index)


sd_avg = share_of_deaths_obesity.groupby('Entity')['share_of_deaths'].mean()
sd_avg.sort_values(ascending=False)  #Fiji (2005) and Cook Islands (2019) had the highest share of deaths in recorded history, Somalia (2019) and Timor had the lowest
max_sd = share_of_deaths_obesity.groupby('Entity')['share_of_deaths'].max()
max_sd = pd.merge(max_sd, share_of_deaths_obesity, on=['share_of_deaths', 'Entity'], how='inner')
max_sd.groupby('Year').count() #164 countries witnessed the highest share of deaths in 2019, latest year on this dataset

sd_sum = share_of_deaths_obesity.groupby('Year')['share_of_deaths'].sum()
sd_sum.plot(kind = 'pie', fig = (8, 8)).legend(bbox_to_anchor = (1.2, 0.5))

uk_adult_man_obesity = adult_man_obesity.loc[adult_man_obesity['Entity'] == 'United Kingdom']
uk_obesity_death_rate = obesity_death_rate.loc[obesity_death_rate['Entity'] == 'United Kingdom']
uk_share_of_deaths = share_of_deaths_obesity.loc[share_of_deaths_obesity['Entity'] == 'United Kingdom']

uk_obesity_stats = uk_adult_man_obesity.merge(uk_obesity_death_rate, on='Year').merge(uk_share_of_deaths, on='Year') #Calory supply fluctuated over the years, but obesity rate remained constant, UK has been successful in reducing the death rate and share of deaths 

sns.relplot(x = 'caloric_supply', y='overweight_obese', sizes = (1, 100), size='overweight_obese', hue='Year', palette='viridis',data = uk_obesity_stats)
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x = 'overweight_obese', y='death_rate', sizes = (1, 100), size='death_rate', hue='Year', palette='viridis', data = uk_obesity_stats)
plt.ticklabel_format(style='plain', axis='y')














