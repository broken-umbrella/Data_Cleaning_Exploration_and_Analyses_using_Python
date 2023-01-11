
import pandas as pd
import numpy as np
import seaborn as sns

fashion_data = pd.read_csv('D:/fashion_dataset.csv')
fashion_brand = pd.read_csv('D:/fashion_brand_details.csv')


fashion_data.info()           
fashion_brand.info()


#deleting rows where p_id and name are null and keeping non-null values

fashion_data_updated = fashion_data.dropna(subset=['p_id', 'name'])
    
#p_id should be integer
fashion_data_updated = fashion_data_updated.astype({'p_id': int})

import matplotlib as mpl
import matplotlib.pyplot as plt

#dropping duplicate values based on p_id
fashion_data_updated = fashion_data_updated.drop_duplicates(subset=['p_id'])


#changing datatype of P_id, couldn't change ratingCount as null values exist
fashion_data_updated = fashion_data_updated.astype({'p_id': int})

#keeping only two decimal places for avg_rating

fashion_data_updated['avg_rating'].head()
fashion_data_updated = fashion_data_updated.round({'avg_rating': 2})
fashion_data_updated['avg_rating'].head()


#checking if brand names are null, populating them
fashion_data_updated.loc[fashion_data_updated['brand'].isnull()]
fashion_data_updated.loc[fashion_data_updated.p_id == 7687291,'brand'] = 'SASSAFRAS'
fashion_data_updated.loc[fashion_data_updated.p_id == 13847264,'brand'] = 'NEUDIS'
fashion_data_updated.loc[fashion_data_updated.p_id == 12414526,'brand'] = 'Biba'


#outliers

sns.boxplot(fashion_data_updated['avg_rating'])



#naming convention

import re
fashion_data_updated.name.str.contains('[^A-Z0-9a-z&%,. -]')
fashion_data_updated.colour.str.contains('[^A-Z0-9a-z&,. -]')
fashion_data_updated.brand.str.contains('[^A-Z0-9a-z&,. -]')


#asking questions from tableau to validate and qa called data validation
sns.lineplot(fashion_data_updated['price'])

#merging datasets
mergedData = pd.merge(fashion_data_updated, fashion_brand, how = 'left', left_on='brand', right_on = 'brand_name')
fashion_data_updated['brand'].count()
#for viz first do python corr, ratingC, avg_r, etc then tableau

pearson_corr = fashion_data_updated.corr(method = 'pearson')
kendall_corr = fashion_data_updated.corr(method = 'kendall')
spearman_corr = fashion_data_updated.corr(method = 'spearman')

sns.heatmap(pearson_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
sns.heatmap(kendall_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')
sns.heatmap(spearman_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Reds')

sns.lineplot(fashion_data_updated['ratingCount'])





import pandas as pd
import numpy as np
import seaborn as sns 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans
                
                
                

sales_data = pd.read_csv('D:/Big Data Assignment/sales.csv')
sales_data.info()
sales_data['order_date'] = pd.to_datetime(sales_data['order_date'])

sales_data = sales_data[sales_data.price != 0] #100 rows deleted
sales_data['total'] = sales_data['qty_ordered'] * sales_data['price']





age_total = sales_data.loc[:, ['age', 'total']].values
    
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(age_total)
    wcss.append(kmeans.inertia_)
plt.figure(figsize = (12, 6))
plt.grid()
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.plot(range(1, 10), wcss, linewidth = 2, color = 'red', marker = '8')
plt.show()
 
kmeans = KMeans(n_clusters = 4)
label = kmeans.fit_predict(age_total)
print(label)

plt.figure(figsize=(8,8))
plt.scatter(age_total[label==0,0], age_total[label==0,1], s=50, c='cyan', label='Cluster 1')
plt.scatter(age_total[label==1,0], age_total[label==1,1], s=50, c='#1f77b4', label='Cluster 2')
plt.scatter(age_total[label==2,0], age_total[label==2,1], s=50, c='blue', label='Cluster 3')
plt.scatter(age_total[label==3,0], age_total[label==3,1], s=50, c='#ff7f0e', label='Cluster 4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'black')
plt.xlabel('Age')
plt.ylabel('Total Purchase Amount')
plt.show()



sales_weekly.head()

def adfuller_test(sale):
    result=adfuller(sale)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
    else:
        print("Weak evidence against null hypothesis")

 

adfuller_test(sales_weekly['total'])   
sales_weekly.plot()



sales_weekly['Difference']=sales_weekly['total']-sales_weekly['total'].shift(4)
adfuller_test(sales_weekly['Difference'].dropna())


size=int(len(sales_weekly)*0.66)
X_train,X_test=sales_weekly[0:size],sales_weekly[size:len(sales_weekly)]
    
smax_model=SARIMAX(sales_weekly['total'],
                   order=(1,1,0),
                   seasonal_order=(0,1,1,4))
res=smax_model.fit()
res.summary()
    
start_index=0
end_index=len(X_train)-1
train_prediction=res.predict(start_index,end_index)
    
st_index=len(X_train)
ed_index=len(sales_weekly)-1
prediction=res.predict(st_index,ed_index)
    
    
    
plt.figure(figsize=(10, 6))
train_prediction.plot(legend=True)
X_train['total'].plot(legend=True)

print('Absolute Error:', metrics.mean_absolute_error(X_train, train_prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(X_train, train_prediction)))

plt.figure(figsize=(10, 4))
prediction.plot(legend=True)
X_test['total'].plot(legend=True)

print('Absolute Error:', metrics.mean_absolute_error(X_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(X_test, prediction)))








# plt.figure(figsize=(10,4))

# X_train['sales'].plot(label="Training",color='green')
# train_prediction.plot(legend=True)
# X_test['sales'].plot(label="Test",color='blue')
# prediction.plot(legend=True)
# forecast.plot(label="Forecast",color="red")
# plt.legend(loc="lower right")




























