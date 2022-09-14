import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.getcwd()
os.chdir('C:/Users\Maruf Ahmed/Desktop/Data Science/Assignment_Two/')
os.getcwd()
miamihousing = pd.read_csv('C:/Users\Maruf Ahmed/Desktop/Data Science/Assignment_Two/miami_housing.csv')

miamihousing.info() #glimpse equivalent
miamihousing = miamihousing.rename(columns = {'age': 'AGE', 'avno60plus': 'AVNO60PLUS', 'month_sold': 'MONTH_SOLD', 'structure_quality': 'STRUCTURE_QUALITY'})
miamihousing['AVNO60PLUS'] = pd.Categorical(miamihousing.AVNO60PLUS)
miamihousing['MONTH_SOLD'] = pd.Categorical(miamihousing.MONTH_SOLD)
miamihousing['STRUCTURE_QUALITY'] = pd.Categorical(miamihousing.STRUCTURE_QUALITY)

miamihousing.info()

miamihousing.head() 
descr = miamihousing.describe()

miamihousing_needed_columns = miamihousing[['SALE_PRC', 'LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'AGE', 'MONTH_SOLD', 'STRUCTURE_QUALITY', 'AVNO60PLUS']]

#compare floor vs land area, 

landvfloor = miamihousing_needed_columns.loc[miamihousing_needed_columns['TOT_LVG_AREA'] > miamihousing_needed_columns['LND_SQFOOT']]
miamihousing_needed_columns = miamihousing_needed_columns.drop(miamihousing_needed_columns[miamihousing_needed_columns.TOT_LVG_AREA > miamihousing_needed_columns.LND_SQFOOT].index)


miamihousing_needed_columns.groupby('MONTH_SOLD').count()
month_viz_tab = pd.crosstab(index = miamihousing_needed_columns['MONTH_SOLD'], columns='MONTH_COUNT')
month_viz_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

strqlty_viz_tab = pd.crosstab(index = miamihousing_needed_columns['STRUCTURE_QUALITY'], columns='STR_QLTY')
strqlty_viz_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

avon_viz_tab = pd.crosstab(index = miamihousing_needed_columns['AVNO60PLUS'], columns='AVNO60PLUS')
avon_viz_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

#visualizing individual column
for col in miamihousing_needed_columns:
    miamihousing_needed_columns[col].hist(bins=25)
    plt.xlabel(col)
    plt.show()
    

houses_sold_each_month_tab = pd.crosstab(index = miamihousing_needed_columns['MONTH_SOLD'], columns='HOUSES_SOLD')
houses_sold_each_month_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

sale_price_sum = miamihousing_needed_columns.groupby('MONTH_SOLD')['SALE_PRC'].sum()
sale_price_sum.plot(kind='line', x='MONTH_SOLD', y='SALE_PRC', fig=(10, 10))
plt.ticklabel_format(style='plain', axis='y')

houses_sold_each_str_qlty_tab = pd.crosstab(index = miamihousing_needed_columns['STRUCTURE_QUALITY'], columns='HOUSES_SOLD')
houses_sold_each_str_qlty_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

houses_sold_each_avno_tab = pd.crosstab(index = miamihousing_needed_columns['AVNO60PLUS'], columns='HOUSES_SOLD')
houses_sold_each_avno_tab.plot(kind='bar', fig=(8, 8)).legend(bbox_to_anchor=(1.2, 0.5))

miamihousing_needed_columns.groupby('STRUCTURE_QUALITY').count()
m_des = miamihousing_needed_columns.describe()

miamihousing_needed_columns['SALE_PRC'].mean()
sale_price_sum_str = miamihousing_needed_columns.groupby('STRUCTURE_QUALITY')['SALE_PRC'].sum()
sale_price_sum_str.plot(kind='bar', x='STRUCTURE_QUALITY', y='SALE_PRC', fig=(10, 10))
plt.ticklabel_format(style='plain', axis='y')

#price per house
27877400 / 172
1093151091 / 4053
29556000 / 16
2892145185 / 7536 
1475031050 / 1981


#finding the most expensive and cheapest houses
million = miamihousing_needed_columns.loc[miamihousing_needed_columns['SALE_PRC'] >= 1000000]
sorted_million = million.sort_values(['SALE_PRC'], ascending = False)
sorted_million = sorted_million[['SALE_PRC', 'LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'AGE', 'STRUCTURE_QUALITY', 'AVNO60PLUS']]
sorted_million.describe()

sns.relplot(x='LND_SQFOOT',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='STRUCTURE_QUALITY', palette='viridis',data=miamihousing_needed_columns)
plt.xlabel('Land Square Foot')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='LND_SQFOOT',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='STRUCTURE_QUALITY', palette='viridis',data=million)
plt.xlabel('Land Square Foot')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='TOT_LVG_AREA',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='AGE', palette='mako',data=miamihousing_needed_columns)
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='TOT_LVG_AREA',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='AGE', palette='mako',data=million)
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='WATER_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SPEC_FEAT_VAL', palette='dark:salmon_r',data=miamihousing_needed_columns)
plt.xlabel('Water Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='WATER_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SPEC_FEAT_VAL', palette='dark:salmon_r',data=million)
plt.xlabel('Water Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='CNTR_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SUBCNTR_DI', palette='flare',data=miamihousing_needed_columns)
plt.xlabel('Centre Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='CNTR_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SUBCNTR_DI', palette='flare',data=million)
plt.xlabel('Centre Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='OCEAN_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='RAIL_DIST', palette='rocket',data=miamihousing_needed_columns)
plt.xlabel('Ocean Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='OCEAN_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='RAIL_DIST', palette='rocket',data=million)
plt.xlabel('Ocean Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sorted_million.groupby('STRUCTURE_QUALITY').count()
sorted_million_bad_quality = sorted_million.loc[(sorted_million['STRUCTURE_QUALITY'] == 1) | (sorted_million['STRUCTURE_QUALITY'] == 2)]
sorted_million_bad_quality['SALE_PRC'].mean()
sorted_million_bad_quality.describe()
sorted_million.groupby('AVNO60PLUS').count()

sns.relplot(x='LND_SQFOOT',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='STRUCTURE_QUALITY', palette='viridis',data=sorted_million_bad_quality)
plt.xlabel('Land Square Foot')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='TOT_LVG_AREA',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='AGE', palette='mako',data=sorted_million_bad_quality)
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='WATER_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SPEC_FEAT_VAL', palette='dark:salmon_r',data=sorted_million_bad_quality)
plt.xlabel('Water Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='CNTR_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SUBCNTR_DI', palette='dark:salmon_r',data=sorted_million_bad_quality)
plt.xlabel('Centre Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='OCEAN_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='RAIL_DIST', palette='dark:salmon_r',data=sorted_million_bad_quality)
plt.xlabel('Ocean Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')


less_than_onefifty = miamihousing_needed_columns.loc[miamihousing_needed_columns['SALE_PRC'] <= 150000]
sorted_less_than_onefifty = less_than_onefifty.sort_values(['SALE_PRC'], ascending = True)
sorted_less_than_onefifty = sorted_less_than_onefifty[['SALE_PRC', 'LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST', 'AGE', 'STRUCTURE_QUALITY', 'AVNO60PLUS']]
sorted_less_than_onefifty.describe()
sorted_less_than_onefifty.groupby('STRUCTURE_QUALITY').count()

sns.relplot(x='LND_SQFOOT',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='STRUCTURE_QUALITY', palette='viridis',data=less_than_onefifty)
plt.xlabel('Land Square Foot')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='TOT_LVG_AREA',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='AGE', palette='mako',data=less_than_onefifty)
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='WATER_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SPEC_FEAT_VAL', palette='dark:salmon_r',data=less_than_onefifty)
plt.xlabel('Water Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='CNTR_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SUBCNTR_DI', palette='flare',data=less_than_onefifty)
plt.xlabel('Centre Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='OCEAN_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='RAIL_DIST', palette='rocket',data=less_than_onefifty)
plt.xlabel('Ocean Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

four_five_cheap = sorted_less_than_onefifty.loc[(sorted_less_than_onefifty['STRUCTURE_QUALITY'] == 4) | (sorted_less_than_onefifty['STRUCTURE_QUALITY'] == 5)]
sorted_less_than_onefifty['SALE_PRC'].mean()
four_five_cheap_age = four_five_cheap.sort_values(['AGE'], ascending = True)
ffc_des = four_five_cheap.describe()
sorted_less_than_onefifty.groupby('AVNO60PLUS').count()

sns.relplot(x='LND_SQFOOT',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='STRUCTURE_QUALITY', palette='viridis',data=four_five_cheap)
plt.xlabel('Land Square Foot')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='TOT_LVG_AREA',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='AGE', palette='mako',data=four_five_cheap)
plt.xlabel('Floor Area')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='WATER_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SPEC_FEAT_VAL', palette='dark:salmon_r',data=four_five_cheap)
plt.xlabel('Water Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='CNTR_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='SUBCNTR_DI', palette='flare',data=four_five_cheap)
plt.xlabel('Centre Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

sns.relplot(x='OCEAN_DIST',y='SALE_PRC',sizes=(1, 100),size='SALE_PRC', hue='RAIL_DIST', palette='rocket',data=four_five_cheap)
plt.xlabel('Ocean Distance')
plt.ylabel('Sale Price')
plt.ticklabel_format(style='plain', axis='y')

#less than million and bigger than 150000 analyse
first = miamihousing_needed_columns.loc[(miamihousing_needed_columns['SALE_PRC'] > 150000) & (miamihousing_needed_columns['SALE_PRC'] <= 350000)]
f_des = first.describe()
first.groupby('STRUCTURE_QUALITY').count()
f_sort = first.sort_values(['SALE_PRC'], ascending = False)

second = miamihousing_needed_columns.loc[(miamihousing_needed_columns['SALE_PRC'] > 350000) & (miamihousing_needed_columns['SALE_PRC'] <= 550000)]
sec_des = second.describe()
second.groupby('STRUCTURE_QUALITY').count()
sec_sort = second.sort_values(['SALE_PRC'], ascending = False)

three = miamihousing_needed_columns.loc[(miamihousing_needed_columns['SALE_PRC'] > 550000) & (miamihousing_needed_columns['SALE_PRC'] <= 750000)]
thr_des = three.describe()
three.groupby('STRUCTURE_QUALITY').count()
thr_sort = three.sort_values(['SALE_PRC'], ascending = False)

four = miamihousing_needed_columns.loc[(miamihousing_needed_columns['SALE_PRC'] > 750000) & (miamihousing_needed_columns['SALE_PRC'] <=  999999)]
fr_des = four.describe()
four.groupby('STRUCTURE_QUALITY').count()
fr_sort = four.sort_values(['SALE_PRC'], ascending = False)

miamihousing_needed_columns_new = miamihousing_needed_columns.drop('MORE_THAN_MILLION', axis = 1)

pearson_corr = miamihousing_needed_columns.corr(method = 'pearson')
kendall_corr = miamihousing_needed_columns.corr(method = 'kendall')
spearman_corr = miamihousing_needed_columns.corr(method = 'spearman')

sns.heatmap(pearson_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
sns.heatmap(kendall_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Greens')
sns.heatmap(spearman_corr, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Reds')


# For Prediction

#Million or not using Logistic Regression
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

miamihousing_needed_columns['MORE_THAN_MILLION'] = ''
miamihousing_needed_columns['MORE_THAN_MILLION'] = np.where(miamihousing_needed_columns['SALE_PRC'] >= 1000000, 1, 0)
miamihousing_needed_columns['MORE_THAN_MILLION'] = pd.Categorical(miamihousing_needed_columns.MORE_THAN_MILLION)
miamihousing_needed_columns.info()

lr_data = miamihousing_needed_columns.drop('SALE_PRC', axis = 1)

depend_var = lr_data['MORE_THAN_MILLION']
indep_vars = lr_data.drop('MORE_THAN_MILLION', axis = 1)

indep_vars, depend_var = make_classification(random_state=42)

indep_vars_train, indep_vars_test, depend_var_train, depend_var_test = train_test_split(indep_vars, depend_var, test_size=0.25, random_state=42)
lrmodel2 = make_pipeline(StandardScaler(), LogisticRegression()) 
lrmodel2.fit(indep_vars_train, depend_var_train)
predictions2 = lrmodel2.predict(indep_vars_test)

classification_report(depend_var_test, predictions2)

confusion_matrix(depend_var_test, predictions2)

accuracy_score(depend_var_test, predictions2)#0.96

# RandomForest -- more or less than a million

np.random.seed(0)

lr_data['is_train'] = np.random.uniform(0, 1, len(lr_data)) <= .75

train, test = lr_data[lr_data['is_train']==True], lr_data[lr_data['is_train'] == False]
len(train)

features = lr_data.columns[:13]
depend_var_rf = pd.factorize(train['MORE_THAN_MILLION'])[0]
depend_var_rf
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], depend_var_rf)
preds = clf.predict(test[features])
pd.crosstab(test['MORE_THAN_MILLION'], preds)
3220+126
3346/3488*100 #95.9%

#SVM milllion or not

depend_var_svm = lr_data['MORE_THAN_MILLION']
indep_vars_svm = lr_data.drop('MORE_THAN_MILLION', axis = 1)

indep_vars_train_svm, indep_vars_test_svm, depend_var_train_svm, depend_var_test_svm = train_test_split(indep_vars_svm, depend_var_svm, test_size=0.3, random_state=42)
clf_svm = svm.SVC(kernel = 'linear')
clf_svm.fit(indep_vars_train_svm, depend_var_train_svm)
pred_svm = clf_svm.predict(indep_vars_test_svm)

metrics.accuracy_score(depend_var_test_svm, y_pred = pred_svm)#0.97

#Decision Tree milllion or not

indep_vars_dt = lr_data.values[:, 0:12]
dep_var_dt = lr_data.values[:, 13]
dep_var_dt = dep_var_dt.astype('int')

indep_vars_train_dt, indep_vars_test_dt, depend_var_train_dt, depend_var_test_dt = train_test_split(indep_vars_dt, dep_var_dt, test_size=0.3, random_state=42)
clf_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth = 3, min_samples_leaf = 5)
clf_dt.fit(indep_vars_train_dt, depend_var_train_dt)
pred_dt = clf_dt.predict(indep_vars_test_dt)
accuracy_score(depend_var_test_dt, pred_dt)#0.97


#House price prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

dummy_df = miamihousing_needed_columns

housing_price_prediction_data = dummy_df.drop(['MONTH_SOLD', 'MORE_THAN_MILLION'], axis=1)

indep_vars_regress = housing_price_prediction_data.drop(['SALE_PRC'], axis=1)
dep_var_regress = housing_price_prediction_data['SALE_PRC']

indep_vars_regress_train, indep_vars_regress_test, dep_var_regress_train, dep_var_regress_test = train_test_split(indep_vars_regress, dep_var_regress, test_size=0.2, random_state=100)

std_sclr = StandardScaler()
indep_vars_regress_train_tr = std_sclr.fit_transform(indep_vars_regress_train)
indep_vars_regress_test_tr = std_sclr.transform(indep_vars_regress_test)

dep_var_regress_train_df = pd.DataFrame(dep_var_regress_train)
dep_var_regress_test_df = pd.DataFrame(dep_var_regress_test)

dep_var_regress_train_tr = std_sclr.fit_transform(dep_var_regress_train_df)
dep_var_regress_test_tr = std_sclr.fit_transform(dep_var_regress_test_df)

#RandomForest
rf_regress = RandomForestRegressor()
rf_regress.fit(indep_vars_regress_train_tr, dep_var_regress_train_tr)
rf_regress_pred = rf_regress.predict(indep_vars_regress_test_tr)

mean_squared_error(dep_var_regress_test_tr, rf_regress_pred)
mean_absolute_error(dep_var_regress_test_tr, rf_regress_pred)
np.sqrt(mean_squared_error(dep_var_regress_test_tr, rf_regress_pred))
r2_score(dep_var_regress_test_tr, rf_regress_pred)

plt.scatter(range(len(dep_var_regress_test_tr)), dep_var_regress_test_tr, color='black')
plt.scatter(range(len(rf_regress_pred)), rf_regress_pred, color='green')

#Decision Tree
dt_regress = DecisionTreeRegressor()
dt_regress.fit(indep_vars_regress_train_tr, dep_var_regress_train_tr)
dt_regress_pred = dt_regress.predict(indep_vars_regress_test_tr)

mean_squared_error(dep_var_regress_test_tr, dt_regress_pred)
mean_absolute_error(dep_var_regress_test_tr, dt_regress_pred)
r2_score(dep_var_regress_test_tr, dt_regress_pred)

plt.scatter(range(len(dep_var_regress_test_tr)), dep_var_regress_test_tr, color='black')
plt.scatter(range(len(dt_regress_pred)), dt_regress_pred, color='blue')

#SVM
svm_regress = SVR(kernel='rbf')
svm_regress.fit(indep_vars_regress_train_tr, dep_var_regress_train_tr)
svm_regress_pred = svm_regress.predict(indep_vars_regress_test_tr)

mean_squared_error(dep_var_regress_test_tr, svm_regress_pred)
mean_absolute_error(dep_var_regress_test_tr, svm_regress_pred)
r2_score(dep_var_regress_test_tr, svm_regress_pred)

plt.scatter(range(len(dep_var_regress_test_tr)), dep_var_regress_test_tr, color='black')
plt.scatter(range(len(svm_regress_pred)), svm_regress_pred, color='red')

#Lasso
lasso_regress = Lasso(alpha=0.1)
lasso_regress.fit(indep_vars_regress_train_tr, dep_var_regress_train_tr)
lasso_regress_pred = lasso_regress.predict(indep_vars_regress_test_tr)

mean_squared_error(dep_var_regress_test_tr, lasso_regress_pred)
mean_absolute_error(dep_var_regress_test_tr, lasso_regress_pred)
r2_score(dep_var_regress_test_tr, lasso_regress_pred)

plt.scatter(range(len(dep_var_regress_test_tr)), dep_var_regress_test_tr, color='black')
plt.scatter(range(len(lasso_regress_pred)), lasso_regress_pred, color='#ff7f0e')

#Ridge
ridge_regress = Ridge(alpha=0.1)
ridge_regress.fit(indep_vars_regress_train_tr, dep_var_regress_train_tr)
ridge_regress_pred = ridge_regress.predict(indep_vars_regress_test_tr)

mean_squared_error(dep_var_regress_test_tr, ridge_regress_pred)
mean_absolute_error(dep_var_regress_test_tr, ridge_regress_pred)
r2_score(dep_var_regress_test_tr, ridge_regress_pred)

plt.scatter(range(len(dep_var_regress_test_tr)), dep_var_regress_test_tr, color='black')
plt.scatter(range(len(ridge_regress_pred)), ridge_regress_pred, color='#bcbd22')

#SVM house is above or below avg
miamihousing_needed_columns['ABOVE_BELOW_AVG'] = ''
miamihousing_needed_columns['ABOVE_BELOW_AVG'] = np.where(miamihousing_needed_columns['SALE_PRC'] >= miamihousing_needed_columns['SALE_PRC'].mean(), 1, 0)
miamihousing_needed_columns['ABOVE_BELOW_AVG'] = pd.Categorical(miamihousing_needed_columns.ABOVE_BELOW_AVG)
miamihousing_needed_columns.info()

svm_avg_data = miamihousing_needed_columns.drop('SALE_PRC', axis = 1)

depend_var_svm2 = svm_avg_data['ABOVE_BELOW_AVG']
indep_vars_svm2 = svm_avg_data.drop('ABOVE_BELOW_AVG', axis = 1)

indep_vars_train_svm2, indep_vars_test_svm2, depend_var_train_svm2, depend_var_test_svm2 = train_test_split(indep_vars_svm2, depend_var_svm2, test_size=0.3, random_state=42)
clf_svm2 = svm.SVC(kernel = 'linear')
clf_svm2.fit(indep_vars_train_svm2, depend_var_train_svm2)
pred_svm2 = clf_svm2.predict(indep_vars_test_svm2)

metrics.accuracy_score(depend_var_test_svm2, y_pred = pred_svm2)#0.90












