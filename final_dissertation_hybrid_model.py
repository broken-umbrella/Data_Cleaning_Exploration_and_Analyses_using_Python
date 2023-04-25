# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:41:22 2023

@author: amaru
"""

#FiveModels
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
research_data = pd.read_excel('D:/Diss/bicycleUpdated.xlsx')
research_data.info()

research_data = research_data.set_index('dates').sort_index()

research_data_encoded = research_data.apply(LabelEncoder().fit_transform)

depend_var = research_data_encoded['severity']
independent_vars = research_data_encoded.drop('severity', axis = 1)

independent_vars_train, independent_vars_test, depend_var_train, depend_var_test = train_test_split(independent_vars, depend_var, test_size=0.10, random_state=1)


knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(independent_vars_train, depend_var_train)

xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(independent_vars_train, depend_var_train)


voting_clf = VotingClassifier(estimators=[('knn', knn_clf), 
                                          ('xgb', xgb_clf)], 
                                          voting='soft')
voting_clf.fit(independent_vars_train, depend_var_train)

hybrid_pred = voting_clf.predict(independent_vars_test)

confusion_matrix(depend_var_test, hybrid_pred)
accuracy_score(depend_var_test, hybrid_pred)#83.99









