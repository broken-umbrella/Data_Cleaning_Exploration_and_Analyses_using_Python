# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 20:41:02 2023

@author: amaru
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

research_data = pd.read_excel('D:/Diss/bicycleUpdated.xlsx')
research_data.info()

research_data = research_data.set_index('dates').sort_index()

research_data_encoded = research_data.apply(LabelEncoder().fit_transform)

depend_var = research_data_encoded['severity']
independent_vars = research_data_encoded.drop('severity', axis = 1)

independent_vars_train, independent_vars_test, depend_var_train, depend_var_test = train_test_split(independent_vars, depend_var, test_size=0.10, random_state=1)

smote = SMOTE(random_state=42)


independent_vars_train_resampled, depend_var_train_resampled = smote.fit_resample(independent_vars, depend_var)

#FFNN
# Define the model
fnn_model = Sequential()
fnn_model.add(Dense(16, input_dim=11, activation='relu'))
fnn_model.add(Dense(8, activation='relu'))
fnn_model.add(Dense(1, activation='sigmoid'))

# Compile the model
fnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
fnn_model.fit(independent_vars_train_resampled, depend_var_train_resampled, epochs=5, batch_size=64)

# Evaluate the model
loss, accuracy = fnn_model.evaluate(independent_vars_test, depend_var_test)

# Make predictions
fnn_predictions = fnn_model.predict(independent_vars_test)

fnn_predictions_binary = (fnn_predictions > 0.5).astype(int)
confusion_matrix(depend_var_test, fnn_predictions_binary)
accuracy_score(depend_var_test, fnn_predictions_binary)#69.24%


