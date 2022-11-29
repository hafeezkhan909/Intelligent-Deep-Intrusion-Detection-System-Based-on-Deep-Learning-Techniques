import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

# sklearn: data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# sklearn: train model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
import time
import datetime
from packaging import version

import os
pip install -U tensorboard_plugin_profile
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)
%load_ext tensorboard

from keras.models import Sequential
from keras import layers
from keras.layers import Dropout,Dense
from keras.models import Sequential

initial_data = pd.read_csv(r"C:\Users\HAFEEZ KHAN\Desktop\UNSW_NB15_training-set.csv")

initial_data.head(n=5)

initial_data.info()

initial_data.drop('id', inplace=True, axis=1)
initial_data.info()


initial_data.drop('proto', inplace=True, axis=1)
initial_data.drop('service', inplace=True, axis=1)
initial_data.drop('state', inplace=True, axis=1)

initial_data.info()

initial_data.isnull().sum()

# Discard the rows with missing values
data_to_use = initial_data.dropna()

# Shape of the data: we could see that the number of rows remains the same as no null values were reported
data_to_use.shape

X = data_to_use.drop(axis=1, columns=['attack_cat']) # X is a dataframe
X = X.drop(axis=1, columns=['label'])


y1 = data_to_use['attack_cat'].values # y is an array
y2 = data_to_use['label'].values

# Calculate Y2 ratio
def data_ratio(y2):
    '''
    Calculate Y2's ratio
    '''
    unique, count = np.unique(y2, return_counts=True)
    ratio = round(count[0]/count[1], 1)
    return f'{ratio}:1 ({count[0]}/{count[1]})'
    
print('The class ratio for the original data:', data_ratio(y1))
plt.figure(figsize=(13,5))
sns.countplot(y1,label="Sum")
plt.show()

print('The class ratio for the original data:', data_ratio(y2))
sns.countplot(y2,label="Sum")
plt.show()

# Load data
test_data = pd.read_csv(r"C:\Users\HAFEEZ KHAN\Desktop\UNSW_NB15_training-set.csv")
X_test = test_data.drop(axis=1, columns=['attack_cat']) # X_test is a dataframe
X_test = X_test.drop(axis=1, columns=['label'])


y1_test = test_data['attack_cat'].values # y is an array
y2_test = test_data['label'].values

X_train = X
y1_train = y1
y2_train = y2

# determine categorical and numerical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

numerical_cols
categorical_cols

# define the transformation methods for the columns
t = [('ohe', OneHotEncoder(drop='first'), categorical_cols),
    ('scale', StandardScaler(), numerical_cols)]

col_trans = ColumnTransformer(transformers=t)

# fit the transformation on training data
col_trans.fit(X_train)

X_train_transform = col_trans.transform(X_train)
# apply transformation to both training and testing data 
# fit the transformation on training data
X_test_transform = col_trans.transform(X_test)

# Note that the distinct values/labels in `y2` target are 1 and 2. 
pd.unique(y1)
pd.unique(y2)

# Define a LabelEncoder() transformation method and fit on y1_train
target_trans = LabelEncoder()
target_trans.fit(y1_train)


# apply transformation method on y1_train and y1_test
y1_train_transform = target_trans.transform(y1_train)
y1_test_transform = target_trans.transform(y1_test)

# view the transformed y1_train
y1_train_transform

# Define a LabelEncoder() transformation method and fit on y2_train
target_trans = LabelEncoder()
target_trans.fit(y2_train)
y2_train_transform = target_trans.transform(y2_train)
y2_test_transform = target_trans.transform(y2_test)

# view the transformed y2_train
y2_train_transform

from keras.layers import BatchNormalization

model = Sequential()
model.add(layers.Reshape((X_train_transform.shape[1], 1), input_shape= (X_train_transform.shape[1],)))

model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(layers.MaxPooling1D())
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(layers.GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid')) #rel
model.summary()

start = time.time()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['acc'])
end = time.time()
print("Run time [s]: ",end-start)

start = time.time()
history = model.fit(X_train_transform, y2_train_transform,
                    epochs=30,
                    validation_data=(X_test_transform,y2_test_transform))
print("Run time [s]: ",end-start)
model.save('Hybrid.h5')

feature_names = np.array(numerical_cols)
feature_names

y_pred = model.predict(X_test_transform)

timeit y_pred[y_pred > 0.5] = 1

timeit y_pred[y_pred <= 0.5] = 0

print(y_pred)
y_pred.astype(int)
print(y2_test_transform)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y2_test_transform, y_pred))
report=metrics.classification_report(y2_test_transform,y_pred)
print(report)

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report

cm_ontest = confusion_matrix(y_true=y2_test_transform, y_pred=y_pred)
print('Confusion Matrix:\n', cm_ontest)
precision_ontest = precision_score(y_true=y2_test_transform, y_pred=y_pred)
# recall score
recall_ontest = recall_score(y_true=y2_test_transform, y_pred=y_pred)

print('The precision score on the test set: {:1.5f}'.format(precision_ontest))
print('The recall score on the test set: {:1.5f}'.format(recall_ontest))

f1_ontest = f1_score(y_true=y2_test_transform, y_pred=y_pred)
print('The f1 score on the test set: {:1.5f}'.format(f1_ontest))

acc_ontest = accuracy_score(y_true=y2_test_transform, y_pred=y_pred)
print('The accuracy score on the test set: {:1.5f}'.format(acc_ontest))
