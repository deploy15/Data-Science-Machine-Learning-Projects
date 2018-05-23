# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 05:55:46 2017

@author: HP
"""

import scipy
from scipy import stats
from scipy.stats import rv_continuous
import numpy as np
import pandas as  pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

# Import evaluation functions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Initialize a few classifiers for comparison
clfs = [DecisionTreeClassifier(), SVC(), BernoulliNB(), RandomForestClassifier()]

filepath_train = "C:/My Projects/datascience/dataset/insurance_claim/train.csv"
filepath_test = "C:/My Projects/datascience/dataset/insurance_claim/train.csv"

df_data_train = pd.read_csv(filepath_train)
df_data_test = pd.read_csv(filepath_test)
#header_training = df_data_train.columns.tolist()
#print (header)

# Create a list of the feature column's names
#features = df_data.columns[:,2]

# View features
#print (features)


#print (df_data.head())

y = pd.factorize(df_data_train['target'])[0]

# View target
print(y.size)
cnt0 = 0
cnt1 = 1
for i in y:
    if(i == 0):
       cnt0 += 1
    elif(i==1):
        cnt1 += 1

print (cnt0)
print (cnt1)

# For Training Data
header_training = df_data_train.columns.tolist()
header_testing = df_data_test.columns.tolist()
header_training.remove('target') # Survive has been removed as a class title
header_testing.remove('target') # y testing
x_header = header_training
y_header = "target"
print (header_training)
print (y_header)
x_data = df_data_train[x_header]
y_data = df_data_train[y_header]
print (x_data.head())
print (y_data.head())

#for Testing Data
x_test_header = header_testing
y_test_header = "target"

x_test_data = df_data_test[x_test_header]
y_test_data = df_data_train[y_test_header]

print (x_test_data.shape)
print (y_test_data.shape)


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(x_data, y_data)

preds = clf.predict(x_test_data)

print ("************")
print (preds.shape)

print (clf.score(x_test_data, y_test_data, sample_weight=None))
final_prop = (clf.predict_proba(x_test_data))

# Train and Score each classifier on a standard single training/test split of the data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
for clf in clfs:
    clf.fit(x_data, y_data)
    print(type(clf).__name__, clf.score(x_test_data, y_test_data))

# Create confusion matrix
print (pd.crosstab(y_test_data, preds, rownames=['Actual Insurance'], colnames=['Predicted Insurance']))
# Create actual english names for the plants for each predicted plant class
#preds = iris.target_names[clf.predict(test[features])]

# Selecting Important features
clf = ExtraTreesClassifier()
clf = clf.fit(x_data, y_data)
#print (clf.feature_importances_) 
#array([ 0.04...,  0.05...,  0.4...,  0.4...])
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(x_data)

#print (X_new)

#clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),('classification', RandomForestClassifier())])
#clf.fit(x_data, y_data)
print (x_test_data.size)
print (final_prop.size)

'''
print (type(final_prop))
for x in np.nditer(final_prop):
    print (x, x_test_data)
    #print (x_test_data)
'''