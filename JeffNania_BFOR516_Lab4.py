#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 21:21:53 2020

@author: jeffnania
"""

#%%
#Python Imports

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn import metrics

#%%

#Import Data

ccfraud = pd.read_csv('creditcard.csv')

#%%

#The data is highly unbalanced, with a fraction of a percent of transactions actually being fraudulent.

ccfraud.describe()
# get the average and count for each type
ccstats = ccfraud.groupby('Class')['Amount'].agg(['mean', 'count'])
# stats for fraud by count and average transaction amount
print(ccstats)

# percent of fraudulent transactions
print("Fraudulent transaction ratio:", ccstats.loc[1, 'count']/ccstats['count'].sum())

#%%

#Training and testing sets

np.random.seed(516)

train, test = train_test_split(ccfraud, test_size=0.25)
print("Rows in train:", len(train))
print("Rows in test:", len(test))
train_stats = train.groupby('Class')['Amount'].agg(['mean', 'count'])
print("Training data:\n", train_stats)
test_stats = test.groupby('Class')['Amount'].agg(['mean', 'count'])
print("Testing data:\n", test_stats)

#%%

#Train the model

# view all columns
print(list(ccfraud.columns))

# use column names 
pred_vars = ['Time', 'Amount', 'V8', 'V1']
print(ccfraud.loc[:, pred_vars])

#Train the actual model
dtree = tree.DecisionTreeClassifier(criterion="entropy")
dtree.fit(train.loc[:, pred_vars], train['Class'])

DecisionTreeClassifier(criterion='entropy')

print(dtree.get_n_leaves())
print(dtree.get_depth())

#%% 

#Evaluate Model Performance

pred_labels = dtree.predict(test.loc[:, pred_vars])
pred_labels[0:4]

#Confusion Matrix

metrics.plot_confusion_matrix(dtree, test.loc[:, pred_vars], test['Class'])

#Classification report

print(metrics.classification_report(test['Class'], pred_labels, digits=5))

#Probabalistic Evaluation

pred_probs = dtree.predict_proba(test.loc[:, pred_vars])
pred_probs[0:5, :]

#%%

#Area Under The Curve

metrics.roc_auc_score(test['Class'], pred_probs[:,1])

#Plot the curve

metrics.plot_roc_curve(dtree, test.loc[:, pred_vars], test['Class'])

#We can run similar stats to AUC, but instead of TPR and FPR, we can use precision and recall.

metrics.average_precision_score(test['Class'], pred_probs[:,1])

#Plot the precision/recall curve

metrics.plot_precision_recall_curve(dtree, test.loc[:, pred_vars], test['Class'])

#Log Loss

print(metrics.log_loss(test['Class'], pred_probs[:,1]))

#%%

#Lab Exercises

#1) What are the strengths and weaknesses of each evaluation criteria (precision/recall/F1/accuracy; model cost; log loss)?



#2) This model is severly over-fit. Try creating a new model and restricting the maximum depth of the tree to 5 levels (using the max_depth parameter). Run the various evaluation statistics on this new model.



###A) How does the tree compare to the original model?



###B) On which measures is it better/worse?



#3) Does adding additional variables to the model improve performance?



#4) This data is anonymized, which means the column names and their values have been obscured. What data columns do you think would be useful for detecting fraudulent credit card transactions?









