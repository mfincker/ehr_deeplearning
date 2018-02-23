# 2018-02-22
# mfincker
# Baseline for EHR project:
#	- split into train : test set (80:20)
#	- PCA to scale down the number of feature
#	- logistic regression



#############
# Libraries #
#############

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn import preprocessing, linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


###################
# Loading dataset #
###################
data_path = "/Users/maeva/Documents/classes/cs230_deep_learning/project/data/sample_codes.csv"

df = pd.read_csv(data_path)

# print df.head()

##########################
# Split train / test set #
##########################
#  will expand to train / dev / set once we have more data

frac_test = 0.2

test_x = df.iloc[0:int(frac_test * df.shape[0]), 3:].values
test_y = df.iloc[0:int(frac_test * df.shape[0]), 2].values

train_x = df.iloc[int(frac_test * df.shape[0]):, 3:].values
train_y = df.iloc[int(frac_test * df.shape[0]):, 2].values

n_obs, n_feature = test_x.shape


####################### 
# Setting up pipeline #
#######################

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA(n_components = n_feature, svd_solver = 'full')

pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

####################
# Center and Scale #
####################

scaler = preprocessing.StandardScaler().fit(train_x)
# print "scaler mean shape: " + str(scaler.mean_.shape)

train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


#######
# PCA #
#######
# We currently don't have enough data to run logistic regression
# with all features (# features > # obs)
# So we'll use PCA to reduce dimensionality 

pca.fit = pca.fit(train_x)

#Variance explained by components
plt.figure(1, figsize=(4, 3))
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n components')
plt.ylabel('explained variance')
plt.savefig("PCA_variance_explained.png")


#######################
# Logistic regression #
#######################
# with L2 regularization

n_components = [5, 20, 40, 64, 100, 300]
Cs = np.logspace(-4, 4, 3)

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(train_x, train_y)


plt.figure(1, figsize=(4, 3))
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n components')
plt.ylabel('explained variance')
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.savefig("PCA_variance_explained_wwith_chosen_param.png")

