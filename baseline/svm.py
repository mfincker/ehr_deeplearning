# 2018-02-22
# mfincker
# Baseline for EHR project:
#	- split into train : test set (80:20)
#	- PCA to scale down the number of feature
#	- svm



#############
# Libraries #
#############

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sys import argv

from sklearn import preprocessing, decomposition, metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


###################
# Loading dataset #
###################
print 'loading data'

train_x = pd.read_table("full_data10000_indexes_180days.train_x.aggregated.tsv", nrows = 1000).values
train_y = pd.read_table("full_data10000_indexes_180days.train_y.aggregated.tsv", nrows = 1000).values.flatten()
#dev_x = pd.read_table("full_data10000_indexes_180days.dev_x.aggregated.tsv").values
#dev_x = pd.read_table("full_data10000_indexes_180days.dev_y.aggregated.tsv").values

test_x = pd.read_table("full_data10000_indexes_180days.test_x.aggregated.tsv", nrows = 1000).values
test_y = pd.read_table("full_data10000_indexes_180days.test_y.aggregated.tsv", nrows = 1000).values.flatten()

##########################
# Split train / test set #
##########################
#  will expand to train / dev / set once we have more data

# frac_test = 0.2
#
# test_x = df.iloc[0:int(frac_test * df.shape[0]), 3:].values
# test_y = df.iloc[0:int(frac_test * df.shape[0]), 2].values
#
# train_x = df.iloc[int(frac_test * df.shape[0]):, 3:].values
# train_y = df.iloc[int(frac_test * df.shape[0]):, 2].values

n_obs, n_feature = test_x.shape


#######################
# Setting up pipeline #
#######################

print 'setting up pipeline'

svm = SVC(class_weight = 'balanced')

pca = decomposition.PCA(n_components = n_feature, svd_solver = 'full')

pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])

####################
# Center and Scale #
####################

print 'center and scale'

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

print 'fitting pca'

pca.fit = pca.fit(train_x)


########
# SVM #
#######
# with L2 regularization

# Hyperparameters grid
# - PCA components to keep
n_components = [5, 20, 40, 64, 100, 300]
# - L2 reg parameter
Cs = np.logspace(-6, 3, 6)

# Hyperparameters grid search
print 'finding hyperparameters'

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              svm__C=Cs))

estimator.fit(train_x, train_y)

# Plot PCA
print 'plotting'

label = 'n_components chosen: ' + str(estimator.best_estimator_.named_steps['pca'].n_components) + "\nL2 param: " + str(estimator.best_estimator_.named_steps['svm'].C)
plt.figure(1, figsize=(8, 6))
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n components')
plt.ylabel('explained variance')
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label=label)
plt.legend(prop=dict(size=12))
plt.savefig("svm_PCA_with_chosen_param.png")

######################
# Prediction on test #
######################

print 'predicting on test set'

predict_y = estimator.best_estimator_.predict(test_x)
accuracy_test = metrics.accuracy_score(test_y, predict_y)
report_test = metrics.classification_report(test_y, predict_y)
conf_matrix_test = metrics.confusion_matrix(test_y, predict_y)

predict_train = estimator.best_estimator_.predict(train_x)
accuracy_train = metrics.accuracy_score(train_y, predict_train)
report_train = metrics.classification_report(train_y, predict_train)
conf_matrix_train = metrics.confusion_matrix(train_y, predict_train)


# Save model performance
with open("report_svm.txt", 'w') as f:

	f.write("estimator cv:\n")
	f.write(str(estimator.cv_results_ ))

	f.write('\n\n#########\n')
	f.write('# Train #\n')
	f.write('#########\n')
	f.write("accuracy: " + str(accuracy_train) + "\n")

	f.write("confusion matrix:\n")
	f.write(str(conf_matrix_train) + "\n")

	f.write("report:\n")
	f.write(str(report_train))

	f.write('\n\n########\n')
	f.write('# Test #\n')
	f.write('########\n')
	f.write("accuracy: " + str(accuracy_test) + "\n")

	f.write("confusion matrix:\n")
	f.write(str(conf_matrix_test))

	f.write("report:\n")
	f.write(str(report_test))
