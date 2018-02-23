# 2018-02-22
# mfincker
# Baseline for EHR project:
#	- split into train : test set (80:20)
#	- PCA to scale down the number of feature
#	- mlp



#############
# Libraries #
#############

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sys import argv

from sklearn import preprocessing, decomposition, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


###################
# Loading dataset #
###################
data_path = argv[1]

df = pd.read_csv(data_path)

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

mlp = MLPClassifier(solver = 'lbfgs')

pca = decomposition.PCA(n_components = n_feature, svd_solver = 'full')

pipe = Pipeline(steps=[('pca', pca), ('mlp', mlp)])

####################
# Center and Scale #
####################

scaler = preprocessing.StandardScaler().fit(train_x)

train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


#######
# PCA #
#######
# We currently don't have enough data to run logistic regression
# with all features (# features > # obs)
# So we'll use PCA to reduce dimensionality 

pca.fit = pca.fit(train_x)

#######################
# Logistic regression #
#######################
# with L2 regularization

# Hyperparameters
# - PCA component to keep
n_components = [5, 20, 40, 64, 100, 300]
# - L2 reg parameter
alpha_l2 = np.logspace(-6, 3, 6)
# - Number of hidden layers / Number of neuron per hidden layers
hidden_layers = [[10, 10],[50, 5],[10, 10, 10]]

# Hyperparameter grid search with cross-validation
estimator = GridSearchCV(pipe,
                         dict(pca__n_components = n_components,
                              mlp__alpha = alpha_l2,
                              mlp__hidden_layer_sizes = hidden_layers),
                         verbose = 1,
                         scoring = 'f1')

estimator.fit(train_x, train_y)

# Save model / training for later
pickle.dump(estimator, open("mlp_gridsearch_cv.sav", 'wb'))
pickle.dump(estimator.best_estimator_, open("mlp_best_model.sav", "wb"))

# Plot PCA
label = 'n_components chosen: ' + str(estimator.best_estimator_.named_steps['pca'].n_components) + "\nL2 param: " + str(estimator.best_estimator_.named_steps['mlp'].alpha)
plt.figure(1, figsize=(8, 6))
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n components')
plt.ylabel('explained variance')
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label=label)
plt.legend(prop=dict(size=12))
plt.savefig("mlp_PCA_with_chosen_param.png")

######################
# Prediction on test #
######################

mlp_model = estimator.best_estimator_


predict_y = mlp_model.predict(test_x)
accuracy_test = metrics.accuracy_score(test_y, predict_y)
report_test = metrics.classification_report(test_y, predict_y)
conf_matrix_test = metrics.confusion_matrix(test_y, predict_y)

predict_train = mlp_model.predict(train_x)
accuracy_train = metrics.accuracy_score(train_y, predict_train)
report_train = metrics.classification_report(train_y, predict_train)
conf_matrix_train = metrics.confusion_matrix(train_y, predict_train)


with open("report_mlp.txt", 'w') as f:

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

