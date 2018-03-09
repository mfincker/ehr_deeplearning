""" Sample Submission.
    My risk function was motivated by the attributes affecting
    Framingham point scores. Age is weighted particularly heavily,
    since since it seems to have the largest influence upon score.
    Specifically, while low age in general reduces the Framingham
    score, it greatly increases the risk of smoking and having high
    blood pressure. Therefore, it's contribution to the risk is high
    at low values and high at high values. HDL and cholesterol seem
    to mitigate one another, where having more good cholesterol reduces
    the effect of high cholesterol. Smoking and blood pressure both
    directly increase Framingham score.

    https://www.nhlbi.nih.gov/health-pro/guidelines/current/
    cholesterol-guidelines/quick-desk-reference-html/
    10-year-risk-framingham-table

    The neural network outperforms the classifier with a score
    of around 0.72 compared to around 0.58. In particular, the age
    attribute affects the difference in performance, since it has
    a large affect on the risk and age does not linearly affect
    risk (it's high at low and high values).
"""
import numpy as np
import scipy as sp
import util
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out

lr = LogisticRegression()
nn = Net(10, 150, 1)

def my_risk_function(X):
        age, chol, pres, smok, hdl = tuple([X[:,i] for i in range(5)])

        risks = (age - 0.5)**2 + 0.1 * chol*(1 - hdl) + 0.1 * smok + 0.1 * pres
        risks = (risks - np.mean(risks))/np.std(risks)
        return sp.special.expit(risks)

def pranav_risk_function(X):
    def covariate_interactions(x, y, y_0_intercept=0.0):
        risk = 0.5*(x+y)*(x+y) + y_0_intercept
        risk = util.set_risk_boundaries(risk)
        return risk

    risks = []
    variable_interactions = [(2, 5), (4, 1), (3, 3)]

    for first_index, second_index in variable_interactions:
        risk = covariate_interactions(X[:, first_index], X[:, second_index], y_0_intercept=-0)
        risks.append(risk)
    risks = np.mean(np.array(risks), axis=0)
    return risks

def trainLR(X, y):
    lr.fit(X, y)

def predictLR(X):
    out = lr.predict(X)
    return out

def my_logistic_model_fn(X_train, y_train, X_test):
    trainLR(X_train, y_train)
    out = predictLR(X_test)
    return out

def trainNN(X, y):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(nn.parameters())
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y.astype(float))
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for epoch in tqdm(range(5)):
        for i, (X_t, y_t) in enumerate(dataloader):
            X_var = torch.autograd.Variable(X_t)
            y_var = torch.autograd.Variable(y_t)
            optimizer.zero_grad()
            outputs = nn(X_var).double().view(-1)
            loss = criterion(outputs, y_var)
            loss.backward()
            optimizer.step()

def predictNN(X):
    X_var = torch.autograd.Variable(torch.FloatTensor(X))
    out = torch.sigmoid(nn(X_var)).data
    return out

def my_nn_model_fn(X_train, y_train, X_test):
    trainNN(X_train, y_train)
    out = predictNN(X_test)
    return out.numpy().flatten()

def main():
    """ Your risk function should be passed into the generate data fn"""
    util.run_model_on_function(my_risk_function, my_logistic_model_fn)
    util.run_model_on_function(my_risk_function, my_nn_model_fn)
if __name__ == '__main__':
    main()