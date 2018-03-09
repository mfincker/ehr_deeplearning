""" Please refrain from changing any of the util code. """
from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Risk is at most 1 and at least 0, with some noise.
def set_risk_boundaries(risk, NOISE=0.001):
    risk[risk > 1] = 1 - NOISE
    risk[risk < 0] = NOISE
    return risk


def get_expected_risk(outcomes):
    return np.count_nonzero(outcomes)/ len(outcomes) * 100


# Plot risk function with 2 axes (x and y, along with z = risk)
def plot_risk(x0, x1, risk, outcomes, title='', indices=(-1,-1)):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60.)
    for color, unique_outcome in zip(['b', 'r'], [0, 1]):
        sampled = outcomes == unique_outcome
        ax.scatter(
            x0[sampled],
            x1[sampled],
            risk[sampled],
            c=color,
            alpha=0.2,
            label='adverse' if unique_outcome == 1 else 'not adverse')
    ax.set_xlabel('Variable ' + str(indices[0]))
    ax.set_ylabel('Variable ' + str(indices[1]))
    ax.set_zlabel(title)
    plt.legend()
    # plt.savefig(title+str(indices))
    plt.show()

## Takes in a risk function, and passes in the X to the function.
def generate_data(n, dims, risk_function):
    X = np.random.uniform(size=(n, dims))
    risks = risk_function(X)
    # generate outcomes
    y = np.random.binomial(1, risks).ravel()
    return X, y, risks

def run_model_on_function(my_risk_function, my_model_fn, log=True):
    X, y, risks = generate_data(100000, 10, my_risk_function)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    y_test_probs = my_model_fn(X_train, y_train, X_test)
    auc = roc_auc_score(y_test, y_test_probs)
    if log is True:
        print(auc)
    return auc

