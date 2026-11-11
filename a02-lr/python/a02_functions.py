# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py3
#     language: python
#     name: py3
# ---

# %%
import numpy as np


# %%
def normalize_data(X, Xtest):
    """Computes z-scores for a training set and a holdout test set.

    Keyword arguments:
    X, Xtest : two 2D ndarrays

    Result:
    Xz, Xtestz : two 2D ndarrays
    """
    # YOUR CODE HERE
    # Mean and standard deviation are estimated on the training data only and then
    # applied to both sets, so that test data is transformed in exactly the same way.
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xz = (X - mean) / std
    Xtestz = (Xtest - mean) / std
    return Xz, Xtestz


# %%
# you don't need to modify this function
def logsumexp(x):
    """Computes log(sum(exp(x)).

    Uses offset trick to reduce risk of numeric over- or underflow. When x is a
    1D ndarray, computes logsumexp of its entries. When x is a 2D ndarray,
    computes logsumexp of each column.

    Keyword arguments:
    x : a 1D or 2D ndarray
    """
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x - offset), axis=0))


# %%
# Define the logistic function. Make sure it operates on both scalars and vectors.
def sigma(x):
    # YOUR CODE HERE
    pass


# %%
# Define the logarithm of the logistic function. Make sure it operates on both scalars
# and vectors. Perhaps helpful: isinstance(x, np.ndarray).
def logsigma(x):
    # YOUR CODE HERE
    pass


# %%
def er(y, X, w):
    """Empirical risk of the logistic regression model.

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary labels (either 0 or 1).
    X : ndarray of shape (N,D)
        Design matrix.
    w : ndarray of shape (D,)
        Weight vector.
    """
    # YOUR CODE HERE
    pass


# %%
def d_er(y, X, w):
    """Gradient of the empirical risk of the logistic regression model

    Parameters
    ----------
    y : ndarray of shape (N,)
        Binary labels (either 0 or 1).
    X : ndarray of shape (N,D)
        Design matrix.
    w : ndarray of shape (D,)
        Weight vector.

    Returns
    -------
    ndarray of shape (D,)

    """
    # YOUR CODE HERE
    pass


# %%
# Define the objective and update function for one gradient-descent epoch for
# fitting an ERM estimate of logistic regression with gradient descent (should
# return a tuple of two functions; see optimize)
def gd_er(y, X):
    def objective(w):
        # YOUR CODE HERE
        pass

    def update(w, eps):
        # YOUR CODE HERE
        pass

    return (objective, update)


# %%
def sgdepoch_er(y, X, w, eps):
    """Run one SGD epoch and return the updated weight vector."""
    # Run N stochastic gradient steps (without replacement).
    # YOUR CODE HERE
    pass


# %%
# define the objective and update function for one gradient-descent epoch for
# fitting an MLE estimate of logistic regression with stochastic gradient descent
# (should return a tuple of two functions; see optimize)
def sgd_er(y, X):
    def objective(w):
        # YOUR CODE HERE
        pass

    def update(w, eps):
        return sgdepoch_er(y, X, w, eps)

    return (objective, update)


# %%
def predict(Xtest, w):
    """Returns vector of predicted confidence values for logistic regression with
    weight vector w."""
    # YOUR CODE HERE
    pass


def classify(Xtest, w):
    """Returns 0/1 vector of predicted class labels for logistic regression with
    weight vector w."""
    # YOUR CODE HERE
    pass


# %%
def rr(y, X, w, lambda_):
    """Regularized risk of logistic regression with weights w and L2
    regularization parameter lambda_"""
    # YOUR CODE HERE
    pass


# %%
def d_rr(y, X, w, lambda_):
    """Gradient of regularized risk of logistic regression with weights
    w and L2 regularization parameter lambda_."""
    # YOUR CODE HERE
    pass


# %%
# Define the objective and update function for one gradient-descent epoch for
# fitting an RRM estimate of logistic regression with gradient descent (should
# return a tuple of two functions; see optimize)
def gd_rr(y, X, lambda_):
    # YOUR CODE HERE
    pass


# %%
# you don't need to modify this function
def optimize(objective_update, theta0, nepochs=50, eps0=0.01, verbose=True):
    """Iteratively minimize a function.

    We use it here to run either gradient descent or stochastic gradient descent, using
    arbitrarly optimization criteria.

    Parameters
    ----------
    objective_update : a tuple of form (f, update) containing two functions f and update.
              f(theta) computes the value of the objective function.
              update(theta,eps) performs an epoch of parameter update with step size
              eps and returns the result.
    theta0  : ndarray of shape (D,)
              Initial parameter vector.
    nepochs : int
              How many epochs (calls to update) to run.
    eps0    : float
              Initial step size.
    verbose : boolean
              Whether to print progress information.

    Returns
    -------
    A triple consisting of the fitted parameter vector, the values of the
    objective function of every epoch (index 0 = initial objective), and
    the step sizes that were used (index 0 = step size of first epoch).
    """

    f, update = objective_update

    # initialize results
    theta = theta0
    f_values = np.zeros(nepochs + 1)
    eps_values = np.zeros(nepochs + 1)
    f_values[0] = f(theta0)
    eps_values[0] = eps0

    # now run the update function nepochs times
    for epoch in range(nepochs):
        if verbose:
            print(
                "Epoch {:3d}: f={:10.3f}, eps={:10.9f}".format(
                    epoch, f_values[epoch], eps_values[epoch]
                )
            )
     
        theta = update(theta, eps_values[epoch])

        # we use the bold driver heuristic
        f_values[epoch + 1] = f(theta)
        if f_values[epoch] < f_values[epoch + 1]:
            eps_values[epoch + 1] = eps_values[epoch] / 2.0
        else:
            eps_values[epoch + 1] = eps_values[epoch] * 1.05

    # all done
    if verbose:
        print("Result after {} epochs: f={}".format(nepochs, f_values[-1]))
    return theta, f_values, eps_values
