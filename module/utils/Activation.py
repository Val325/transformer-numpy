import numpy as np

def softmax(z):
    """Computes softmax function.

    z: array of input values.

    Returns an array of outputs with the same shape as z."""
    # For numerical stability: make the maximum of z's to be 0.
    shiftz = z - np.max(z)
    exps = np.exp(shiftz)
    return exps / np.sum(exps)

def softmax_gradient(z):
    """Computes the gradient of the softmax function.

    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.

    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.greater(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)
