import numpy as np
#from Activation import softmax

def mse_loss(y_pred, y_true):
    """
    Calculates the mean squared error (MSE) loss between predicted and true values.
    
    Args:
    - y_pred: predicted values
    - y_true: true values
    
    Returns:
    - mse_loss: mean squared error loss
    """
    n = len(y_pred)    
    mse_loss = np.sum((y_pred - y_true) ** 2) / (2*n) 
    return mse_loss

def mse_derivative(y_true, y_pred): 
    n = len(y_pred)
    return (y_pred - y_true) / n

def binary_cross_entropy_loss(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()

def binary_cross_entropy_derivative(y_pred, y_true):
    return -((y_true / y_pred) - (1 - y_true / 1 - y_pred))

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce



#def cross_entropy(X,y):
#    """
#    X is the output from fully connected layer (num_examples x num_classes)
#    y is labels (num_examples x 1)
#    	Note that y is not one-hot encoded vector.
#    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
#    """
#    m = y.shape[0]
#    #p = softmax(X)
#    # We use multidimensional array indexing to extract
#    # softmax probability of the correct label for each sample.
#    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
#    #log_likelihood = -np.log(p[range(m),y])
#    log_likelihood = -(np.log(X) * y)  
#    loss = np.sum(log_likelihood) / m
#    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
   	Note that y is not one-hot encoded vector.
   	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    #grad = softmax(X)
    grad = X - y 
    grad = grad/m
    return grad

#predictions = np.array([[0.25,0.25,0.25,0.25],[0.01,0.01,0.01,0.96]])
#targets = np.array([[0,0,0,1],[0,0,0,1]])
#cross = cross_entropy(predictions, targets)
#delta_cross = delta_cross_entropy(predictions, targets)

#print(cross)
#print(delta_cross)



