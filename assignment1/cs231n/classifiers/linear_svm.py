import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):  # for each example
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]  # y[i] - correct class number
        for j in xrange(num_classes):  # iterate for classes
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W # regularize the weights

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    delta = np.ones_like(scores) # all delta=1
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W) # scores for classes 
    right_class_scores = scores[np.arange(num_train), y] # score for right class
    right_class_scores = right_class_scores.reshape((num_train, 1))
    margins = scores - right_class_scores.dot(np.ones((1,num_classes))) + delta  # margins
    margins[np.arange(num_train), y] = 0  # remove margins for true classes
    margins = np.maximum(margins, 0)  # maximize
    losses = np.sum(margins, axis=1) # loss for each item
    loss = losses.sum() # all loss
    loss /= num_train # average loss
    loss += 0.5 * reg * np.sum(W * W) # add reg term to loss
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    binary_mask = margins.copy()  # this is d_L/d_score matrix !!!
    binary_mask[margins > 0] = 1   # 1 - where margins for incorrect classes more than for correct
    incorrect = binary_mask.sum(axis=1) # for each item how many incorrect classes are predicted
    binary_mask[np.arange(num_train), y] = -incorrect  # count the number of inc classes  
                                                        #(https://cs231n.github.io/optimization-1/)
    
    dW = X.T.dot(binary_mask)
    
    dW /= num_train # average out weights
    dW += reg*W # regularize the weights
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
