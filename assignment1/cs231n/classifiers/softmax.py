import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        shift_scores = scores - max(scores)
        unnorm_probs = np.exp(shift_scores)
        norm_probs = unnorm_probs/sum(unnorm_probs)
        
        loss += -np.log(unnorm_probs[y[i]]/sum(unnorm_probs))
        
        for j in range(num_classes): 
            dW[:,j] += X[i]* unnorm_probs[j]/sum(unnorm_probs)
            if j == y[i]:
                dW[:,j] -= X[i]
                                
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    
    dW /= num_train
    dW += reg*W
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W) # raw scores

    max_scores_per_item = np.max(scores, axis=1).reshape((num_train,1))
    shifted_scores = scores - max_scores_per_item.dot(np.ones((1, num_classes))) # for computational 
                                                                                 # stability

    unnorm_probs = np.exp(shifted_scores)

    sum_rows = unnorm_probs.sum(axis=1).reshape((num_train,1))
    norm_probs = unnorm_probs/sum_rows.dot(np.ones((1,num_classes)))  # normalize probabilities

    correct_class_probs = norm_probs[np.arange(num_train), y]
    loss = -np.log(correct_class_probs)  # loss by item
    loss = loss.sum() # all loss
    loss /= num_train  # average loss
    loss += 0.5 * reg * np.sum(W*W)
    
    #######  GRADIENT #########
    
    dScores = norm_probs.copy()
    dScores[np.arange(num_train), y] -= 1
    
    dW = X.T.dot(dScores)
    dW /= num_train  # average the weights
    
    dW += reg*W  #regularize the weights
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
