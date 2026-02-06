"""
tree_utils.py

Defines two Python classes, one for each kind of nodes:
- InternalDecisionNode
- LeafNode

Your job is to edit the *predict* method for each node.

Examples
--------
>>> N = 6
>>> F = 1
>>> x_NF = np.linspace(-5, 5, N).reshape((N,F))
>>> y_N = np.hstack([np.linspace(0, 1, N//2), np.linspace(-1, 0, N//2)])

>>> feat_id = 0
>>> thresh_val = 0.0
>>> left_mask_N = x_NF[:, feat_id] < thresh_val
>>> right_mask_N = np.logical_not(left_mask_N)
>>> left_leaf = LeafNode(x_NF[left_mask_N], y_N[left_mask_N])
>>> right_leaf = LeafNode(x_NF[right_mask_N], y_N[right_mask_N])

>>> left_leaf.y_N
array([0. , 0.5, 1. ])

>>> root = InternalDecisionNode(
...     x_NF, y_N, feat_id, thresh_val, left_leaf, right_leaf)

# Display the tree
>>> print(root)
Decision: X[0] < 0.000?
  Y: Leaf: predict y = 0.500
  N: Leaf: predict y = -0.500

# Remember the true label of each node in train set
>>> y_N
array([ 0. ,  0.5,  1. , -1. , -0.5,  0. ])

# Predictions of the whole 3-node tree for each example in training set
>>> yhat_N = root.predict(x_NF)
>>> np.round(yhat_N, 4)
array([ 0.5,  0.5,  0.5, -0.5, -0.5, -0.5])

# Predictions of the left leaf for each example in training set:
>>> yhat_N = left_leaf.predict(x_NF)
>>> np.round(yhat_N, 4)
array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Predictions of the right leaf for each example in training set:
>>> yhat_N = right_leaf.predict(x_NF)
>>> np.round(yhat_N, 4)
array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])

# Predictions for new input never seen before
>>> np.round(root.predict(x_NF[::-1] + 1.23), 4)
array([-0.5, -0.5, -0.5, -0.5,  0.5,  0.5])

"""

"""
tree_utils.py

Defines two Python classes, one for each kind of nodes:
- InternalDecisionNode
- LeafNode

Completes the *predict* method for each node.
"""

import numpy as np

class InternalDecisionNode(object):

    '''
    Defines a single node used to make yes/no decisions within a binary tree.
    '''

    def __init__(self, x_NF, y_N, feat_id, thresh_val, left_child, right_child):
        self.x_NF = x_NF
        self.y_N = y_N
        self.feat_id = feat_id
        self.thresh_val = thresh_val
        self.left_child = left_child
        self.right_child = right_child


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        
        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        T, F = x_TF.shape
        yhat_T = np.zeros(T, dtype=np.float64)

        # Determine which of the input T examples belong to the left/right child
        left_mask_T = x_TF[:, self.feat_id] < self.thresh_val
        right_mask_T = np.logical_not(left_mask_T)

        # Ask the left child for its predictions
        if np.any(left_mask_T):
            x_L = x_TF[left_mask_T]
            yhat_T[left_mask_T] = self.left_child.predict(x_L)

        # Ask the right child for its predictions
        if np.any(right_mask_T):
            x_R = x_TF[right_mask_T]
            yhat_T[right_mask_T] = self.right_child.predict(x_R)
        
        return yhat_T


    def __str__(self):
        ''' Pretty print a string representation of this node '''
        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()
        lines = [
            "Decision: X[%d] < %.3f?" % (self.feat_id, self.thresh_val),
            "  Y: " + left_str.replace("\n", "\n    "),
            "  N: " + right_str.replace("\n", "\n    "),
            ]
        return '\n'.join(lines)


class LeafNode(object):
    
    '''
    Defines a single node within a binary tree that makes constant predictions.
    '''

    def __init__(self, x_NF, y_N):
        self.x_NF = x_NF
        self.y_N = y_N


    def predict(self, x_TF):
        ''' Make prediction given provided feature array
        
        Args
        ----
        x_TF : 2D numpy array, shape (T, F)

        Returns
        -------
        yhat_T : 1D numpy array, shape (T,)
        '''
        T = x_TF.shape[0]
        # Prediction is the mean of the training labels (y_N)
        yhat_val = np.mean(self.y_N)
        return yhat_val * np.ones(T)


    def __str__(self):
        ''' Pretty print a string representation of this node '''
        return "Leaf: predict y = %.3f" % np.mean(self.y_N)



# if __name__ == '__main__':
#     # Just does the same as doctest above, to help with debugging.

#     N = 6
#     F = 1
#     x_NF = np.linspace(-5, 5, N).reshape((N,F))
#     y_N = np.hstack([np.linspace(0, 1, N//2), np.linspace(-1, 0, N//2)])

#     feat_id = 0
#     thresh_val = 0.0
#     left_mask_N = x_NF[:, feat_id] < thresh_val
#     right_mask_N = np.logical_not(left_mask_N)
#     left_leaf = LeafNode(x_NF[left_mask_N], y_N[left_mask_N])
#     right_leaf = LeafNode(x_NF[right_mask_N], y_N[right_mask_N])
#     root = InternalDecisionNode(
#         x_NF, y_N, feat_id, thresh_val, left_leaf, right_leaf)

#     print("Displaying the tree")
#     print(root)

#     print("Predictions of the whole 3-node tree for each example in training set:")
#     yhat_N = root.predict(x_NF)
#     print(np.round(yhat_N, 4))
#     print("Predictions of the left leaf for each example in training set:")
#     yhat_N = left_leaf.predict(x_NF)
#     print(np.round(yhat_N, 4))
#     print("Predictions of the right leaf for each example in training set:")
#     yhat_N = right_leaf.predict(x_NF)
#     print(np.round(yhat_N, 4))