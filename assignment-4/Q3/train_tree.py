import numpy as np

from tree_utils import LeafNode, InternalDecisionNode
from select_best_binary_split import select_best_binary_split

def train_tree_greedy(
        x_NF, y_N, depth,
        MAX_DEPTH=10,
        MIN_SAMPLES_INTERNAL=1,
        MIN_SAMPLES_LEAF=1):
    ''' Train a binary decision tree on provided dataset in greedy fashion.
    '''
    N, F = x_NF.shape
    # Ensure minimum internal node size is at least twice the minimum leaf size
    MIN_SAMPLES_INTERNAL = np.maximum(
        MIN_SAMPLES_INTERNAL, 2 * MIN_SAMPLES_LEAF)
    
    # Stopping criteria
    if depth >= MAX_DEPTH:
        return LeafNode(x_NF, y_N)
    elif N < MIN_SAMPLES_INTERNAL:
        return LeafNode(x_NF, y_N)
    else:
        # Find the best split
        feat_id, thresh_val, x_LF, y_L, x_RF, y_R = select_best_binary_split(
                x_NF, y_N, MIN_SAMPLES_LEAF)
        
        if feat_id is None:
            # Case where further split is not possible
            return LeafNode(x_NF, y_N)
        else:
            # Recursively call train_tree_greedy to build the left child
            left_child = train_tree_greedy(
                x_LF, y_L, depth + 1,
                MAX_DEPTH=MAX_DEPTH,
                MIN_SAMPLES_INTERNAL=MIN_SAMPLES_INTERNAL,
                MIN_SAMPLES_LEAF=MIN_SAMPLES_LEAF)
            
            # Recursively call train_tree_greedy to build the right child
            right_child = train_tree_greedy(
                x_RF, y_R, depth + 1,
                MAX_DEPTH=MAX_DEPTH,
                MIN_SAMPLES_INTERNAL=MIN_SAMPLES_INTERNAL,
                MIN_SAMPLES_LEAF=MIN_SAMPLES_LEAF)
            
            return InternalDecisionNode(
                x_NF, y_N, feat_id, thresh_val, left_child, right_child)