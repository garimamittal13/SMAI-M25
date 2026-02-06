import numpy as np

from tree_utils import LeafNode, InternalDecisionNode

def select_best_binary_split(x_NF, y_N, MIN_SAMPLES_LEAF=1):
    ''' Determine best single feature binary split for provided dataset
    '''
    N, F = x_NF.shape

    cost_F = np.inf * np.ones(F)
    thresh_val_F = np.zeros(F)
    
    # Calculate the cost of the current node (no split) for the comparison check
    y_mean = np.mean(y_N)
    current_node_cost = np.sum(np.square(y_N - y_mean))

    for f in range(F):

        xunique_U = np.unique(x_NF[:,f])
        
        if xunique_U.size < 2:
            cost_F[f] = np.inf
            continue
            
        possib_xthresh_V = 0.5 * (xunique_U[:-1] + xunique_U[1:])
        
        # Determine valid indices for splits according to MIN_SAMPLES_LEAF
        V_full = possib_xthresh_V.size
        m = MIN_SAMPLES_LEAF - 1
        valid_split_indices = np.arange(V_full)[m:(V_full - m)]
        possib_xthresh_V_constrained = possib_xthresh_V[valid_split_indices]
        V = possib_xthresh_V_constrained.size

        if V == 0:
            cost_F[f] = np.inf
            continue

        total_cost_V = np.inf * np.ones(V)
        
        for v_id, v_thresh in enumerate(possib_xthresh_V_constrained):
            # 1. Split data
            left_mask_N = x_NF[:, f] < v_thresh
            y_L = y_N[left_mask_N]
            y_R = y_N[~left_mask_N]
            
            # 3. Calculate mean prediction for each child
            mu_L = np.mean(y_L)
            mu_R = np.mean(y_R)
            
            # 4. Calculate total Squared Error (SSE) cost: SSE_L + SSE_R
            cost_L = np.sum(np.square(y_L - mu_L))
            cost_R = np.sum(np.square(y_R - mu_R))
            
            total_cost_V[v_id] = cost_L + cost_R
        
        
        # Check if the best split cost is strictly better than the current node cost.
        min_cost = np.min(total_cost_V)
        
        if min_cost >= current_node_cost:
            cost_F[f] = np.inf
            continue

        # Pick out the split candidate that has best cost
        chosen_v_id = np.argmin(total_cost_V)
        cost_F[f] = total_cost_V[chosen_v_id]
        thresh_val_F[f] = possib_xthresh_V_constrained[chosen_v_id]

    # Determine single best feature to use
    best_feat_id = np.argmin(cost_F)
    
    if not np.isfinite(cost_F[best_feat_id]):
        return (None, None, None, None, None, None)

    best_thresh_val = thresh_val_F[best_feat_id]
    
    ## Assemble the left and right child datasets
    left_mask_N = x_NF[:, best_feat_id] < best_thresh_val
    right_mask_N = np.logical_not(left_mask_N)
    x_LF, y_L = x_NF[left_mask_N], y_N[left_mask_N]
    x_RF, y_R = x_NF[right_mask_N], y_N[right_mask_N]

    return (best_feat_id, best_thresh_val, x_LF, y_L, x_RF, y_R)