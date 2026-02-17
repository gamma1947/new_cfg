import numpy as np
from tqdm import tqdm
def seq_to_indices(seq, mapping):
    return np.array([mapping.get(s, 0) for s in seq], dtype=np.int32)

def get_kmer_counts_vectorized(sequences, m, mapping):
    
    total_bins = 4**(m+1)
    counts = np.zeros(total_bins, dtype=np.int32)
    
    powers = 4 ** np.arange(m, -1, -1) 
    
    for seq in tqdm(sequences):
        if len(seq) < m + 1:
            continue
        
        seq_arr = seq_to_indices(seq, mapping)

        windows = np.lib.stride_tricks.sliding_window_view(seq_arr, window_shape=m+1)
        
        indices = np.dot(windows, powers)
        
        binc = np.bincount(indices, minlength=total_bins)
        counts += binc.astype(np.int32)
        
    return counts

def markov_model_numpy(counts_m_plus_1, m, pseudocount=1):
    
    counts_reshaped = counts_m_plus_1.reshape((-1, 4))
    
    
    counts_reshaped = counts_reshaped + pseudocount
    
    
    row_sums = counts_reshaped.sum(axis=1, keepdims=True)

    transition_matrix = counts_reshaped / row_sums
 
    log_trans_matrix = np.log(transition_matrix)
    
    return log_trans_matrix

def log_likelihood_numpy(log_tm, seq, m, mapping):
    
    if len(seq) < m + 1:
        return 0.0

    seq_arr = seq_to_indices(seq, mapping)
    
    
    powers_history = 4 ** np.arange(m-1, -1, -1) # Powers for history part
    
    
    if m > 0:
        windows_hist = np.lib.stride_tricks.sliding_window_view(seq_arr[:-1], window_shape=m)
        row_indices = np.dot(windows_hist, powers_history)
    else:
        row_indices = np.zeros(len(seq), dtype=int)

    col_indices = seq_arr[m:]

    score = np.sum(log_tm[row_indices, col_indices])
    
    return score