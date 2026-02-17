import numpy as np
from tqdm import tqdm
def seq_to_indices(seq, mapping):
    """Converts a DNA string to an integer array based on mapping."""
    # We map unknown chars (N) to 0 (A) to prevent crashes, or you can handle differently
    return np.array([mapping.get(s, 0) for s in seq], dtype=np.int32)

def get_kmer_counts_vectorized(sequences, m, mapping):
    """
    Counts k-mers (of length m+1) using integer arrays.
    Returns a flat array of size 4**(m+1).
    """
    # 4^(m+1) possible combinations
    total_bins = 4**(m+1)
    counts = np.zeros(total_bins, dtype=np.int32)
    
    # Pre-compute powers of 4 for converting sliding windows to integers
    # e.g., for 2-mer: [4, 1] -> index = char1*4 + char2*1
    powers = 4 ** np.arange(m, -1, -1) 
    
    for seq in tqdm.tqdm(sequences):
        if len(seq) < m + 1:
            continue
            
        # Convert string 'ACGT' to [0, 1, 2, 3]
        seq_arr = seq_to_indices(seq, mapping)
        
        # Create sliding window view
        # This creates a view of shape (len-m, m+1) without copying memory
        windows = np.lib.stride_tricks.sliding_window_view(seq_arr, window_shape=m+1)
        
        # Convert windows to single integer indices using dot product
        # fast way to do: char[0]*4^m + char[1]*4^{m-1} ...
        indices = np.dot(windows, powers)
        
        # Count occurrences of each index
        # np.bincount is extremely fast
        binc = np.bincount(indices, minlength=total_bins)
        counts += binc.astype(np.int32)
        
    return counts

def markov_model_numpy(counts_m_plus_1, m, pseudocount=1):
    """
    Calculates transition matrix from count array.
    counts_m_plus_1: Flat array of size 4^(m+1)
    """
    # Reshape into (4^m, 4)
    # Rows = Context (history), Cols = Next Character (A, T, G, C)
    counts_reshaped = counts_m_plus_1.reshape((-1, 4))
    
    # Add pseudocounts once (Vectorized)
    counts_reshaped = counts_reshaped + pseudocount
    
    # Calculate row sums (counts of the history context)
    row_sums = counts_reshaped.sum(axis=1, keepdims=True)
    
    # Divide to get probabilities
    transition_matrix = counts_reshaped / row_sums
    
    # Log transform for numerical stability (and speed in likelihood calc)
    log_trans_matrix = np.log(transition_matrix)
    
    return log_trans_matrix

def log_likelihood_numpy(log_tm, seq, m, mapping):
    """
    Calculates log likelihood using the pre-computed log transition matrix.
    """
    if len(seq) < m + 1:
        return 0.0

    seq_arr = seq_to_indices(seq, mapping)
    
    # Create windows for (History + Next)
    # We need to find the specific row (History) and col (Next) for every position
    powers_history = 4 ** np.arange(m-1, -1, -1) # Powers for history part
    
    # 1. Get History Indices (Rows)
    # Windows of size m
    if m > 0:
        windows_hist = np.lib.stride_tricks.sliding_window_view(seq_arr[:-1], window_shape=m)
        row_indices = np.dot(windows_hist, powers_history)
    else:
        # For m=0, history is empty, effectively row 0 (or we just use 0-order freq)
        row_indices = np.zeros(len(seq)-1, dtype=int)

    # 2. Get Next Char Indices (Cols)
    # The character immediately following the history window
    col_indices = seq_arr[m:]
    
    # 3. Sum the Log Probabilities
    # Use fancy indexing to grab all probabilities at once
    score = np.sum(log_tm[row_indices, col_indices])
    
    return score