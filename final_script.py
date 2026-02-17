import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import argparse 
import tqdm as tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as met
from functions_final import * # Import the NEW functions
import time
import tracemalloc
import pandas as pd
from datetime import datetime

np.random.seed(42)

def the_ultimate_function(tsv_path, tf_name, m = 1, k = 3):
    tsv_file = Path(tsv_path)

    TF_indexes = [3,4,5,6]
    
    TF_names = []
    with open(tsv_file, "r") as f:
        for i,l in enumerate(f):
            l = l.strip()
            elements = l.split('\t')
            if i == 0:
                TF_names.extend([elements[j] for j in TF_indexes])
            else:
                break 

    TF_dict = dict(zip(TF_names,TF_indexes))
    TF_idx = TF_dict[tf_name]

    with open(tsv_file, "r") as f:
        next(f, None)
        binding_info = []
        for l in f:
            elements = l.split('\t')
            u_b_string = "".join(elements[TF_idx])
            binding_info.append(u_b_string.strip())

    binding_info = np.array(binding_info)
    binding_info_dict = {'U': np.where(binding_info == 'U')[0], 
                        'B': np.where(binding_info == 'B')[0]}

    # Reading Fasta
    fasta_file = Path(f"./fasta_outputs/{tsv_file.stem}.fa")
    seq_dict = dict()

    with open(fasta_file , 'r') as ff:
        seq_idx = -1
        for l in ff:
            l = l.strip()
            if l.startswith('>'):
                seq_idx += 1
            else:
                if seq_idx in seq_dict:
                    seq_dict[seq_idx] += l
                else:
                    seq_dict[seq_idx] = l

    # Optimization: Pre-define integer mapping
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    
    chars = ['A', 'C', 'G', 'T'] # Order matters: must match 0,1,2,3
    pseudocount = 1

    U_idx = binding_info_dict['U']
    B_idx = binding_info_dict['B']
    np.random.shuffle(U_idx)
    np.random.shuffle(B_idx)
    U_split = np.array_split(U_idx, k)
    B_split = np.array_split(B_idx, k)        

    log_dict = {
                'chrm':[], 'TF':[], 'order':[], 'fold_idx':[],
                'start_time':[], 'stop_time':[], 'time_elapsed':[],
                'AOC-ROC':[], 'AOC-PR':[], 'memory_used (MB)':[]
                }
    
    for i in tqdm.tqdm(range(k)):
        tracemalloc.start()
        tic = time.perf_counter()
        start_time = datetime.now()
        
        train_indices = [j for j in range(k) if j != i]
        train_block_pos_idx = np.concatenate([B_split[l] for l in train_indices])
        train_block_neg_idx = np.concatenate([U_split[l] for l in train_indices])
        
        # Get actual sequences
        train_seqs_pos = [seq_dict[idx] for idx in train_block_pos_idx]
        train_seqs_neg = [seq_dict[idx] for idx in train_block_neg_idx]
        
        test_block_pos = set(B_split[i])
        test_block_neg = set(U_split[i])

        tqdm.tqdm.write(f"training started for fold-{i}")

        # 1. Count (m+1)-mers
        counts_pos = get_kmer_counts_vectorized(train_seqs_pos, m, base_map)
        counts_neg = get_kmer_counts_vectorized(train_seqs_neg, m, base_map)
        
        # 2. Build Markov Model (Transition Matrix)
        # Returns Log-Likelihood Matrix
        tm_pos_log = markov_model_numpy(counts_pos, m, pseudocount)
        tm_neg_log = markov_model_numpy(counts_neg, m, pseudocount)

        tqdm.tqdm.write(f"training completed for fold-{i}...starting test")
        
        log_likelihood_scores = []
        y_true = []


        test_indices = list(test_block_pos | test_block_neg)
        
        for idx in tqdm.tqdm(test_indices):
            seq = seq_dict[idx]
            
            # Calculate score using the log-matrices
            ll_pos = log_likelihood_numpy(tm_pos_log, seq, m, base_map)
            ll_neg = log_likelihood_numpy(tm_neg_log, seq, m, base_map)
            
            score = ll_pos - ll_neg
            log_likelihood_scores.append(score)

            if idx in test_block_pos:
                y_true.append(1)
            else:
                y_true.append(0)

        # Metrics Calculation
        fpr, tpr, thresholds = met.roc_curve(y_true, log_likelihood_scores)
        roc_auc = met.roc_auc_score(y_true, log_likelihood_scores)
        precision, recall, thresholds = met.precision_recall_curve(y_true, log_likelihood_scores)
        pr_auc = met.average_precision_score(y_true, log_likelihood_scores)
        
        tqdm.tqdm.write(f"Area Under Curve (AUC): {roc_auc:.4f}")
        tqdm.tqdm.write(f"AUC-PR (Average Precision): {pr_auc:.4f}")
        
        toc = time.perf_counter()
        end_time = datetime.now()
        t = toc - tic
        
        # Plotting (Kept mostly same, added check for display)
        baseline = sum(y_true) / len(y_true)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (0.5)')
        ax1.set_title(f'ROC (Order {m})')
        ax1.legend(loc="lower right")

        ax2.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        ax2.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.2f})')
        ax2.set_title(f'PR Curve (Order {m})')
        ax2.legend(loc="upper right")

        plt.tight_layout()
        output_dir = Path("./plots")
        output_dir.mkdir(exist_ok=True) # Ensure directory exists
        output_file = output_dir / f"metrics_plot_{m}_{tsv_file.stem}_{tf_name}_fold_{i}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        log_dict['chrm'].append(tsv_file.stem[:4])
        log_dict['order'].append(m)
        log_dict['TF'].append(tf_name)
        log_dict['fold_idx'].append(f"{i}/{k}")
        log_dict['start_time'].append(start_time.time())
        log_dict['stop_time'].append(end_time.time())
        log_dict['time_elapsed'].append(round(t, 3))
        log_dict['AOC-ROC'].append(round(roc_auc, 3))
        log_dict['AOC-PR'].append(round(pr_auc, 3))
        
        _,peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        module_memory = tracemalloc.get_tracemalloc_memory()
        log_dict['memory_used (MB)'].append((peak-module_memory)/(1024**2))
        
    return log_dict

def main():
    parser = argparse.ArgumentParser(description="Markov Model Cross Validation")
    parser.add_argument('--order', type = int, default=1, help="Order of the Markov model(m)")
    parser.add_argument('--k', type = int, default=3, help = "k-value")
    parser.add_argument('--input', type = Path, required=True, help = 'enter the path of the tsv file')
    parser.add_argument('--TF', type = str, required=True, default='CTCF', help = 'enter the name of the transcription factor')
    
    args = parser.parse_args()
    m = args.order
    k = args.k
    tsv_file = Path(args.input)
    tf = args.TF
    
    print(f"Running with: Order={m}, K={k}, TF={tf}, File={tsv_file.name}")
    
    # Ensure fasta_outputs folder exists or code will fail
    Path("./fasta_outputs").mkdir(exist_ok=True)
    
    a = the_ultimate_function(tsv_file, tf, m, k)
    print(a)

if __name__ == '__main__':
    main()