import numpy as np
from pathlib import Path
import argparse
from functions_final import *

np.random.seed(42)

def read_fasta(fasta_path):       #reads fasta file and returns sequences
    sequences = []
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))    #for final line not followed by '>'

    return sequences

#markov model

def train_markov_and_score(fasta_path, m):
    fasta_path = Path(fasta_path)

    sequences = read_fasta(fasta_path) #reading sequences from input file

    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}

    chars = ['A', 'C', 'G', 'T']
    pseudocount = 1

    counts = get_kmer_counts_vectorized(sequences, m, base_map)

    tm_log = markov_model_numpy(counts, m, pseudocount) #training markov model using all sequences

    for seq in sequences:
        ll = log_likelihood_numpy(tm_log, seq, m, base_map) #log likelihood for each sequence in fasta file
        print(ll)


def main():
    parser = argparse.ArgumentParser(description='Simpler Markov Model')
    parser.add_argument('fasta', type=Path, help='Path to FASTA file')
    parser.add_argument('order', type=int, help='Order of the Markov model (m)')

    args = parser.parse_args()

    train_markov_and_score(args.fasta, args.order)


if __name__ == '__main__':
    main()