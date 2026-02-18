# BI3464_CFG-Project
January 2026 Semester. Course: BI3464 (CFG). Hitesh & Sindhuja.

# Running simplerVersion.py

## Description

This script implements an m-th order Markov model for DNA sequences.

It trains the model on all sequences contained in a given FASTA file and computes the log-likelihood of each sequence under the same trained model. One log-likelihood score is printed per line.

## Running code

Run from the terminal as:

python simplerVersion.py <fasta_file> <m>

Example:

python simplerVersion.py chr4_200bp_bins.fa 3

Arguments taken:

- fasta_file : Path to a FASTA file containing DNA sequences
- m          : Order of the Markov model

## Method

1. All sequences are read from the FASTA file using function fasta_read
2. An m-th order Markov model is trained using the full dataset
3. Transition probabilities are estimated from frequency counts
4. Log probabilities are computed
5. For each sequence, the total log-likelihood is calculated by summing log transition probabilities

## Output

The script prints one number per line.
Each number represents the total log-likelihood of a sequence under the trained model
