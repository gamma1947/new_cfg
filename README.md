# BI3464_CFG-Project (Hitesh C K, Sindhuja T)
January 2026 Semester. Course: BI3464 (CFG). Hitesh & Sindhuja.

# Requirements

- Python 3.10+
- Libraries: 'numpy', 'matplotlib', 'tqdm', 'scikit-learn', 'pandas'
- Standard Python libraries used: 'pathlib', 'argparse', 'time', 'tracemalloc', 'datetime'

Installation by pip intall numpy matplotlib tqdm scikit-learn pandas

## All Project Files

- 'functions_final.py' - helper functions
- 'multi_order_mm_run_v2.py'
- 'final_script.py' - script to run for final solutions to assigned questions
- 'simplerVersion.py' - script to run for given problem

- 'projectData' - contains provided chromosome 4 .tsv file
- 'fasta_outputs' - contains fasta file sequences extracted from tsv file
- 'log' - contains information about time taken and memory used while running file
- 'plots' - contains plots for each fold for k=5, m in the range(0,11) (1 to 10)

# Running simplerVersion.py

This script implements an m-th order Markov model for DNA sequences.

It trains the model on all sequences contained in a given FASTA file and computes the log-likelihood of each sequence under the same trained model. One log-likelihood score is printed per line.

Running 'simplerVersion.py' uses lesser number of libraries: 'numpy', 'argparse', 'pathlib'

## Running code

Run from the terminal as:

python simplerVersion.py <fasta_file> <m>

Example:

python simplerVersion.py chr4_200bp_bins.fa 3

Arguments taken:

- fasta_file : Path to a FASTA file containing DNA sequences
- m          : Order of the Markov model

## Output

The script prints one number per line.
Each number represents the total log-likelihood of a sequence under the trained model
