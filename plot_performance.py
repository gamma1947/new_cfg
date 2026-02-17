import matplotlib.pyplot as plt
import numpy as np

log_file = 'new_cfg/log/ultimate_log_k_5_chr4_200bp_bins_CTCF.txt'

# try:
with open(log_file, "r") as log:
    auc_roc = []
    auc_pr =[]
    time = []
    mem = []
    for l in log :
        next(log, None)
        line = l.strip()
        print(repr(line))
        
        elements = line.split('\t')
        # print(elements[7])

