import sys
import subprocess
from pathlib import Path
import pandas as pd
import argparse
import datetime 

parser = argparse.ArgumentParser()
parser.add_argument("--order", default=10, type=int)
parser.add_argument("--k", default=3, type=int)
args = parser.parse_args()

k = args.k
max_order = args.order
input_file_path = Path("projectData/chr4_200bp_bins.tsv")
tf_name = 'CTCF'

output_filename = Path(f"log/ultimate_log_k_{k}_{input_file_path.stem}_{tf_name}.txt")

df_global = pd.DataFrame()

for m in range(max_order + 1):
    print(f"\n--- Starting subprocess for m={m} ---")
    
    command = [
        sys.executable, '-u', 'final_script.py', 
        '--order', str(m), 
        '--k', str(k), 
        '--input', str(input_file_path), 
        '--TF', tf_name
    ]
    
    output_buffer = []

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1) as proc:
        for line in proc.stdout: # type: ignore
            print(line, end='')  
            output_buffer.append(line)
            
    full_output = "".join(output_buffer)
    
    try:
        dict_start = full_output.rfind("{")
        dict_end = full_output.rfind("}") + 1
        
        if dict_start != -1 and dict_end > dict_start:
            dict_str = full_output[dict_start:dict_end]
            data_dict = eval(dict_str)
            
            df_m = pd.DataFrame(data_dict)
            df_global = pd.concat([df_global, df_m], ignore_index=True)
            
            if not df_m.empty:
                file_exists = output_filename.exists()
                df_m.to_csv(
                    output_filename, 
                    sep='\t', 
                    index=False, 
                    mode='a',
                    header=not file_exists
                )
                print(f"Saved m={m} to {output_filename}")
                
        else:
            print(f"\n[Error] No dictionary structure found for m={m}")

    except Exception as e:
        print(f"\n[Exception] Parsing failed: {e}")

print("\nFinal Global DataFrame:")
print(df_global)
