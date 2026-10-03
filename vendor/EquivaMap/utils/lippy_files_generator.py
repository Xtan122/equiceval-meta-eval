import os
import json
import glob

# Base path
base_path = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# We assume that each directory under `sample-data-easy` looks like:
#   number/number_c/model_data.json
# For example:
#   1/1_c/model_data.json
#   2/2_c/model_data.json
#   ...
# We'll glob for all model_data.json files in this structure.
json_files = glob.glob(os.path.join(base_path, "*", "*_c", "model_data.json"))

for json_file in json_files:
    # Extract directory info
    # For example, if json_file = "/Users/.../1/1_c/model_data.json"
    #   dir_path = "/Users/.../1/1_c"
    dir_path = os.path.dirname(json_file)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    c_vec = data.get("objective_coeffs", [])
    a_matrix = data.get("A", [])
    b_vec = data.get("b", [])
    
    # Prepare paths
    lippy_file_path = os.path.join(dir_path, "solve.py")
    log_file_path = os.path.join(dir_path, "log.txt")
    
    # Generate Python code for lippy file
    # We'll write a code snippet that:
    # 1) Imports lippy
    # 2) Defines c_vec, a_matrix, b_vec from the JSON data
    # 3) Creates a CuttingPlaneMethod instance
    # 4) Redirects stdout to "log.txt" and solves the problem
    # Note: The redirection of stdout is a simple Python trick. 
    # If lippy provides a built-in way to write logs to a file, you can use that instead.
    
    code = f"""import sys
import os
import lippy as lp

c_vec = {c_vec}
a_matrix = {a_matrix}
b_vec = {b_vec}

gomory = lp.CuttingPlaneMethod(c_vec, a_matrix, b_vec, log_mode=lp.LogMode.MEDIUM_LOG)

# Redirect stdout to the log file
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
with open(log_file_path, "w") as f:
    sys.stdout = f
    gomory.solve()

# Reset stdout if needed
sys.stdout = sys.__stdout__
"""

    # Write the lippy file
    with open(lippy_file_path, 'w') as lf:
        lf.write(code)

print("Lippy files have been generated.")
