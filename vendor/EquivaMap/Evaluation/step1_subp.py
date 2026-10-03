import os
import subprocess
import glob

# Base directory where your data is stored
base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

# Create a pattern to match directories ending with '_2'
pattern = os.path.join(base_dir, '*', '*_i*', 'optimus-code.py')

# Find all matching 'optimus-code.py' files
script_paths = glob.glob(pattern)

# Iterate over each script and execute it
for script_path in script_paths:
    try:
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Script {script_path} executed successfully.")
        print("Standard Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the script {script_path}.")
        print("Standard Error:")
        print(e.stderr)
