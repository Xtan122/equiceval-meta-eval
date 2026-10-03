import glob
import subprocess
import os

base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# This pattern will find all optimus-code.py under directories ending with _c:
# e.g., /Users/stevenzhai/Desktop/MILP_data/sample-data-easy/*/*_c/optimus-code.py
pattern = os.path.join(base_dir, "*", "*_c", "optimus-code.py")

files_to_run = glob.glob(pattern)

for fpath in files_to_run:
    print(f"Running: {fpath}")
    result = subprocess.run(["python", fpath], capture_output=True, text=True)
    
    # Print stdout and stderr for debugging
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
    print("Return code:", result.returncode)
    print("=======================================")
