import glob
import subprocess
import os

base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

pattern = os.path.join(base_dir, "*", "*_c", "solve.py")
files_to_run = glob.glob(pattern)

time_limit = 20  # time limit in seconds

for fpath in files_to_run:
    print(f"Running: {fpath}")
    try:
        result = subprocess.run(["python", fpath], capture_output=True, text=True, timeout=time_limit)
        # If successful:
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
        print("Return code:", result.returncode)
    except subprocess.TimeoutExpired:
        print(f"Process timed out after {time_limit} seconds: {fpath}")
        # Continue to the next file without breaking out of the loop
    print("=======================================")
