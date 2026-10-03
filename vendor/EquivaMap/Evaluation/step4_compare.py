import os
import subprocess
import json

# ---------------------------------
# 1. Helper function to detect LP/MIP from "optimus-code_e.py"
# ---------------------------------
def get_problem_type_from_e(dir_path):
    """
    Reads the 3rd line of optimus-code_e.py (or whichever line you need),
    expecting "# Problem type: LP" or "# Problem type: MIP".
    
    Returns 'LP', 'MIP', or None if it fails to detect.
    """
    e_file = os.path.join(dir_path, "optimus-code_e.py")
    if not os.path.isfile(e_file):
        return None
    
    with open(e_file, "r") as f:
        lines = f.readlines()
    
    # Make sure we have at least 3 lines
    if len(lines) < 3:
        return None

    # The third line, for instance, might be "# Problem type: LP"
    # Adjust index if your file has a different structure
    line3 = lines[2].strip()
    if "# Problem type: LP" in line3:
        return "LP"
    elif "# Problem type: MIP" in line3:
        return "MIP"
    
    return None


def main():
    base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'

    # We store data in a structure keyed by problem type (LP or MIP).
    results = {
        "LP": {
            "processed_dirs": [],
            "error_dirs": [],
            "total_files": 0,
            "same_objectives": 0,
            "different_objectives": [],
        },
        "MIP": {
            "processed_dirs": [],
            "error_dirs": [],
            "total_files": 0,
            "same_objectives": 0,
            "different_objectives": [],
        }
    }

    # We'll track total runtime errors too (across LP and MIP).
    runtime_errors = 0

    # Tolerance for floating-point comparisons
    tolerance = 1e-6

    # ---------------------------------
    # 2a. Iterate over directories
    # ---------------------------------
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        # Figure out if it's LP or MIP by inspecting optimus-code_e.py
        problem_type = get_problem_type_from_e(dir_path)
        if problem_type not in ("LP", "MIP"):
            # No recognized problem type => skip
            print(f"Skipping '{dir_name}' (could not determine LP or MIP from optimus-code_e.py).")
            continue

        # Make sure optimus-code_e.py actually exists (our function already checked, but let's be safe)
        optimus_code_e_path = os.path.join(dir_path, 'optimus-code_e.py')
        if not os.path.exists(optimus_code_e_path):
            print(f"optimus-code_e.py not found in {dir_path}")
            continue

        # Try running optimus-code_e.py
        try:
            result = subprocess.run(
                ['python', optimus_code_e_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Executed script in {dir_path} successfully.")
            print("Standard Output:")
            print(result.stdout)

            # If script runs, we add to 'processed_dirs' for that problem type
            results[problem_type]["processed_dirs"].append(dir_name)

        except subprocess.CalledProcessError as e:
            # This means a runtime error occurred (non-zero exit code)
            print(f"An error occurred while executing the script in {dir_path}.")
            print("Standard Error:")
            print(e.stderr)
            results[problem_type]["error_dirs"].append(dir_name)
            runtime_errors += 1
            # Skip the rest of the logic if there's a runtime error
            continue

        # ---------------------------------
        # 2b. Compare solutions if both exist
        # ---------------------------------
        solution_path = os.path.join(dir_path, 'solution.json')
        solution_e_path = os.path.join(dir_path, 'solution_e.json')

        if os.path.exists(solution_path) and os.path.exists(solution_e_path):
            # We'll increment the total files count for the problem type
            results[problem_type]["total_files"] += 1

            with open(solution_path, 'r') as f:
                solution = json.load(f)
                objective = solution.get('objective')

            with open(solution_e_path, 'r') as f:
                solution_e = json.load(f)
                objective_e = solution_e.get('objective')

            # Check if 'objective' keys exist
            if objective is None or objective_e is None:
                print(f"Objective not found in one of the solution files in {dir_path}")
                continue

            # Compare
            if abs(objective - objective_e) <= tolerance:
                results[problem_type]["same_objectives"] += 1
            else:
                results[problem_type]["different_objectives"].append(dir_name)
        else:
            print(f"One or both solution files are missing in {dir_path}")

    # ---------------------------------
    # 2c. Print Summary
    # ---------------------------------
    print(f"\nNumber of runtime errors (across all types): {runtime_errors}")

    for ptype in ("LP", "MIP"):
        processed_dirs = results[ptype]["processed_dirs"]
        error_dirs = results[ptype]["error_dirs"]
        total_files = results[ptype]["total_files"]
        same_objs = results[ptype]["same_objectives"]
        diff_objs = results[ptype]["different_objectives"]

        print(f"\n=== Summary for {ptype} problems ===")
        print(f"Processed Directories: {processed_dirs}")
        print(f"Error Directories: {error_dirs}")
        print(f"Total solution files compared: {total_files}")
        print(f"Number of same objectives: {same_objs}")
        print(f"Number of different objectives: {total_files - same_objs}")
        if diff_objs:
            print("Directories with different objectives:")
            for d in diff_objs:
                print(f"- {d}")

if __name__ == "__main__":
    main()
