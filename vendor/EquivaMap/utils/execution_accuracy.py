import os
import json

def main():
    base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"
    
    same_objective_dirs = []
    different_objective_dirs = []
    
    # List all items in base_dir; keep only those that are directories and look numeric
    candidates = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    
    for problem_dir in candidates:
        # Full path to the numeric directory
        problem_path = os.path.join(base_dir, problem_dir)
        
        # Look for an _d subdirectory (e.g. "208_c")
        c_subdir_name = f"{problem_dir}_i"
        c_subdir_path = os.path.join(problem_path, c_subdir_name)
        
        # Continue only if that _d directory exists
        if not os.path.isdir(c_subdir_path):
            continue
        
        # Path to optimus-code.py in the _d subdirectory
        code_file_path = os.path.join(c_subdir_path, "optimus-code.py")
        
        # If optimus-code.py doesn't exist, skip
        if not os.path.exists(code_file_path):
            continue
        
        # Read the third line to check the problem type
        with open(code_file_path, "r") as f:
            lines = f.readlines()
        
        # Make sure the file has at least three lines
        if len(lines) < 3:
            continue
        
        # The third line typically looks like: "# Problem type: LP"
        problem_type_line = lines[2].strip()
        
        # We only want to proceed if it's an LP problem
        if "# Problem type: MIP" not in problem_type_line:
            continue
        
        # Compare the solution JSON files for LP problems
        # (1) /N/solution.json
        # (2) /N/N_c/solution.json
        original_solution_file = os.path.join(problem_path, "solution.json")
        compared_solution_file = os.path.join(c_subdir_path, "solution.json")
        
        # Check if both solution files exist
        if not (os.path.exists(original_solution_file) and os.path.exists(compared_solution_file)):
            continue
        
        # Load both solutions
        with open(original_solution_file, "r") as f1:
            solution1 = json.load(f1)
        with open(compared_solution_file, "r") as f2:
            solution2 = json.load(f2)
        
        # Compare objective values
        obj1 = solution1.get("objective", None)
        obj2 = solution2.get("objective", None)
        
        # Track whether objectives are the same or different
        if abs(obj1 - obj2) < 1e-6:
            same_objective_dirs.append(problem_dir)
        else:
            different_objective_dirs.append(problem_dir)
    
    # Print results
    print("Directories with the SAME objective:")
    print(same_objective_dirs)
    print(f"Count: {len(same_objective_dirs)}\n")
    
    print("Directories with DIFFERENT objectives:")
    print(different_objective_dirs)
    print(f"Count: {len(different_objective_dirs)}")

if __name__ == "__main__":
    main()
