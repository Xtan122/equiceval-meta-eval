import os

# The base directory containing the `optimus-code.py` files
base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

for root, dirs, files in os.walk(base_dir):
    if "optimus-code.py" in files:
        file_path = os.path.join(root, "optimus-code.py")
        
        # Read the file content
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Identify the subdirectory from the parameters.json loading line
        param_line_candidates = [l for l in lines if "with open(" in l and "parameters.json" in l]
        if not param_line_candidates:
            print(f"Skipping {file_path} - no parameters line found.")
            continue
        
        # Extract the subdirectory from the parameter line
        param_line = param_line_candidates[0]
        start_idx = param_line.find("with open(\"") + len("with open(\"")
        end_idx = param_line.find("/parameters.json")
        sub_path = param_line[start_idx:end_idx]  # e.g., "1/1_c"
        
        # Ensure the path contains "_d"
        if "_i" not in sub_path:
            print(f"Skipping {file_path} - extracted path '{sub_path}' does not contain '_i'.")
            continue
        
        # Process lines to remove any existing model.write lines and add the new one
        new_lines = []
        inserted = False
        for line in lines:
            # Skip lines that start with "model.write"
            if line.strip().startswith("model.write("):
                continue
            new_lines.append(line)
            # Add the new model.write line after model.optimize()
            if "model.optimize()" in line and not inserted:
                new_lines.append(f'model.write("{sub_path}/model.lp")\n')
                inserted = True
        
        # Write the modified file back
        with open(file_path, "w") as f:
            f.writelines(new_lines)

        print(f"Updated file: {file_path}")

