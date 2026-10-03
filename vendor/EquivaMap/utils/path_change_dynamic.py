import os
import re

# Define the root directory where your Gurobi code files are stored
root_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# Updated regex patterns to handle both numeric and alphabetic suffixes
data_load_pattern = re.compile(r'with open\("(\d+)/\d+_[0-9a-zA-Z]/parameters.json", "r"\) as f:')
solution_save_pattern = re.compile(r"with open\('(\d+)/\d+_[0-9a-zA-Z]/solution.json', 'w'\) as f:")

# Traverse the directory structure
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == "optimus-code.py":
            # Check if the directory depth is correct
            rel_path = os.path.relpath(subdir, root_dir)
            subdir_parts = rel_path.split(os.sep)

            # Ensure that the directory structure includes at least two levels
            if len(subdir_parts) >= 2:
                new_prefix = f"{subdir_parts[0]}/{subdir_parts[1]}"

                # Read the file content
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    content = f.read()

                # Replace the paths for loading data and saving solutions
                content = data_load_pattern.sub(f'with open("{new_prefix}/parameters.json", "r") as f:', content)
                content = solution_save_pattern.sub(f"with open('{new_prefix}/solution.json', 'w') as f:", content)

                # Write the modified content back to the file
                with open(file_path, 'w') as f:
                    f.write(content)

                print(f"Updated {file_path}")