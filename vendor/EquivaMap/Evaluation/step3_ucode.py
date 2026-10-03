import os

# Base directory
base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'

# Iterate over the directories in base_dir
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    if os.path.isdir(dir_path):
        optimus_code_path = os.path.join(dir_path, 'optimus-code.py')
        if os.path.exists(optimus_code_path):
            # Construct the updated code path
            updated_code_path = os.path.join(dir_path, 'optimus-code_e.py')

            # Find the subdirectory with '_e' in its name
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and '_e' in d]
            if subdirs:
                subdir_name = subdirs[0]  # Assuming only one '_e' subdirectory
                subdir_path = os.path.join(dir_path, subdir_name)
                map_constraints_path = os.path.join(subdir_path, 'map_constraints.py')

                if os.path.exists(map_constraints_path):
                    # Read the original Gurobi code
                    with open(optimus_code_path, 'r') as f:
                        code_lines = f.readlines()

                    # Read the constraints from map_constraints.py
                    with open(map_constraints_path, 'r') as f:
                        constraints_lines = f.readlines()

                    # Define the rule for insertion
                    # Find the indices for the constraints section
                    constraints_start_idx = None
                    constraints_end_idx = None

                    for idx, line in enumerate(code_lines):
                        if line.strip() == '# Constraints':
                            constraints_start_idx = idx
                        elif line.strip().startswith('# Objective'):
                            constraints_end_idx = idx
                            break

                    if constraints_start_idx is None or constraints_end_idx is None:
                        print(f"Could not find the constraints section in the code for {dir_name}.")
                        continue
                    else:
                        # Insert the new constraints before the '# Objective' section
                        updated_code_lines = (
                            code_lines[:constraints_end_idx] + ['\n'] + constraints_lines + ['\n'] + code_lines[constraints_end_idx:]
                        )

                        # Update the output path in the code
                        for idx, line in enumerate(updated_code_lines):
                            # Look for the line where the solution is saved
                            if 'with open(' in line and 'solution.json' in line:
                                # Extract the path including quotes
                                start_idx = line.find('with open(') + len('with open(')
                                end_idx = line.find(',', start_idx)
                                original_path_with_quotes = line[start_idx:end_idx].strip()
                                
                                # Extract the file path without quotes
                                original_path = original_path_with_quotes.strip("'\"")
                                
                                # Append '_e' before '.json' in the filename
                                if 'solution.json' in original_path:
                                    new_file_path = original_path.replace('solution.json', 'solution_e.json')
                                else:
                                    print(f"Unexpected format in solution path in {dir_name}: {original_path}")
                                    continue
                                
                                # Build the new path with the original quotes
                                new_path_with_quotes = original_path_with_quotes.replace(original_path, new_file_path)
                                
                                # Replace the line with the updated path
                                updated_line = line.replace(original_path_with_quotes, new_path_with_quotes)
                                updated_code_lines[idx] = updated_line
                                break


                        # Write the updated code to optimus-code_e.py
                        with open(updated_code_path, 'w') as f:
                            f.writelines(updated_code_lines)

                        print(f"Updated code has been saved to {updated_code_path}")
                else:
                    print(f"map_constraints.py not found in {subdir_path}")
            else:
                print(f"No subdirectory with '_e' found in {dir_path}")
        else:
            print(f"optimus-code.py not found in {dir_path}")
