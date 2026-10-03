import os
import json

# Base directory
base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'

# Walk through the base directory
for root, dirs, files in os.walk(base_dir):
    # Skip subdirectories beyond the immediate children of base_dir
    if root.count(os.sep) > base_dir.count(os.sep) + 1:
        continue
    for dir_name in dirs:
        if '_e' in dir_name:
            # Construct the full path to the directory
            dir_path = os.path.join(root, dir_name)
            
            # Define the paths to the necessary files within this directory
            variable_mappings_path = os.path.join(dir_path, 'variable_mappings.json')
            solution_path = os.path.join(dir_path, 'solution.json')
            output_path = os.path.join(dir_path, 'map_constraints.py')
            
            # Check if variable_mappings.json and solution.json exist
            if os.path.exists(variable_mappings_path) and os.path.exists(solution_path):
                # Read the variable mappings
                with open(variable_mappings_path, 'r') as f:
                    variable_mappings = json.load(f)
                
                # Read the solution.json to get the values of rhs variables
                with open(solution_path, 'r') as f:
                    solution = json.load(f)
                
                variables_values = solution.get('variables', {})
                
                # Open map_constraints.py for writing
                with open(output_path, 'w') as f:
                    for lhs_var, terms in variable_mappings.items():
                        # Check if terms is a valid iterable
                        if not terms or not isinstance(terms, list):
                            print(f"Warning: 'terms' for '{lhs_var}' is not a valid list or is empty in {dir_path}. Skipping.")
                            continue
                        
                        # Initialize a dictionary to hold RHS values for each index
                        rhs_values = {}  # key: index (or None for scalar), value: accumulated rhs_value
                        
                        # Process each term in the mapping
                        try:
                            for term in terms:
                                # Additional check if term is a dict with required keys
                                if not isinstance(term, dict) or 'constant' not in term or 'variable' not in term:
                                    print(f"Warning: Invalid term format for '{lhs_var}' in {dir_path}. Skipping this term.")
                                    continue
                                
                                constant = term['constant']
                                rhs_variable = term['variable']
                                rhs_var_value = variables_values.get(rhs_variable)
                                
                                if rhs_var_value is None:
                                    print(f"Warning: Variable '{rhs_variable}' not found in solution at {dir_path}.")
                                    continue
                                
                                if isinstance(rhs_var_value, dict):
                                    # Multi-dimensional variable represented as dict
                                    for index, value in rhs_var_value.items():
                                        term_value = constant * value
                                        rhs_values[index] = rhs_values.get(index, 0) + term_value
                                elif isinstance(rhs_var_value, list):
                                    # Multi-dimensional variable represented as list
                                    for index, value in enumerate(rhs_var_value):
                                        term_value = constant * value
                                        rhs_values[index] = rhs_values.get(index, 0) + term_value
                                else:
                                    # Scalar variable
                                    term_value = constant * rhs_var_value
                                    rhs_values[None] = rhs_values.get(None, 0) + term_value
                        except TypeError as e:
                            # This will catch cases where something wasn't iterable
                            print(f"Warning: Encountered TypeError for '{lhs_var}' in {dir_path}: {e}. Skipping this variable.")
                            continue
                        
                        # Write constraints based on the accumulated rhs_values
                        if rhs_values:
                            if None in rhs_values and len(rhs_values) == 1:
                                # Scalar lhs_var and scalar rhs_value
                                rhs_value = rhs_values[None]
                                f.write(f"model.addConstr({lhs_var} == {rhs_value})\n")
                            else:
                                # Multi-dimensional lhs_var
                                for index, rhs_value in rhs_values.items():
                                    if index is None:
                                        # Scalar term mapped to multi-dimensional lhs_var
                                        print(f"Warning: Scalar term mapped to multi-dimensional lhs_var '{lhs_var}'.")
                                        continue
                                    # Write constraint for each index
                                    f.write(f"model.addConstr({lhs_var}[{index}] == {rhs_value})\n")
                        else:
                            print(f"Warning: No valid mapping for '{lhs_var}' in directory {dir_path}.")
                    
                    print(f"Processed directory: {dir_path}")
            else:
                print(f"Skipped directory {dir_path} (missing variable_mappings.json or solution.json)")
