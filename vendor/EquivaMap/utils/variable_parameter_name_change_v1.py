import json
import re
import os
import string

# Define the root directory path
root_directory = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

# Traverse all subdirectories and files
for subdir, _, files in os.walk(root_directory):
    # Only process directories ending in '_0' and containing problem_info.json
    if subdir.endswith('_0') and 'problem_info.json' in files:
        print(f'Processing directory: {subdir}')
        
        # Find the parent directory and original problem_info.json
        parent_dir = os.path.dirname(subdir)
        original_problem_info_path = os.path.join(parent_dir, 'problem_info.json')
        
        # Load JSON data from the original problem_info.json to create mappings
        with open(original_problem_info_path, 'r') as f:
            original_data = json.load(f)
        
        # Generate mappings from the original file's parameters and variables
        parameter_names = list(original_data['parameters'].keys())
        variable_names = list(original_data['variables'].keys())
        
        # Map parameters to uppercase letters and variables to lowercase letters
        capital_letters = list(string.ascii_uppercase)
        parameter_mapping = {name: capital_letters[i % 26] for i, name in enumerate(parameter_names)}
        
        lowercase_letters = list(string.ascii_lowercase)
        variable_mapping = {name: lowercase_letters[i % 26] for i, name in enumerate(variable_names)}

        # Function to replace indexed names in JSON
        def replace_indexed_names_in_json(data, param_mapping, var_mapping):
            json_str = json.dumps(data)
            # Replace parameter names with indices (e.g., AllocatedSpace_{I} to A_{I})
            for original, replacement in param_mapping.items():
                json_str = re.sub(rf'\b{re.escape(original)}(_\{{\w+\}}|\[\w+\])', rf'{replacement}\1', json_str)
            # Replace variable names with indices (e.g., AllocatedSpace_{I} to a_{I})
            for original, replacement in var_mapping.items():
                json_str = re.sub(rf'\b{re.escape(original)}(_\{{\w+\}}|\[\w+\])', rf'{replacement}\1', json_str)
            return json.loads(json_str)

        # Load the JSON data from the _0 directory’s problem_info.json
        problem_info_path = os.path.join(subdir, 'problem_info.json')
        with open(problem_info_path, 'r') as f:
            data = json.load(f)

        # Apply the mapping replacements to indexed names in the _0 file
        new_data = replace_indexed_names_in_json(data, parameter_mapping, variable_mapping)

        # Save the modified problem_info.json back to the _0 directory
        with open(problem_info_path, 'w') as f:
            json.dump(new_data, f, indent=4)

        print(f'Processed and saved modified problem_info.json in: {subdir}\n')
