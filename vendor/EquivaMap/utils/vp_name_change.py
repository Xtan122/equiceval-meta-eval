import json
import re
import string
import os
import random

# Define the root directory path
root_directory = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

# Traverse all subdirectories and files
for subdir, dirs, files in os.walk(root_directory):
    # Calculate the depth of current directory relative to root
    relative_path = os.path.relpath(subdir, root_directory)
    depth = len(relative_path.split(os.sep))
    
    # Only process directories at depth 1 (immediate subdirectories of root)
    if depth != 1:
        continue
        
    # Check if the required files exist in the current directory
    if {'problem_info.json', 'parameters.json', 'optimus-code.py'}.issubset(set(files)):
        # Get the base name of the current directory
        base_name = os.path.basename(subdir)
        
        # Create an output directory
        output_directory = os.path.join(subdir, f'{base_name}_a')
        os.makedirs(output_directory, exist_ok=True)
        
        print(f'Processing directory: {subdir}')
        
        # Step 1: Load JSON data from problem_info.json
        problem_info_path = os.path.join(subdir, 'problem_info.json')
        with open(problem_info_path, 'r') as f:
            data = json.load(f)
        
        parameter_names = list(data['parameters'].keys())
        variable_names = list(data['variables'].keys())
        
        # Shuffle letters here for each directory, so we get a new mapping every time
        capital_letters = list(string.ascii_uppercase)
        lowercase_letters = list(string.ascii_lowercase)
        random.shuffle(capital_letters)
        random.shuffle(lowercase_letters)
        
        # Step 2: Create randomized mappings per directory
        parameter_mapping = {name: capital_letters[i % 26] for i, name in enumerate(parameter_names)}
        variable_mapping = {name: lowercase_letters[i % 26] for i, name in enumerate(variable_names)}
        
        # Function to replace names in JSON
        def replace_names_in_json(data, param_mapping, var_mapping):
            json_str = json.dumps(data)
            # Replace parameter names
            for original, replacement in param_mapping.items():
                json_str = re.sub(re.escape(original), replacement, json_str)
            # Replace variable names
            for original, replacement in var_mapping.items():
                json_str = re.sub(re.escape(original), replacement, json_str)
            return json.loads(json_str)
        
        new_data = replace_names_in_json(data, parameter_mapping, variable_mapping)
        
        # Save the modified problem_info.json
        new_problem_info_path = os.path.join(output_directory, 'problem_info.json')
        with open(new_problem_info_path, 'w') as f:
            json.dump(new_data, f, indent=4)
        
        # Step 3b: Replace names in parameters.json
        parameters_path = os.path.join(subdir, 'parameters.json')
        with open(parameters_path, 'r') as f:
            params = json.load(f)
        
        new_params = {}
        for key, value in params.items():
            new_key = parameter_mapping.get(key, key)
            new_params[new_key] = value
        
        # Save the modified parameters.json
        new_parameters_path = os.path.join(output_directory, 'parameters.json')
        with open(new_parameters_path, 'w') as f:
            json.dump(new_params, f, indent=4)
        
        # Step 3c: Replace names in optimus-code.py
        code_path = os.path.join(subdir, 'optimus-code.py')
        with open(code_path, 'r') as f:
            code = f.read()
        
        combined_mapping = {**parameter_mapping, **variable_mapping}
        sorted_names = sorted(combined_mapping.keys(), key=len, reverse=True)
        
        for name in sorted_names:
            replacement = combined_mapping[name]
            code = re.sub(re.escape(name), replacement, code)
        
        # Save the modified optimus-code.py
        new_code_path = os.path.join(output_directory, 'optimus-code.py')
        with open(new_code_path, 'w') as f:
            f.write(code)
        
        print(f'Processed and saved modified files in: {output_directory}\n')
