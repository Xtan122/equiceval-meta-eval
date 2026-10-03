import os
import re
import json
import shutil

def extract_objective_components(objective_code):
    """
    Extracts the objective expression and optimization direction from the gurobipy code.
    Returns a tuple (expression, direction), where direction is 'MAXIMIZE' or 'MINIMIZE'.
    """
    # Pattern to match: model.setObjective(expression, GRB.MAXIMIZE) or GRB.MINIMIZE
    pattern = r'model\.setObjective\s*\(\s*(.+?)\s*,\s*GRB\.(MAXIMIZE|MINIMIZE)\s*\)'
    match = re.search(pattern, objective_code)
    if match:
        expression = match.group(1).strip()
        direction = match.group(2).strip()
        return expression, direction
    else:
        raise ValueError("Objective function format not recognized.")

def modify_json_data(json_data):
    """
    Modifies the JSON data by replacing the objective with variable zed and adding a constraint for zed.
    Returns the modified JSON data.
    """
    # Step 1: Extract the original objective function
    objective = json_data.get('objective', {})
    objective_code = objective.get('code', {}).get('gurobipy', '')
    
    try:
        original_expression, direction = extract_objective_components(objective_code)
    except ValueError as e:
        print(f"Error: {e}")
        return json_data  # Return unmodified if error occurs
    
    # Step 2: Add variable zed to the variables section
    if 'zed' not in json_data['variables']:
        json_data['variables']['zed'] = {
            'description': 'New variable representing the objective function',
            'type': 'continuous',
            'shape': []
        }
    else:
        print("Variable 'zed' already exists. Skipping addition.")
    
    # Step 3: Modify the objective to maximize or minimize zed based on original direction
    if direction == 'MAXIMIZE':
        new_direction = 'GRB.MAXIMIZE'
        new_formulation = 'Maximize \\ zed'
        new_description = 'Maximize the new variable zed.'
    elif direction == 'MINIMIZE':
        new_direction = 'GRB.MINIMIZE'
        new_formulation = 'Minimize \\ zed'
        new_description = 'Minimize the new variable zed.'
    else:
        # Default to maximize if direction is unrecognized
        new_direction = 'GRB.MAXIMIZE'
        new_formulation = 'Maximize \\ zed'
        new_description = 'Maximize the new variable zed.'
    
    json_data['objective'] = {
        'description': new_description,
        'formulation': new_formulation,
        'code': {
            'gurobipy': f'model.setObjective(zed, {new_direction})'
        }
    }
    
    # Step 4: Add constraint defining zed
    new_constraint = {
        'description': 'Constraint defining zed in terms of original variables.',
        'formulation': f'zed = {original_expression}',
        'code': {
            'gurobipy': f'model.addConstr(zed == {original_expression})'
        }
    }
    
    # Insert the new constraint at the beginning of the constraints list
    constraints = json_data.get('constraints', [])
    # Check if such a constraint already exists to avoid duplication
    existing_z_constraints = [
        c for c in constraints 
        if 'Constraint defining zed' in c.get('description', '')
    ]
    if not existing_z_constraints:
        constraints.insert(0, new_constraint)  # Insert at the beginning
    else:
        print("Constraint defining 'zed' already exists. Skipping addition.")
    json_data['constraints'] = constraints
    
    return json_data

def transform_gurobi_code(input_code):
    """
    Transforms Gurobi model code by replacing the objective function
    with a variable zed, adding a constraint defining zed, and updating
    the solution extraction code to include zed.
    
    Parameters:
    - input_code (str): The original Gurobi code as a string.
    
    Returns:
    - str: The transformed Gurobi code.
    """
    # Step 1: Parse the code to find the objective function
    pattern = r'model\.setObjective\((.+?),\s*(GRB\.(MAXIMIZE|MINIMIZE))\)'
    matches = re.findall(pattern, input_code, re.DOTALL)
    
    if not matches:
        raise ValueError("Objective function not found or in unexpected format.")
    
    # Assume the last occurrence is the actual objective function
    original_objective, direction_full, direction = matches[-1]
    original_objective = original_objective.strip()
    direction_full = direction_full.strip()
    direction = direction.strip()
    
    # Step 2: Check if variable 'zed' already exists
    if re.search(r'\n\s*zed\s*=', input_code):
        print("Variable 'zed' already exists in the code. Skipping addition of variable 'zed'.")
        z_defined = True
    else:
        z_defined = False
    
    # Step 3: Replace the objective function with 'zed'
    transformed_code = re.sub(
        r'model\.setObjective\(.+?,\s*GRB\.(MAXIMIZE|MINIMIZE)\)',
        f'model.setObjective(zed, {direction_full})',
        input_code
    )
    
    # Step 4: Add the definition of 'zed' if not already defined
    if not z_defined:
        # Find the variables section
        variables_match = re.search(
            r'(# Variables\s+)((?:.|\n)*?)(# Constraints)', transformed_code, re.MULTILINE
        )
        if variables_match:
            variables_section = variables_match.group(2)
            # Add 'zed' variable definition at the end of variables section
            z_var_code = '\n# @Variable zed @Def: New variable representing the objective function @Shape: []\nz = model.addVar(vtype=GRB.CONTINUOUS, name="zed")\n'
            new_variables_section = variables_section + z_var_code
            # Replace the old variables section with the new one
            transformed_code = transformed_code.replace(variables_section, new_variables_section)
        else:
            # If variables section not found, append at the end of variable declarations
            z_var_code = '\n# @Variable zed @Def: New variable representing the objective function @Shape: []\nz = model.addVar(vtype=GRB.CONTINUOUS, name="zed")\n'
            # Find last variable declaration
            last_var_match = re.search(r'(model\.addVar\(.*?\))', transformed_code, re.DOTALL | re.MULTILINE)
            if last_var_match:
                last_var = last_var_match.group(1)
                transformed_code = transformed_code.replace(
                    last_var,
                    last_var + z_var_code
                )
            else:
                # If no variable declarations found, add at the top
                transformed_code = z_var_code + transformed_code
        
    # Step 5: Add the constraint defining 'zed'
    z_constraint_code = f'\n# Constraint defining zed in terms of original variables\nmodel.addConstr(zed == {original_objective})\n'
    
    # Find where to insert the new constraint (at the end of constraints section)
    constraints_match = re.search(
        r'(# Constraints\s+)((?:.|\n)*?)(# Objective)', transformed_code, re.MULTILINE
    )
    if constraints_match:
        constraints_section = constraints_match.group(2)
        # Check if the constraint already exists
        if f'zed == {original_objective}' not in constraints_section:
            new_constraints_section = constraints_section + z_constraint_code
            # Replace the old constraints section with the new one
            transformed_code = transformed_code.replace(constraints_section, new_constraints_section)
        else:
            print("Constraint defining 'zed' already exists. Skipping addition.")
    else:
        # If constraints section not found, append at the end of the code
        transformed_code += z_constraint_code
    
    # Step 6: Add variables['zed'] = zed.x to the solution extraction code
    # New, simplified pattern to match variables block
    variables_block_pattern = r'(variables\s*=\s*\{\}\s*\n(?:\s*variables\[[^\]]+\]\s*=\s*.+\n)*)'
    variables_block_match = re.search(variables_block_pattern, transformed_code)
    
    # Debug information to help identify the block issue
    if variables_block_match:
        print("Variables assignment block found.")
        variables_block_full = variables_block_match.group(0)
        # Check if variables['zed'] = zed.x already exists
        if 'variables[\'zed\'] = zed.x' not in variables_block_full:
            new_assignment_line = 'variables[\'zed\'] = zed.x\n'  # Indented to match other assignments
            transformed_code = transformed_code.replace(variables_block_full, variables_block_full + new_assignment_line)
    else:
        print("Variables assignment block not found.")
    
    return transformed_code

def process_json_file(input_json_path, output_json_path):
    """
    Processes a JSON file: modifies it and writes the output to the specified path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON data
    try:
        with open(input_json_path, 'r') as infile:
            json_data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON file {input_json_path}: {e}")
        return
    except FileNotFoundError:
        print(f"Input file {input_json_path} does not exist.")
        return
    
    # Modify the JSON data
    modified_json_data = modify_json_data(json_data)
    
    # Write the modified JSON to the output file
    try:
        with open(output_json_path, 'w') as outfile:
            json.dump(modified_json_data, outfile, indent=4)
        print(f"Successfully wrote modified JSON to {output_json_path}")
    except Exception as e:
        print(f"Failed to write modified JSON to {output_json_path}: {e}")

def process_code_file(input_code_path, output_code_path):
    """
    Reads the Gurobi code from input_code_path, transforms it,
    and writes the transformed code to output_code_path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_code_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input code
    try:
        with open(input_code_path, 'r') as f:
            input_code = f.read()
    except FileNotFoundError:
        print(f"Input file {input_code_path} not found.")
        return
    
    # Transform the code
    try:
        transformed_code = transform_gurobi_code(input_code)
    except ValueError as e:
        print(f"Error during transformation: {e}")
        return
    
    # Write the transformed code to the output file
    with open(output_code_path, 'w') as f:
        f.write(transformed_code)
    
    print(f"Successfully wrote transformed code to {output_code_path}")

def copy_additional_files(input_dir, output_dir, files_to_copy):
    """
    Copies additional files from input_dir to output_dir.
    """
    for file_name in files_to_copy:
        src = os.path.join(input_dir, file_name)
        dst = os.path.join(output_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File {src} does not exist. Skipping.")

def main():
    base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'
    files_to_copy = ['parameters.json']  # List any additional files you want to copy
    
    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name.endswith('_c'):
                # Construct input and output directory paths
                input_dir = os.path.join(root, dir_name)
                output_dir = os.path.join(root, dir_name[:-1] + 'f')  # Replace '_8' with '_3'
                
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Process problem_info.json
                input_json_path = os.path.join(input_dir, 'problem_info.json')
                output_json_path = os.path.join(output_dir, 'problem_info.json')
                process_json_file(input_json_path, output_json_path)
                
                # Process optimus-code.py
                input_code_path = os.path.join(input_dir, 'optimus-code.py')
                output_code_path = os.path.join(output_dir, 'optimus-code.py')
                process_code_file(input_code_path, output_code_path)
                
                # Copy additional files
                copy_additional_files(input_dir, output_dir, files_to_copy)
                
                print(f"Processed directory {input_dir}")

        # No need to recurse further into subdirectories
        break  # Remove this line if you have nested directories

if __name__ == "__main__":
    main()
