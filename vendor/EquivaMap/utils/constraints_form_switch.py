import json
import os

def transform_formulation(formulation):
    # Skip transformation if there's a newline character
    if '\\n' in formulation:
        return formulation
    
    # Count occurrences of \leq and \geq
    leq_count = formulation.count('\\leq')
    geq_count = formulation.count('\\geq')
    
    # If there is more than one occurrence of \leq or \geq or both occur, don't transform
    if leq_count > 1 or geq_count > 1 or (leq_count == 1 and geq_count == 1):
        return formulation
    
    # Handle single \leq transformation
    if leq_count == 1:
        left, right = formulation.split('\\leq')
        return right.strip() + ' \\geq ' + left.strip()
    
    # Handle single \geq transformation
    if geq_count == 1:
        left, right = formulation.split('\\geq')
        return right.strip() + ' \\leq ' + left.strip()
    
    # No transformations if none of the above conditions matched
    return formulation


def transform_json_file(input_path, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read input JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Transform constraints
    if 'constraints' in data:
        for constraint in data['constraints']:
            if 'formulation' in constraint:
                constraint['formulation'] = transform_formulation(constraint['formulation'])
    
    # Write transformed JSON to output file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Set your base directory
base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

# Walk through the directory structure
for root, dirs, files in os.walk(base_dir):
    if 'problem_info.json' in files:
        # Add this check to only process directories ending with '_b'
        if root.endswith('_b'):
            print(f"Processing directory: {root}")
            
        
            # Create output directory structure (_0 replaced by _1)
            rel_path = os.path.relpath(root, base_dir)
            path_parts = rel_path.split(os.sep)
            
            transformed_path_parts = []
            for part in path_parts:
                if part.endswith('_b'):
                    new_part = part[:-2] + '_b'  # replace _0 with _1
                    transformed_path_parts.append(new_part)
                else:
                    transformed_path_parts.append(part)
            
            transformed_rel_path = os.path.join(*transformed_path_parts)
            
            input_path = os.path.join(root, 'problem_info.json')
            output_path = os.path.join(base_dir, transformed_rel_path, 'problem_info.json')
            
            # Print input and output paths before transformation
            print(f"  Input: {input_path}")
            print(f"  Output: {output_path}")
            
            # Perform the transformation
            transform_json_file(input_path, output_path)
            
            # Print completion message for the file
            print("  Transformation complete for this file.\n")

print("All transformations complete.")
