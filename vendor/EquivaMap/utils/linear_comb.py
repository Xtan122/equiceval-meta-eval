import json
import re
import os
import glob

def replace_variables_formulation(text, var_replacements):
    """
    Replaces variables in LaTeX formulations.
    """
    def replace_var(match):
        var_name = match.group(1)
        index = match.group(2) or ''
        if var_name in var_replacements:
            replacement = var_replacements[var_name]['formulation']
            if index:
                # Remove leading underscores and braces
                index = index.strip()
                index = index.lstrip('_')
                index = index.strip('{}')
                return replacement.format(index=index)
            else:
                return var_replacements[var_name]['formulation'].format(index='')
        else:
            return match.group(0)

    # Pattern to match LaTeX variables with optional subscripts: a_i, a_{i,j}, etc.
    pattern = re.compile(r'\\?([a-zA-Z][a-zA-Z0-9]*)\s*(?:_\{?([^{}\s]+)\}?)?')

    return pattern.sub(replace_var, text)

def replace_variables_code(text, var_replacements):
    """
    Replaces variables in code.
    """
    def replace_var(match):
        var_name = match.group(1)
        index = match.group(2) or ''
        if var_name in var_replacements:
            replacement = var_replacements[var_name]['code']
            index = index.strip()
            return replacement.format(index=index)
        else:
            return match.group(0)

    # Pattern to match code variables with optional indices: a[i], a[i][j], etc.
    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(\[[^\]]+\])?')

    return pattern.sub(replace_var, text)

def split_variables(json_data):
    """
    Splits each variable into two parts in the JSON data.
    Replaces occurrences of the original variables with the sum of the two new variables.
    """
    variables = json_data['variables']
    original_variables = list(variables.keys())

    # Build mappings for variable replacements
    var_replacements = {}

    for var_name in original_variables:
        var_info = variables[var_name]
        # Remove the original variable
        del variables[var_name]
        # Create var1 and var2
        for suffix in ['1', '2']:
            new_var_name = var_name + suffix
            new_var_info = var_info.copy()
            new_var_info['description'] = f"Part {suffix} of variable {var_name}: {var_info.get('description', '')}"
            variables[new_var_name] = new_var_info
        # Prepare replacements for indexed and non-indexed variables
        if 'shape' in var_info and var_info['shape']:
            # Variable is indexed
            var_replacements[var_name] = {
                'formulation': f"{var_name}1_{{{{index}}}} + {var_name}2_{{{{index}}}}",
                'code': f"({var_name}1{{index}} + {var_name}2{{index}})"
            }
        else:
            # Variable is scalar
            var_replacements[var_name] = {
                'formulation': f"{var_name}1 + {var_name}2",
                'code': f"({var_name}1 + {var_name}2)"
            }

    # Replace in constraints
    for constraint in json_data.get('constraints', []):
        # Replace in 'formulation'
        if 'formulation' in constraint:
            constraint['formulation'] = replace_variables_formulation(constraint['formulation'], var_replacements)
        # Replace in 'code'
        if 'code' in constraint:
            for code_lang, code_str in constraint['code'].items():
                constraint['code'][code_lang] = replace_variables_code(code_str, var_replacements)

    # Replace in objective
    objective = json_data.get('objective', {})
    if 'formulation' in objective:
        objective['formulation'] = replace_variables_formulation(objective['formulation'], var_replacements)
    if 'code' in objective:
        for code_lang, code_str in objective['code'].items():
            objective['code'][code_lang] = replace_variables_code(code_str, var_replacements)

    return json_data

def process_single_json(input_filepath, output_filepath):
    """
    Processes a single JSON file: modifies it and writes the output to the specified path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON data
    try:
        with open(input_filepath, 'r') as infile:
            json_data = json.load(infile)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON file {input_filepath}: {e}")
        return
    except FileNotFoundError:
        print(f"Input file {input_filepath} does not exist.")
        return

    # Modify the JSON data
    modified_json_data = split_variables(json_data)

    # Write the modified JSON to the output file
    try:
        with open(output_filepath, 'w') as outfile:
            json.dump(modified_json_data, outfile, indent=4)
        print(f"Successfully wrote modified JSON to {output_filepath}")
    except Exception as e:
        print(f"Failed to write modified JSON to {output_filepath}: {e}")

def process_all_problem_info_files():
    """
    Processes all problem_info.json files in directories ending with '_8'
    and outputs the transformed files to directories ending with '_6'.
    """
    input_files = glob.glob('/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/*/*_c/problem_info.json')
    for input_file in input_files:
        # Get the directory of the input file
        input_dir = os.path.dirname(input_file)
        # Replace '_8' with '_6' in the directory name
        output_dir = input_dir.replace('_c', '_h')
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Construct the output file path
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        # Process the file
        process_single_json(input_file, output_file)

if __name__ == "__main__":
    process_all_problem_info_files()
