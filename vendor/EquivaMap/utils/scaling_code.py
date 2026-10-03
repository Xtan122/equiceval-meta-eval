import re
import os

def process_gurobi_code(input_filepath, output_filepath):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    # Read the Gurobi code file
    try:
        with open(input_filepath, 'r') as f:
            code_lines = f.readlines()
    except FileNotFoundError:
        print(f"Input file {input_filepath} does not exist.")
        return

    # Variables to store information
    continuous_vars = []
    var_scaling = {}
    new_code_lines = []

    # Regular expressions to identify variable declarations
    var_decl_pattern = re.compile(r'^(\s*)# @Variable (\w+) @Def: (.*?) @Shape: (.*)')
    addvar_pattern = re.compile(r'^\s*(\w+)\s*=\s*model\.addVar\((.*?)\)')
    addvars_pattern = re.compile(r'^\s*(\w+)\s*=\s*model\.addVars\((.*?)\)')

    # First pass to identify continuous variables and update descriptions
    i = 0
    while i < len(code_lines):
        line = code_lines[i]
        var_decl_match = var_decl_pattern.match(line)
        if var_decl_match:
            indent, var_name, var_desc, var_shape = var_decl_match.groups()
            # Look ahead to find the model.addVar(s) line
            next_line = code_lines[i+1] if i+1 < len(code_lines) else ''
            addvar_match = addvar_pattern.match(next_line)
            addvars_match = addvars_pattern.match(next_line)
            is_continuous = False
            if addvar_match and addvar_match.group(1) == var_name:
                # Single variable
                if 'vtype=GRB.CONTINUOUS' in addvar_match.group(2):
                    is_continuous = True
            elif addvars_match and addvars_match.group(1) == var_name:
                # Variable with indices
                if 'vtype=GRB.CONTINUOUS' in addvars_match.group(2):
                    is_continuous = True
            if is_continuous:
                scaling_factor = 10 ** (len(continuous_vars) + 1)
                continuous_vars.append(var_name)
                var_scaling[var_name] = scaling_factor
                # Update the variable description
                new_var_desc = f"{var_desc} ({scaling_factor} times before)"
                new_code_lines.append(f"{indent}# @Variable {var_name} @Def: {new_var_desc} @Shape: {var_shape}\n")
                i += 1  # Move to the next line
                new_code_lines.append(code_lines[i])  # Add the model.addVar(s) line as is
            else:
                new_code_lines.append(line)
        else:
            new_code_lines.append(line)
        i += 1

    # Second pass to substitute variables in constraints and objective functions
    substituted_code_lines = []
    for line in new_code_lines:
        stripped_line = line.strip()
        if 'model.addConstr' in stripped_line or 'model.setObjective' in stripped_line:
            # Only substitute in lines with 'model.addConstr' or 'model.setObjective'
            # For each continuous variable, replace its occurrences
            for var_name in continuous_vars:
                scaling_factor = var_scaling[var_name]
                # Build pattern to match variable usages with optional indices
                # We use word boundaries to ensure exact matches
                var_usage_pattern = re.compile(r'\b' + re.escape(var_name) + r'(\s*(?:\[[^\]]*\])*)\b')
                # Replace variable usages with scaled versions
                def replace_var_usage(match):
                    index = match.group(1) or ''
                    return f"(1/{scaling_factor}) * {var_name}{index}"
                line = var_usage_pattern.sub(replace_var_usage, line)
            substituted_code_lines.append(line)
        else:
            substituted_code_lines.append(line)

    # Write the modified code to the output file
    try:
        with open(output_filepath, 'w') as f:
            f.writelines(substituted_code_lines)
        print(f"Successfully wrote modified code to {output_filepath}")
    except Exception as e:
        print(f"Failed to write modified code to {output_filepath}: {e}")

# The rest of your code remains the same

def process_all_gurobi_files(base_dir):
    """
    Processes all Gurobi code files in the given directory structure.
    """
    for root, dirs, files in os.walk(base_dir):
        if 'optimus-code.py' in files and root.endswith('_0'):
            input_filepath = os.path.join(root, 'optimus-code.py')
            output_dir = root[:-2] + '_1'
            output_filepath = os.path.join(output_dir, 'optimus-code.py')
            process_gurobi_code(input_filepath, output_filepath)

if __name__ == "__main__":
    base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'
    process_all_gurobi_files(base_dir)