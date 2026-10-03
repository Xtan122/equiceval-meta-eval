import json
import re
import os

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
                return replacement + '_{' + index + '}'
            else:
                return replacement
        else:
            return match.group(0)

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
            if index:
                return replacement + index
            else:
                return replacement
        else:
            return match.group(0)

    pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(\[[^\]]+\])?')
    return pattern.sub(replace_var, text)

def get_continuous_vars_from_json(json_filepath):
    """
    Extracts continuous variables from problem_info.json
    """
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
        return {var_name for var_name, var_info in json_data.get('variables', {}).items() 
                if var_info.get('type') == 'continuous'}
    except Exception as e:
        print(f"Error reading JSON file {json_filepath}: {e}")
        return set()

def get_continuous_vars_from_gurobi(gurobi_filepath):
    """
    Extracts continuous variables from optimus-code.py
    """
    continuous_vars = set()
    try:
        with open(gurobi_filepath, 'r') as f:
            code_lines = f.readlines()
        
        var_decl_pattern = re.compile(r'^(\s*)# @Variable (\w+)')
        addvar_pattern = re.compile(r'^\s*(\w+)\s*=\s*model\.addVar\((.*?)\)')
        addvars_pattern = re.compile(r'^\s*(\w+)\s*=\s*model\.addVars\((.*?)\)')
        
        i = 0
        while i < len(code_lines):
            line = code_lines[i]
            var_decl_match = var_decl_pattern.match(line)
            if var_decl_match:
                var_name = var_decl_match.group(2)
                next_line = code_lines[i+1] if i+1 < len(code_lines) else ''
                addvar_match = addvar_pattern.match(next_line)
                addvars_match = addvars_pattern.match(next_line)
                
                if ((addvar_match and addvar_match.group(1) == var_name and 
                     'vtype=GRB.CONTINUOUS' in addvar_match.group(2)) or
                    (addvars_match and addvars_match.group(1) == var_name and 
                     'vtype=GRB.CONTINUOUS' in addvars_match.group(2))):
                    continuous_vars.add(var_name)
            i += 1
        return continuous_vars
    except Exception as e:
        print(f"Error reading Gurobi file {gurobi_filepath}: {e}")
        return set()

def process_json_file(json_filepath, output_filepath, common_continuous):
    """
    Process the JSON file with filtered continuous variables
    """
    try:
        with open(json_filepath, 'r') as f:
            json_data = json.load(f)
        
        variables = json_data['variables']
        var_replacements = {}
        continuous_count = 0
        
        for var_name in common_continuous:
            if var_name in variables:
                continuous_count += 1
                scaling_factor = 10 ** continuous_count
                var_info = variables[var_name]
                original_description = var_info.get('description', '')
                var_info['description'] = f"{original_description} ({scaling_factor} times before)"
                var_replacements[var_name] = {
                    'scaling_factor': scaling_factor,
                    'formulation': f"\\frac{{1}}{{{scaling_factor}}} {var_name}",
                    'code': f"(1/{scaling_factor}) * {var_name}"
                }
        
        if var_replacements:
            for constraint in json_data.get('constraints', []):
                if 'formulation' in constraint:
                    constraint['formulation'] = replace_variables_formulation(
                        constraint['formulation'], var_replacements)
                if 'code' in constraint:
                    for code_lang, code_str in constraint['code'].items():
                        constraint['code'][code_lang] = replace_variables_code(
                            code_str, var_replacements)
            
            objective = json_data.get('objective', {})
            if 'formulation' in objective:
                objective['formulation'] = replace_variables_formulation(
                    objective['formulation'], var_replacements)
            if 'code' in objective:
                for code_lang, code_str in objective['code'].items():
                    objective['code'][code_lang] = replace_variables_code(
                        code_str, var_replacements)
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Successfully processed JSON file: {output_filepath}")
        
    except Exception as e:
        print(f"Error processing JSON file {json_filepath}: {e}")

def process_gurobi_file(gurobi_filepath, output_filepath, common_continuous):
    """
    Process the Gurobi file with filtered continuous variables
    """
    try:
        with open(gurobi_filepath, 'r') as f:
            code_lines = f.readlines()
        
        var_scaling = {}
        continuous_count = 0
        new_code_lines = []
        
        # First pass: update descriptions and build scaling factors
        i = 0
        while i < len(code_lines):
            line = code_lines[i]
            var_decl_match = re.compile(r'^(\s*)# @Variable (\w+)').match(line)
            if var_decl_match:
                var_name = var_decl_match.group(2)
                if var_name in common_continuous:
                    continuous_count += 1
                    scaling_factor = 10 ** continuous_count
                    var_scaling[var_name] = scaling_factor
                    desc_pattern = r'(@Def: .*?)(?= @Shape:|$)'
                    updated_line = re.sub(desc_pattern, 
                                       lambda m: f"{m.group(1)} ({scaling_factor} times before)", 
                                       line)
                    new_code_lines.append(updated_line)
                else:
                    new_code_lines.append(line)
            else:
                new_code_lines.append(line)
            i += 1
        
        # Second pass: substitute variables in constraints and objective
        final_lines = []
        for line in new_code_lines:
            if 'model.addConstr' in line or 'model.setObjective' in line:
                for var_name, scaling_factor in var_scaling.items():
                    pattern = r'\b' + re.escape(var_name) + r'(\s*(?:\[[^\]]*\])*)\b'
                    line = re.sub(pattern, 
                                lambda m: f"(1/{scaling_factor}) * {var_name}{m.group(1)}", 
                                line)
            final_lines.append(line)
        
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w') as f:
            f.writelines(final_lines)
        print(f"Successfully processed Gurobi file: {output_filepath}")
        
    except Exception as e:
        print(f"Error processing Gurobi file {gurobi_filepath}: {e}")

def process_directory(base_dir):
    """
    Process all matching file pairs in the directory
    """
    for root, dirs, files in os.walk(base_dir):
        if root.endswith('_c') and 'problem_info.json' in files and 'optimus-code.py' in files:
            json_filepath = os.path.join(root, 'problem_info.json')
            gurobi_filepath = os.path.join(root, 'optimus-code.py')
            
            # Get continuous variables from both files
            json_continuous = get_continuous_vars_from_json(json_filepath)
            gurobi_continuous = get_continuous_vars_from_gurobi(gurobi_filepath)
            
            # Only process variables that are continuous in both files
            common_continuous = json_continuous.intersection(gurobi_continuous)
            
            # Create output directory paths
            output_dir = root[:-2] + '_i'
            json_output = os.path.join(output_dir, 'problem_info.json')
            gurobi_output = os.path.join(output_dir, 'optimus-code.py')
            
            # Process both files
            process_json_file(json_filepath, json_output, common_continuous)
            process_gurobi_file(gurobi_filepath, gurobi_output, common_continuous)

if __name__ == "__main__":
    base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'
    process_directory(base_dir)