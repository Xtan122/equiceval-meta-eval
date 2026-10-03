import json
import os
import re
import ast
import astor
import shutil

def normalize_code(code_str):
    """
    Normalizes code strings by removing whitespace, converting to lowercase,
    standardizing array indexing, and removing redundant parentheses.
    """
    # Remove all whitespace
    code_str = re.sub(r'\s+', '', code_str)
    # Convert to lowercase
    code_str = code_str.lower()
    # Standardize array indexing C[0, i] -> C[0][i]
    code_str = re.sub(r'\[(\w+),\s*(\w+)\]', r'[\1][\2]', code_str)
    # Remove redundant parentheses
    code_str = code_str.replace('(', '').replace(')', '')
    # Remove constraint names or additional arguments
    code_str = re.sub(r',["\'].*?["\']', '', code_str)
    return code_str

def has_variable_indices(constraint_expr):
    """
    Checks if the constraint expression contains variable indices.
    Returns True if any index is not purely digits, else False.
    """
    # Find all array accesses like C[j][i] or D[j]
    # Match patterns like C[j][i], C[1][i], D[j], etc.
    array_accesses = re.findall(r'\w+\[(\w+)\](?:\[(\w+)\])?', constraint_expr)
    for access in array_accesses:
        for idx in access:
            if idx and not idx.isdigit():
                return True
    return False

def modify_problem_info(json_data):
    """
    Modifies the JSON data by adding slack variables to inequality constraints
    that can be transformed in the code.
    Returns the modified JSON data and a mapping of original constraints to slack variable names.
    """
    # Initialize a counter for slack variables to ensure unique names
    slack_counter = 0

    # Dictionary to map original constraint code to slack variable names
    constraint_slack_map = {}

    # Get the list of constraints
    constraints = json_data.get('constraints', [])

    # List to hold new constraints after modification
    new_constraints = []

    # Iterate over the constraints to find inequality constraints
    for constraint in constraints:
        description = constraint.get('description', '')
        formulation = constraint.get('formulation', '')
        code_dict = constraint.get('code', {})
        gurobipy_code = code_dict.get('gurobipy', '')

        # Split multiple constraints in one code snippet
        code_snippets = gurobipy_code.strip().split('\n')
        for code_snippet in code_snippets:
            code_snippet = code_snippet.strip()
            if not code_snippet:
                continue

            # Skip constraints defined using generator expressions or addConstrs
            if 'addConstrs' in code_snippet:
                new_constraints.append(constraint)
                continue

            # Standardize array indexing in code_snippet
            code_snippet_std = re.sub(r'\[(\w+),\s*(\w+)\]', r'[\1][\2]', code_snippet)

            # Check if the constraint is an inequality
            if '<=' in code_snippet_std or '>=' in code_snippet_std:
                # Determine the type of inequality
                if '<=' in code_snippet_std:
                    inequality = '<='
                elif '>=' in code_snippet_std:
                    inequality = '>='
                else:
                    continue  # Not an inequality constraint

                # Extract the constraint expression inside model.addConstr()
                match = re.search(r'model\.addConstr\((.*)\)', code_snippet_std)
                if not match:
                    continue  # Could not parse constraint; skip processing
                constraint_expr = match.group(1)

                # Remove any additional arguments (e.g., constraint names)
                constraint_expr = re.sub(r',\s*["\'].*?["\']', '', constraint_expr)

                # Check for variable indices
                if has_variable_indices(constraint_expr):
                    # Skip adding slack variable for constraints with variable indices
                    new_constraints.append(constraint)
                    continue

                # Split the constraint expression into LHS and RHS
                lhs_rhs = re.split(r'<=|>=', constraint_expr)
                if len(lhs_rhs) != 2:
                    new_constraints.append(constraint)
                    continue  # Could not parse constraint; skip processing

                lhs = lhs_rhs[0].strip()
                rhs = lhs_rhs[1].strip()

                # Generate a unique slack variable name
                slack_var_name = f'slack_{slack_counter}'
                slack_counter += 1

                # Map the normalized code snippet to the slack variable name
                normalized_code = normalize_code(code_snippet_std)
                constraint_slack_map[normalized_code] = slack_var_name

                # Add the slack variable to the variables section
                if slack_var_name not in json_data['variables']:
                    json_data['variables'][slack_var_name] = {
                        'description': f'Slack variable for constraint: {description}',
                        'type': 'continuous',
                        'shape': []
                    }

                # Modify the constraint to include the slack variable
                if inequality == '<=':
                    new_constraint_expr = f"{lhs} + {slack_var_name} == {rhs}"
                    new_formulation = f"{lhs} + {slack_var_name} = {rhs}"
                else:  # '>='
                    new_constraint_expr = f"{lhs} - {slack_var_name} == {rhs}"
                    new_formulation = f"{lhs} - {slack_var_name} = {rhs}"

                # Update the constraint
                modified_constraint = {
                    "description": description + f" (Modified to include slack variable {slack_var_name})",
                    "formulation": new_formulation,
                    "code": {
                        "gurobipy": f"model.addConstr({new_constraint_expr})"
                    }
                }
                new_constraints.append(modified_constraint)
            else:
                # Not an inequality constraint; keep as is
                new_constraints.append(constraint)

    # Update the constraints in the JSON data
    json_data['constraints'] = new_constraints

    return json_data, constraint_slack_map

def modify_code(code_data, constraint_slack_map):
    """
    Modifies the code data by adding slack variables to inequality constraints
    based on the mapping from the problem_info.json file.
    Returns the modified code data.
    """
    # Create a class to transform the code AST
    class SlackVariableTransformer(ast.NodeTransformer):
        def __init__(self, constraint_slack_map):
            self.constraint_slack_map = constraint_slack_map
            self.slack_vars = []
            self.slack_var_extractions = []

        def visit_Call(self, node):
            # Check if the function call is model.addConstr
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'addConstr':
                # Process the constraint
                args = node.args
                if not args:
                    return node  # No arguments, skip

                # Extract the constraint expression
                constraint_expr = args[0]
                # Remove any additional arguments (e.g., constraint names)
                node_copy = ast.Call(func=node.func, args=[constraint_expr], keywords=[])
                original_code = astor.to_source(node_copy).strip()
                normalized_code = normalize_code(original_code)

                # Match the original constraint code to find the slack variable name
                slack_var_name = self.constraint_slack_map.get(normalized_code)

                if slack_var_name:
                    new_constraint_expr = self.process_constraint_expr(constraint_expr, slack_var_name)
                    if new_constraint_expr is not None:
                        args[0] = new_constraint_expr

                return node

            return self.generic_visit(node)

        def process_constraint_expr(self, expr, slack_var_name):
            if isinstance(expr, ast.Compare):
                # Handle inequality comparisons
                if len(expr.ops) != 1:
                    return None  # Complex comparison, skip

                op = expr.ops[0]
                if isinstance(op, (ast.LtE, ast.GtE)):
                    # Define slack variable (lb=0)
                    slack_var_def = ast.Assign(
                        targets=[ast.Name(id=slack_var_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(value=ast.Name(id='model', ctx=ast.Load()), attr='addVar', ctx=ast.Load()),
                            args=[],
                            keywords=[
                                ast.keyword(arg='lb', value=ast.Num(n=0)),
                                ast.keyword(arg='name', value=ast.Str(s=slack_var_name))
                            ]
                        )
                    )
                    self.slack_vars.append(slack_var_def)

                    # Modify the constraint expression
                    if isinstance(op, ast.LtE):
                        # expr.left + slack_var == expr.comparators[0]
                        new_left = ast.BinOp(
                            left=expr.left,
                            op=ast.Add(),
                            right=ast.Name(id=slack_var_name, ctx=ast.Load())
                        )
                    else:  # GtE
                        # expr.left - slack_var == expr.comparators[0]
                        new_left = ast.BinOp(
                            left=expr.left,
                            op=ast.Sub(),
                            right=ast.Name(id=slack_var_name, ctx=ast.Load())
                        )

                    new_expr = ast.Compare(
                        left=new_left,
                        ops=[ast.Eq()],
                        comparators=expr.comparators
                    )

                    # Add slack variable extraction
                    slack_var_extraction = ast.Assign(
                        targets=[ast.Subscript(
                            value=ast.Name(id='variables', ctx=ast.Load()),
                            slice=ast.Index(ast.Str(slack_var_name)),
                            ctx=ast.Store()
                        )],
                        value=ast.Attribute(
                            value=ast.Name(id=slack_var_name, ctx=ast.Load()),
                            attr='X',
                            ctx=ast.Load()
                        )
                    )
                    self.slack_var_extractions.append(slack_var_extraction)

                    return new_expr

            return None  # Not an inequality constraint or unsupported format

    # Parse the code into an AST
    tree = ast.parse(code_data)

    # Create a transformer and apply it to the AST
    transformer = SlackVariableTransformer(constraint_slack_map)
    transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Reconstruct the code from the AST
    transformed_code = astor.to_source(tree)

    # Insert slack variable definitions after model creation
    model_creation_pattern = r'(model\s*=\s*Model\(\))'
    slack_vars_code = '\n'.join([astor.to_source(node).strip() for node in transformer.slack_vars])

    transformed_code = re.sub(
        model_creation_pattern,
        r'\1\n' + slack_vars_code,
        transformed_code
    )

    # Insert slack variable extractions before solution extraction
    variables_extraction_pattern = r'(variables\s*=\s*\{\})'
    slack_vars_extraction_code = '\n'.join([astor.to_source(node).strip() for node in transformer.slack_var_extractions])

    transformed_code = re.sub(
        variables_extraction_pattern,
        r'\1\n' + slack_vars_extraction_code,
        transformed_code
    )

    return transformed_code

def process_files(input_json_path, input_code_path, output_json_path, output_code_path):
    """
    Processes the problem_info.json and code files, modifies them by adding slack variables,
    and writes the output to the specified paths.
    """
    # Ensure the output directories exist
    output_json_dir = os.path.dirname(output_json_path)
    output_code_dir = os.path.dirname(output_code_path)
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_code_dir, exist_ok=True)

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

    # Initialize variables section if it doesn't exist
    if 'variables' not in json_data:
        json_data['variables'] = {}

    # Read the input code
    try:
        with open(input_code_path, 'r') as f:
            code_data = f.read()
    except FileNotFoundError:
        print(f"Input file {input_code_path} not found.")
        return

    # Modify the JSON data and get the constraint_slack_map
    modified_json_data, constraint_slack_map = modify_problem_info(json_data)

    # Modify the code data based on the constraint_slack_map
    modified_code_data = modify_code(code_data, constraint_slack_map)

    # Write the modified JSON to the output file
    try:
        with open(output_json_path, 'w') as outfile:
            json.dump(modified_json_data, outfile, indent=4)
        print(f"Successfully wrote modified JSON to {output_json_path}")
    except Exception as e:
        print(f"Failed to write modified JSON to {output_json_path}: {e}")

    # Write the modified code to the output file
    with open(output_code_path, 'w') as f:
        f.write(modified_code_data)
    print(f"Successfully wrote transformed code to {output_code_path}")

def main():
    base_input_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'
    base_output_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'

    # Get a list of all problem directories in the base_input_dir
    problem_dirs = [d for d in os.listdir(base_input_dir) if os.path.isdir(os.path.join(base_input_dir, d))]

    for problem in problem_dirs:
        input_dir = os.path.join(base_input_dir, problem)
        output_dir = os.path.join(base_output_dir, problem)

        # Find all subdirectories ending with '_0' in the input_dir
        subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.endswith('_c')]

        for input_subdir in subdirs:
            # Construct the corresponding output_subdir by replacing '_0' with '_7'
            output_subdir = input_subdir[:-2] + '_g'

            # Full paths to input and output subdirectories
            input_subdir_path = os.path.join(input_dir, input_subdir)
            output_subdir_path = os.path.join(output_dir, output_subdir)

            # Ensure the output subdirectory exists
            os.makedirs(output_subdir_path, exist_ok=True)

            # Paths to input and output files
            input_json_path = os.path.join(input_subdir_path, 'problem_info.json')
            input_code_path = os.path.join(input_subdir_path, 'optimus-code.py')
            input_params_path = os.path.join(input_subdir_path, 'parameters.json')

            output_json_path = os.path.join(output_subdir_path, 'problem_info.json')
            output_code_path = os.path.join(output_subdir_path, 'optimus-code.py')
            output_params_path = os.path.join(output_subdir_path, 'parameters.json')

            # Process the files
            print(f"Processing problem {problem}, subdir {input_subdir}...")
            process_files(input_json_path, input_code_path, output_json_path, output_code_path)

            # Copy the parameters.json file
            try:
                shutil.copy(input_params_path, output_params_path)
                print(f"Copied parameters.json to {output_params_path}")
            except FileNotFoundError:
                print(f"parameters.json not found in {input_subdir_path}")

if __name__ == "__main__":
    main()
