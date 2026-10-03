import json
import math
import os
import re
import shutil

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def replace_in_formulation(text, latex_digit_sums):
    # Word boundary may still be acceptable for the formulation if it uses variable names clearly,
    # but you can apply a similar token approach if needed.
    for var_name, replacement in latex_digit_sums.items():
        pattern = r'\b' + re.escape(var_name) + r'\b'
        text = re.sub(pattern, replacement, text)
    return text

def split_line_into_tokens(line):
    # This regex splits the line into a list of tokens, separating on non-alphanumeric characters 
    # but keeping them as separate tokens (by using a capturing group).
    # Example: "model.setObjective(v_0+N*K)" -> ["model", ".", "setObjective", "(", "v_0", "+", "N", "*", "K", ")", ...]
    tokens = re.findall(r'[A-Za-z0-9_]+|\S', line)
    return tokens

def replace_tokens_in_line(line, digit_sums):
    tokens = split_line_into_tokens(line)
    # For each token, if it's an exact var_name, replace it with digit_sums[var_name]
    replaced_tokens = []
    for t in tokens:
        if t in digit_sums:
            replaced_tokens.append(digit_sums[t])
        else:
            replaced_tokens.append(t)
    return "".join(replaced_tokens)

def replace_in_code_line(line, digit_sums):
    # Only replace tokens that exactly match variable names
    return replace_tokens_in_line(line, digit_sums)

def is_var_line(line):
    return "model.addVar" in line

def is_constr_line(line):
    return "model.addConstr" in line

def is_objective_line(line):
    return "model.setObjective" in line

def is_solution_extraction_line(line):
    return "variables[" in line and ".x" in line

def perform_digit_decomposition(problem_info_file, solution_file, original_code_file, digit_problem_info_file, digit_code_file):
    # Load the original files
    problem_info = load_json(problem_info_file)
    solution = load_json(solution_file)

    variables_solution = solution.get("variables", {})

    # 1. Identify which variables need decomposition
    variables_to_decompose = {}
    for var_name, var_val in variables_solution.items():
        if isinstance(var_val, (int, float)):
            # Check integrality and non-negativity
            if abs(var_val - round(var_val)) < 1e-9 and var_val >= 0:
                int_val = int(round(var_val))
                digits_str = str(int_val)
                k = len(digits_str)
                variables_to_decompose[var_name] = {
                    "value": int_val,
                    "digits": [int(d) for d in digits_str],
                    "k": k
                }

    if not variables_to_decompose:
        # No variables found that require digit decomposition
        # Just copy files
        os.makedirs(os.path.dirname(digit_problem_info_file), exist_ok=True)
        shutil.copyfile(problem_info_file, digit_problem_info_file)
        os.makedirs(os.path.dirname(digit_code_file), exist_ok=True)
        shutil.copyfile(original_code_file, digit_code_file)
        return False  # Indicates no decomposition was done

    # 2. Modify problem_info.json
    problem_info_digit = problem_info.copy()
    new_variables_info = problem_info["variables"].copy()
    for var_name in variables_to_decompose.keys():
        # Safely retrieve the original description, if it exists
        if var_name in problem_info["variables"]:
            orig_description = problem_info["variables"][var_name].get("description", f"variable {var_name}")
        else:
            # If the variable wasn't in the problem_info, use a fallback description
            orig_description = f"variable {var_name}"

        # Remove the original variable entry if it exists
        if var_name in new_variables_info:
            del new_variables_info[var_name]
        
        k = variables_to_decompose[var_name]["k"]
        for i in range(k):
            new_var_name = f"{var_name}_{i}"
            new_variables_info[new_var_name] = {
                "description": f"Digit {i} of the {orig_description}",
                "type": "integer",
                "shape": []
            }

    problem_info_digit["variables"] = new_variables_info

    # Create digit sums for each variable
    digit_sums = {}
    for var_name, info in variables_to_decompose.items():
        k = info["k"]
        # Construct the digit sum expression
        digit_sum_expr = " + ".join([f"{var_name}_{i}*10**{i}" for i in range(k)])
        digit_sums[var_name] = f"({digit_sum_expr})"

    latex_digit_sums = {}
    for var_name, info in variables_to_decompose.items():
        k = info["k"]
        latex_expr_terms = [f"{var_name}_{i}*10^{i}" for i in range(k)]
        latex_digit_sums[var_name] = f"({ ' + '.join(latex_expr_terms) })"

    # Update constraints and objective in problem_info_digit
    if "constraints" in problem_info_digit:
        for c in problem_info_digit["constraints"]:
            if "formulation" in c:
                c["formulation"] = replace_in_formulation(c["formulation"], latex_digit_sums)
            if "code" in c and "gurobipy" in c["code"]:
                # For code, do a safer line-by-line token replacement
                c_code_lines = c["code"]["gurobipy"].split("\n")
                new_c_code_lines = []
                for cline in c_code_lines:
                    new_c_code_lines.append(replace_in_code_line(cline, digit_sums))
                c["code"]["gurobipy"] = "\n".join(new_c_code_lines)

    if "objective" in problem_info_digit:
        if "formulation" in problem_info_digit["objective"]:
            problem_info_digit["objective"]["formulation"] = replace_in_formulation(problem_info_digit["objective"]["formulation"], latex_digit_sums)
        if "code" in problem_info_digit["objective"] and "gurobipy" in problem_info_digit["objective"]["code"]:
            o_code_lines = problem_info_digit["objective"]["code"]["gurobipy"].split("\n")
            new_o_code_lines = []
            for oline in o_code_lines:
                new_o_code_lines.append(replace_in_code_line(oline, digit_sums))
            problem_info_digit["objective"]["code"]["gurobipy"] = "\n".join(new_o_code_lines)

    save_json(digit_problem_info_file, problem_info_digit)

    # 3. Modify the code file (optimus-code.py)
    with open(original_code_file, "r") as f:
        code_lines = f.readlines()

    new_code_lines = []
    digit_definitions = {}
    for var_name, info in variables_to_decompose.items():
        k = info["k"]
        # Create code snippet for digit variables
        digit_vars_code = []
        for i in range(k):
            digit_vars_code.append(
                f"{var_name}_{i} = model.addVar(vtype=GRB.INTEGER, lb=0, ub=9, name=\"{var_name}_{i}\")"
            )
        digit_definitions[var_name] = "\n".join(digit_vars_code)

    for line in code_lines:
        # Handle variable declaration lines
        if is_var_line(line):
            found_var = None
            # Instead of a simple substring check, we can also token-check:
            tokens = split_line_into_tokens(line)
            for var_name in variables_to_decompose:
                if var_name in tokens and "addVar" in tokens:
                    found_var = var_name
                    break
            if found_var:
                new_line = digit_definitions[found_var] + "\n"
            else:
                new_line = line
        elif is_constr_line(line) or is_objective_line(line):
            # Replace occurrences of original variables using token-based approach
            new_line = replace_in_code_line(line, digit_sums)
        elif is_solution_extraction_line(line):
            # For solution extraction lines, replace references to original var with digit vars
            # This is a special case: we know how these lines are structured.
            new_line = line
            for var_name, info in variables_to_decompose.items():
                if f"variables['{var_name}']" in new_line and f"{var_name}.x" in new_line:
                    k = variables_to_decompose[var_name]["k"]
                    # We replace the entire line with multiple lines
                    new_line = ""
                    for i in range(k):
                        new_line += f"variables['{var_name}_{i}'] = {var_name}_{i}.x\n"
                    break
        else:
            new_line = line

        new_code_lines.append(new_line)

    os.makedirs(os.path.dirname(digit_code_file), exist_ok=True)
    with open(digit_code_file, "w") as f:
        f.writelines(new_code_lines)

    return True  # Indicates decomposition was performed

#### Main part of the script ####

base_directory = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"  # Adjust as needed

decomposed_files = []

for root, dirs, files in os.walk(base_directory):
    if root.endswith("_c"):
        problem_info_file = os.path.join(root, "problem_info.json")
        solution_file = os.path.join(root, "solution.json")
        original_code_file = os.path.join(root, "optimus-code.py")

        if (os.path.isfile(problem_info_file) and 
            os.path.isfile(solution_file) and 
            os.path.isfile(original_code_file)):

            output_dir = root[:-2] + "_d"
            digit_problem_info_file = os.path.join(output_dir, "problem_info.json")
            digit_code_file = os.path.join(output_dir, "optimus-code.py")

            did_decompose = perform_digit_decomposition(
                problem_info_file,
                solution_file,
                original_code_file,
                digit_problem_info_file,
                digit_code_file
            )

            if did_decompose:
                decomposed_files.append(root)

if decomposed_files:
    print("Directories that required digit decomposition:")
    for d in decomposed_files:
        print(f" - {d}")
    print(f"Total: {len(decomposed_files)} directories.")
else:
    print("No directories required digit decomposition.")
