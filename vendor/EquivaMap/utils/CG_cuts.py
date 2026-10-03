import os
import json
import random
import glob
import re

base_path = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

def parse_constraint(constraint_line):
    operators = ['<=', '>=', '==', '=']
    chosen_op = None
    for op in operators:
        if op in constraint_line:
            chosen_op = op
            break
    if not chosen_op:
        return None
    left_side, right_side = constraint_line.split(chosen_op)
    left_side = left_side.strip()
    right_side = right_side.strip()
    return (left_side, chosen_op, right_side)

def parse_constraint_to_gurobipy(constraint_line):
    # Directly wrap with model.addConstr(...)
    return f"model.addConstr({constraint_line})"

def parse_linear_expr(expr_str):
    # Convert all '-' to '+-' to split easily
    expr_str = expr_str.replace('-', '+-')
    parts = expr_str.split('+')
    coeffs = {}
    for part in parts:
        part = part.strip()
        if part == '':
            continue
        # part could be something like "var", "-var", "2*var", "-2*var"
        if '*' in part:
            val, var = part.split('*', 1)
            val = val.strip().replace(' ', '')
            var = var.strip()
            if val == '' or val == '-':
                val = '-1'
            if val == '+':
                val = '1'
            coeff = float(val)
        else:
            # Could be just "var" or "-var" or a numeric-var combination without '*'
            # Try to match a pattern like: optional sign, number, then var
            p = part.replace(' ', '')
            match = re.match(r'^([+-]?[\d\.]+)([A-Za-z_]\w*)$', p)
            if match:
                val_str, var = match.groups()
                coeff = float(val_str)
            else:
                # Just var with implicit coeff 1 or -1
                if p.startswith('-'):
                    coeff = -1.0
                    var = p[1:]
                elif p.startswith('+'):
                    coeff = 1.0
                    var = p[1:]
                else:
                    coeff = 1.0
                    var = p
        if var not in coeffs:
            coeffs[var] = 0.0
        coeffs[var] += coeff
    return coeffs

def stringify_linear_expr(coeffs):
    terms = []
    for var, coeff in coeffs.items():
        if abs(coeff - round(coeff)) < 1e-9:
            coeff = int(round(coeff))
        if coeff == 0:
            continue
        if coeff == 1:
            terms.append(var)
        elif coeff == -1:
            terms.append(f"-{var}")
        else:
            # Include coefficient
            # If coeff is positive, just "coeff*var"
            # If negative, it will be handled naturally
            terms.append(f"{coeff}*{var}")
    if not terms:
        # If no terms, it's 0
        expr = "0"
    else:
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
    return expr

def combine_constraints(constraints):
    # Combine two constraints by summing their left sides and right sides
    # If different operators or no constraints, return None
    parsed = [parse_constraint(c) for c in constraints]
    if any(p is None for p in parsed):
        return None
    ops = {p[1] for p in parsed}
    if len(ops) > 1:
        # If they have different operators, can't sum meaningfully.
        return None
    operator = ops.pop()

    # Parse and sum left sides and right sides
    combined_left = {}
    combined_right = 0.0

    for p in parsed:
        left_side, op, right_side = p
        left_expr = parse_linear_expr(left_side)
        try:
            rv = float(right_side)
        except ValueError:
            return None

        # Add them directly (coefficients = 1 for each constraint)
        for var, val in left_expr.items():
            combined_left[var] = combined_left.get(var, 0.0) + val
        combined_right += rv

    left_str = stringify_linear_expr(combined_left)
    # Round right side if needed
    if abs(combined_right - round(combined_right)) < 1e-9:
        combined_right = int(round(combined_right))
    new_constraint = f"{left_str} {operator} {combined_right}"
    return new_constraint

def ensure_e_directory(instance_dir_c):
    instance_dir_e = instance_dir_c.replace('_c', '_e')
    if not os.path.exists(instance_dir_e):
        os.makedirs(instance_dir_e)
    return instance_dir_e

def insert_constraint_into_code(code_lines, gurobi_constraint_code):
    insert_index = None
    for i, line in enumerate(code_lines):
        if "# Constraints" in line:
            insert_index = i + 1
        if "model.optimize()" in line and insert_index is None:
            insert_index = i
            break
    if insert_index is None:
        insert_index = len(code_lines)

    code_lines.insert(insert_index, "# Added generated constraint\n")
    code_lines.insert(insert_index+1, gurobi_constraint_code + "\n")
    return code_lines

def update_problem_info(problem_info, description, formulation, gurobi_constraint_code):
    if "constraints" not in problem_info:
        problem_info["constraints"] = []
    new_constraint_entry = {
        "description": description,
        "formulation": formulation,
        "code": {
            "gurobipy": gurobi_constraint_code
        }
    }
    problem_info["constraints"].append(new_constraint_entry)
    return problem_info

def process_lp(instance_dir_c):
    code_file_path_c = os.path.join(instance_dir_c, "optimus-code.py")
    model_data_path_c = os.path.join(instance_dir_c, "model_data.json")
    problem_info_path_c = os.path.join(instance_dir_c, "problem_info.json")

    if not os.path.isfile(model_data_path_c):
        return
    with open(model_data_path_c, 'r') as f:
        model_data = json.load(f)
    constraints_readable = model_data.get("constraints_readable", [])
    if not constraints_readable:
        return

    # If we have at least two constraints, combine them
    if len(constraints_readable) >= 2:
        chosen_constraints = random.sample(constraints_readable, 2)
        combined = combine_constraints(chosen_constraints)
        if combined is not None:
            chosen_constraint = combined
            description = "Combined two constraints with all coefficients = 1"
        else:
            # If cannot combine for some reason, fallback to single constraint
            chosen_constraint = random.choice(constraints_readable)
            description = "Added a random single constraint (could not combine)"
    else:
        # Only one constraint available
        chosen_constraint = constraints_readable[0]
        description = "Added the only available constraint"

    gurobi_constraint_code = parse_constraint_to_gurobipy(chosen_constraint)

    with open(code_file_path_c, 'r') as f:
        code_lines = f.readlines()

    code_lines = insert_constraint_into_code(code_lines, gurobi_constraint_code)

    instance_dir_e = ensure_e_directory(instance_dir_c)
    code_file_path_e = os.path.join(instance_dir_e, "optimus-code.py")
    problem_info_path_e = os.path.join(instance_dir_e, "problem_info.json")

    with open(code_file_path_e, 'w') as f:
        f.writelines(code_lines)

    if os.path.isfile(problem_info_path_c):
        with open(problem_info_path_c, 'r') as f:
            problem_info = json.load(f)
    else:
        problem_info = {}

    problem_info = update_problem_info(problem_info,
                                       description,
                                       chosen_constraint,
                                       gurobi_constraint_code)
    with open(problem_info_path_e, 'w') as f:
        json.dump(problem_info, f, indent=4)

def process_mip(instance_dir_c):
    code_file_path_c = os.path.join(instance_dir_c, "optimus-code.py")
    problem_info_path_c = os.path.join(instance_dir_c, "problem_info.json")
    log_file_path_c = os.path.join(instance_dir_c, "log.txt")
    model_data_path_c = os.path.join(instance_dir_c, "model_data.json")

    new_equation_line = None
    if os.path.isfile(log_file_path_c):
        with open(log_file_path_c, 'r') as f:
            for line in f:
                if line.startswith("New equation:"):
                    new_equation_line = line[len("New equation:"):].strip()
                    break

    if new_equation_line:
        # Before using this equation, we must substitute variable names
        if not os.path.isfile(model_data_path_c):
            # If no model_data, can't substitute, fallback to LP logic
            process_lp(instance_dir_c)
            return

        with open(model_data_path_c, 'r') as f:
            model_data = json.load(f)
        variables = model_data.get("variables", [])

        def substitute_var(match):
            var_index = int(match.group(1)) - 1  # x1 -> index 0
            if var_index < 0 or var_index >= len(variables):
                return match.group(0)
            return variables[var_index]

        # Replace all x<number> in new_equation_line
        new_equation_line_substituted = re.sub(r'x(\d+)', substitute_var, new_equation_line)

        gurobi_constraint_code = parse_constraint_to_gurobipy(new_equation_line_substituted)

        with open(code_file_path_c, 'r') as f:
            code_lines = f.readlines()
        code_lines = insert_constraint_into_code(code_lines, gurobi_constraint_code)

        instance_dir_e = ensure_e_directory(instance_dir_c)
        code_file_path_e = os.path.join(instance_dir_e, "optimus-code.py")
        problem_info_path_e = os.path.join(instance_dir_e, "problem_info.json")

        with open(code_file_path_e, 'w') as f:
            f.writelines(code_lines)

        if os.path.isfile(problem_info_path_c):
            with open(problem_info_path_c, 'r') as f:
                problem_info = json.load(f)
        else:
            problem_info = {}

        problem_info = update_problem_info(problem_info,
                                           "New equation extracted from MIP log with variable substitution",
                                           new_equation_line_substituted,
                                           gurobi_constraint_code)
        with open(problem_info_path_e, 'w') as f:
            json.dump(problem_info, f, indent=4)
    else:
        # No new equation found, do LP-like process (combine two constraints if possible)
        process_lp(instance_dir_c)

# Main script
json_files = glob.glob(os.path.join(base_path, "*", "*_c", "model_data.json"))

for json_file in json_files:
    dir_path = os.path.dirname(json_file)  # e.g. /Users/.../<instance>/<instance>_c
    code_file_path_c = os.path.join(dir_path, "optimus-code.py")
    if not os.path.isfile(code_file_path_c):
        continue
    with open(code_file_path_c, 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        continue
    problem_type_line = lines[2].strip()
    if "Problem type:" not in problem_type_line:
        continue
    problem_type = problem_type_line.split("Problem type:")[-1].strip()

    if problem_type == "LP":
        process_lp(dir_path)
    elif problem_type == "MIP":
        process_mip(dir_path)

print("Done updating problems, updated files stored in directories ending with _e.")
