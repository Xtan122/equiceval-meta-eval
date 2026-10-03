import os
import shutil
import json
import re

def transform_objective_line(line, new_value):
    """
    Detects and replaces the first argument in a line containing:
        model.setObjective(<ANY EXPRESSION>, GRB.MAXIMIZE)
    or
        model.setObjective(<ANY EXPRESSION>, GRB.MINIMIZE)
    
    <ANY EXPRESSION> may contain nested parentheses (e.g. quicksum(...)).

    We replace it with:
        model.setObjective(new_value, GRB.<direction>)
    
    Returns (new_line, direction):
      - new_line: the updated line
      - direction: 'MAXIMIZE' or 'MINIMIZE' if found; otherwise None.
    """
    if 'model.setObjective(' not in line:
        return line, None
    
    # Check direction by presence of GRB.MAXIMIZE or GRB.MINIMIZE
    if 'GRB.MAXIMIZE' in line:
        direction = 'MAXIMIZE'
    elif 'GRB.MINIMIZE' in line:
        direction = 'MINIMIZE'
    else:
        return line, None

    # Regex pattern to capture:
    # (1) 'model.setObjective(...)' up to the first argument 
    # (2) the entire first argument with a lazy match (.+?) 
    # (3) the comma plus 'GRB.<direction>' plus closing parenthesis
    #
    # Example line: 
    #   model.setObjective(quicksum(L[i] * z[i] for i in range(O)), GRB.MAXIMIZE)
    pattern = r'(model\.setObjective\s*\()(.+?)(,\s*GRB\.(MAXIMIZE|MINIMIZE)\s*\))'
    match = re.search(pattern, line)
    if match:
        prefix = match.group(1)  # e.g. "model.setObjective("
        # group(2) is the entire expression 'quicksum(L[i] * z[i] for i in range(O))'
        # group(3) is the comma and 'GRB.MAXIMIZE)' or 'GRB.MINIMIZE)'
        
        # Build a new line with new_value in place of the old expression.
        # We preserve the direction so if we found 'MAXIMIZE' above, keep it.
        new_line = prefix + str(new_value) + match.group(3)
        return new_line, direction
    
    return line, None

def modify_code_file(code_j_path, code_k_path, new_objective_value):
    """
    Reads code from code_j_path, replaces the objective argument with new_objective_value
    (keeping the direction: MAXIMIZE or MINIMIZE), and writes modified code to code_k_path.
    Returns the direction found: 'MAXIMIZE'/'MINIMIZE' or None if no objective line found.
    """
    direction_found = None

    with open(code_j_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    new_lines = []
    for line in lines:
        updated_line, dir_this_line = transform_objective_line(line, new_objective_value)
        new_lines.append(updated_line)
        # If transform_objective_line found a direction, keep track
        if dir_this_line is not None:
            direction_found = dir_this_line

    with open(code_k_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(new_lines)
    
    return direction_found

def modify_problem_info(info_j_path, info_k_path, obj_val, direction):
    """
    Loads problem_info.json from info_j_path,
    updates the "objective" block with the numeric obj_val and direction,
    then writes to info_k_path.
    """
    # If the code didn't have an objective line, we can default to MAXIMIZE or skip
    if direction is None:
        direction = "MAXIMIZE"
    
    with open(info_j_path, 'r', encoding='utf-8') as f_in:
        info_data = json.load(f_in)
    
    # Build a new objective block
    if direction == "MAXIMIZE":
        formulation_str = f"Max({obj_val})"
    else:
        formulation_str = f"Min({obj_val})"
    
    info_data["objective"] = {
        "description": "The objective has been replaced by the solution value.",
        "formulation": formulation_str,
        "code": {
            "gurobipy": f"model.setObjective({obj_val}, GRB.{direction})"
        }
    }

    with open(info_k_path, 'w', encoding='utf-8') as f_out:
        json.dump(info_data, f_out, indent=4)

def main():
    base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

    # Walk through the entire base_dir
    for root, dirs, files in os.walk(base_dir):
        # If a folder ends with "_j" and contains the three required files
        if root.endswith("_j"):
            required = {"optimus-code.py", "problem_info.json", "parameters.json"}
            if required.issubset(files):
                # Build the corresponding _k folder
                parent_dir = os.path.dirname(root)
                j_basename = os.path.basename(root)  # e.g. "249_j"
                k_basename = j_basename.replace("_j", "_k")
                k_dir = os.path.join(parent_dir, k_basename)
                
                if not os.path.exists(k_dir):
                    os.makedirs(k_dir, exist_ok=True)
                
                # Also figure out the matching _c folder
                c_basename = j_basename.replace("_j", "_c")
                c_dir = os.path.join(parent_dir, c_basename)
                
                # We read the solution.json from the _c folder to get the objective value
                solution_path = os.path.join(c_dir, "solution.json")
                obj_val = 0.0  # Default if not found
                if os.path.isfile(solution_path):
                    with open(solution_path, 'r', encoding='utf-8') as f_sol:
                        sol_data = json.load(f_sol)
                        if "objective" in sol_data:
                            obj_val = sol_data["objective"]
                
                # Paths in the _j folder
                code_j_path = os.path.join(root, "optimus-code.py")
                info_j_path = os.path.join(root, "problem_info.json")
                param_j_path = os.path.join(root, "parameters.json")

                # Paths in the _k folder
                code_k_path = os.path.join(k_dir, "optimus-code.py")
                info_k_path = os.path.join(k_dir, "problem_info.json")
                param_k_path = os.path.join(k_dir, "parameters.json")

                # 1) Modify code file to use the new numeric objective
                direction_found = modify_code_file(code_j_path, code_k_path, obj_val)

                # 2) Copy parameters.json unchanged
                shutil.copy(param_j_path, param_k_path)

                # 3) Modify problem_info.json
                modify_problem_info(info_j_path, info_k_path, obj_val, direction_found)
                
                print(f"Created {k_dir}, replaced objective with {obj_val} (direction={direction_found}).")

if __name__ == "__main__":
    main()
