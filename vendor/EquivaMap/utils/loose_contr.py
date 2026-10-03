import os
import shutil
import json
import random

def collect_constraint_indices(code_lines):
    """
    Scans code_lines for lines containing either 'model.addConstr(' or 'model.addConstrs('.
    Returns a list of line indices for each match.
    """
    indices = []
    for i, line in enumerate(code_lines):
        # Normalize line by removing whitespace
        stripped = line.strip().replace(' ', '')
        # Check for both addConstr( and addConstrs(
        if 'model.addConstr(' in stripped or 'model.addConstrs(' in stripped:
            indices.append(i)
    return indices

def remove_all_but_one_constraint(code_lines):
    """
    Finds lines with 'model.addConstr(' or 'model.addConstrs('.
    If >= 2 constraints, keep exactly one (random).
    If exactly 1, remove it (so 0 remain).
    If 0, do nothing.

    Returns:
      (new_code_lines, kept_index)
        new_code_lines: The updated lines of code
        kept_index: The original line index of the constraint we kept, or None if none were kept
    """
    indices = collect_constraint_indices(code_lines)
    n_constr = len(indices)

    if n_constr == 0:
        # No constraints at all
        return code_lines, None

    if n_constr == 1:
        # Exactly 1 constraint => remove it => keep 0 constraints
        new_lines = []
        for i, line in enumerate(code_lines):
            if i not in indices:
                new_lines.append(line)
        return new_lines, None

    # If >= 2 constraints, pick one randomly to keep
    keep_idx = random.choice(indices)
    new_lines = []
    for i, line in enumerate(code_lines):
        if i == keep_idx:
            new_lines.append(line)
        elif i in indices:
            # skip any other constraint lines
            continue
        else:
            new_lines.append(line)

    return new_lines, keep_idx

def update_problem_info(info_data, kept_index, original_code_lines):
    """
    Adjust info_data['constraints'] so that it keeps only the constraint that matches
    the line at kept_index (if any). If kept_index is None, remove all constraints.

    We match constraints by comparing the 'code.gurobipy' string with the corresponding line in code.
    If no match found, we remove them all.
    """
    if "constraints" not in info_data or not isinstance(info_data["constraints"], list):
        # No constraints to modify
        return info_data

    # If we didn't keep any constraint line
    if kept_index is None:
        info_data["constraints"] = []
        return info_data

    # Otherwise, try to match the line we kept
    kept_line = original_code_lines[kept_index].strip().replace(' ', '')

    chosen_index = -1
    for i, constr_block in enumerate(info_data["constraints"]):
        c_code = constr_block.get("code", {}).get("gurobipy", "")
        if c_code:
            # remove spaces for matching
            c_code_stripped = c_code.strip().replace(' ', '')
            # If the constraint code appears in the line we kept, assume it's the match
            if c_code_stripped in kept_line:
                chosen_index = i
                break

    if chosen_index == -1:
        # No match found => remove them all
        info_data["constraints"] = []
    else:
        # Keep only the matched constraint
        info_data["constraints"] = [info_data["constraints"][chosen_index]]

    return info_data

def main():
    base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

    for root, dirs, files in os.walk(base_dir):
        if root.endswith("_c"):
            required = {"optimus-code.py", "problem_info.json", "parameters.json"}
            if required.issubset(files):
                parent_dir = os.path.dirname(root)
                c_basename = os.path.basename(root)  # e.g. "243_c"
                # Build _l directory
                l_basename = c_basename.replace("_c", "_l")
                l_dir = os.path.join(parent_dir, l_basename)
                os.makedirs(l_dir, exist_ok=True)

                # Paths in _c
                code_c_path = os.path.join(root, "optimus-code.py")
                info_c_path = os.path.join(root, "problem_info.json")
                param_c_path = os.path.join(root, "parameters.json")

                # Paths in _l
                code_l_path = os.path.join(l_dir, "optimus-code.py")
                info_l_path = os.path.join(l_dir, "problem_info.json")
                param_l_path = os.path.join(l_dir, "parameters.json")

                # 1) Read the code
                with open(code_c_path, "r", encoding="utf-8") as fc:
                    code_lines = fc.readlines()

                # 2) Remove constraints (keep exactly 1 if >=2, else 0)
                new_code_lines, kept_index = remove_all_but_one_constraint(code_lines)

                # 3) Write updated code to _l
                with open(code_l_path, "w", encoding="utf-8") as fc_out:
                    fc_out.writelines(new_code_lines)

                # 4) Copy parameters.json as is
                shutil.copy(param_c_path, param_l_path)

                # 5) Modify the info file
                with open(info_c_path, "r", encoding="utf-8") as fi_in:
                    info_data = json.load(fi_in)

                info_data = update_problem_info(info_data, kept_index, code_lines)

                # 6) Write updated info to _l
                with open(info_l_path, "w", encoding="utf-8") as fi_out:
                    json.dump(info_data, fi_out, indent=4)

                print(f"Created {l_dir}, originally had constraint lines at {collect_constraint_indices(code_lines)}; kept index: {kept_index}.")

if __name__ == "__main__":
    main()
