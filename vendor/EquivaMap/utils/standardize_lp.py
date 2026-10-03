import re
import glob
import os

base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"
pattern = os.path.join(base_dir, "*", "*_c", "model.lp")

def parse_line_of_vars(line):
    line = line.strip()
    # Insert a "+" before standalone "-" signs to separate terms correctly
    line = re.sub(r'(?<!\+)\-', '+ -', line)
    parts = [p.strip() for p in line.split('+')]
    var_dict = {}
    for p in parts:
        if p == '':
            continue
        segs = p.split()
        if len(segs) == 0:
            continue
        var = segs[-1]
        if len(segs) == 1:
            # Only var is given
            if var.replace('.','',1).isdigit():
                continue
            coeff = 1.0
        else:
            coeff_str = "".join(segs[:-1])
            try:
                coeff = float(coeff_str)
            except ValueError:
                if coeff_str == '-':
                    coeff = -1.0
                elif coeff_str == '':
                    coeff = 1.0
                else:
                    # Could not parse coefficient, set to 0
                    coeff = 0.0
        var_dict[var] = coeff
    return var_dict

def parse_constraint(line):
    if ':' not in line:
        return None, None, None, None
    left_part, rest = line.split(':', 1)
    cname = left_part.strip()
    rest = rest.strip()
    
    # Match <=, >=, or =
    m = re.search(r'(<=|>=|=)', rest)
    if not m:
        return None, None, None, None
    sense = m.group(1)

    expr_part, rhs_part = rest.split(sense, 1)
    expr_part = expr_part.strip()
    rhs_part = rhs_part.strip()
    try:
        var_dict = parse_line_of_vars(expr_part)
        rhs = float(rhs_part)
    except ValueError:
        # Could not parse RHS as float
        return None, None, None, None
    return cname, var_dict, sense, rhs

files_to_process = glob.glob(pattern)

for input_file in files_to_process:
    output_file = os.path.join(os.path.dirname(input_file), "model_updated.lp")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    objective_sense = None
    objective_dict = {}
    constraints = []
    bounds = []
    generals = []
    binaries = []

    in_objective = False
    in_constraints = False
    in_bounds = False
    in_generals = False
    in_binaries = False

    header_comments = []

    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith('\\'):
            header_comments.append(line)
            continue
        if line_strip.lower().startswith('maximize'):
            objective_sense = 'Maximize'
            in_objective = True
            in_constraints = False
            in_bounds = False
            in_generals = False
            in_binaries = False
            continue
        elif line_strip.lower().startswith('minimize'):
            objective_sense = 'Minimize'
            in_objective = True
            in_constraints = False
            in_bounds = False
            in_generals = False
            in_binaries = False
            continue
        elif line_strip.lower().startswith('subject to'):
            in_objective = False
            in_constraints = True
            in_bounds = False
            in_generals = False
            in_binaries = False
            continue
        elif line_strip.lower().startswith('bounds'):
            in_objective = False
            in_constraints = False
            in_bounds = True
            in_generals = False
            in_binaries = False
            continue
        elif line_strip.lower().startswith('generals'):
            in_objective = False
            in_constraints = False
            in_bounds = False
            in_generals = True
            in_binaries = False
            continue
        elif line_strip.lower().startswith('binaries'):
            in_objective = False
            in_constraints = False
            in_bounds = False
            in_generals = False
            in_binaries = True
            continue
        elif line_strip.lower().startswith('end'):
            in_objective = False
            in_constraints = False
            in_bounds = False
            in_generals = False
            in_binaries = False
            continue

        if in_objective:
            if line_strip == '':
                continue
            line_vars = parse_line_of_vars(line)
            for v, c in line_vars.items():
                objective_dict[v] = objective_dict.get(v, 0) + c

        elif in_constraints:
            if line_strip == '':
                continue
            cname, var_dict, sense, rhs = parse_constraint(line)
            if cname is not None:
                constraints.append((cname, var_dict, sense, rhs))
            else:
                # Debug print to see what happened if a constraint doesn't parse
                print("Warning: Could not parse constraint line:", line.strip())

        elif in_bounds:
            if line_strip == '':
                continue
            bounds.append(line.rstrip('\n'))

        elif in_generals:
            if line_strip == '':
                continue
            line_vars = line_strip.split()
            for gv in line_vars:
                generals.append(gv)

        elif in_binaries:
            if line_strip == '':
                continue
            line_vars = line_strip.split()
            for bv in line_vars:
                binaries.append(bv)

    # Collect all variables
    all_vars = set(objective_dict.keys())
    for cname, var_dict, sense, rhs in constraints:
        all_vars.update(var_dict.keys())
    for b_line in bounds:
        tokens = b_line.split()
        for t in tokens:
            if t in ['<=','>=','=',':']:
                continue
            try:
                float(t)
            except:
                all_vars.add(t)
    for gv in generals:
        all_vars.add(gv)
    for bv in binaries:
        all_vars.add(bv)

    all_vars = list(all_vars)
    for v in all_vars:
        if v not in objective_dict:
            objective_dict[v] = 0.0

    originally_minimize = (objective_sense == 'Minimize')
    if originally_minimize:
        for v in objective_dict:
            objective_dict[v] = -objective_dict[v]
        objective_sense = 'Maximize'

    # Unify constraints to <=
    transformed_constraints = []
    for cname, var_dict, sense, rhs in constraints:
        if sense == '>=':
            new_var_dict = {v: -coef for v, coef in var_dict.items()}
            new_sense = '<='
            new_rhs = -rhs
        elif sense == '=':
            # Transform '=' into '<=' (Note: This is lossy)
            new_var_dict = var_dict
            new_sense = '<='
            new_rhs = rhs
        else:
            new_var_dict = var_dict
            new_sense = sense
            new_rhs = rhs
        transformed_constraints.append((cname, new_var_dict, new_sense, new_rhs))
    constraints = transformed_constraints

    all_vars.sort()
    full_constraints = []
    for cname, var_dict, sense, rhs in constraints:
        full_var_dict = {}
        for v in all_vars:
            full_var_dict[v] = var_dict.get(v, 0.0)
        full_constraints.append((cname, full_var_dict, sense, rhs))
    constraints = full_constraints

    with open(output_file, 'w') as out:
        for c in header_comments:
            out.write(c)
        if not any("LP format" in c for c in header_comments):
            out.write("\\ LP format - for model browsing. Use MPS format to capture full model detail.\n")

        out.write(f"{objective_sense}\n  ")
        obj_terms = []
        for i, v in enumerate(all_vars):
            coeff = objective_dict[v]
            obj_terms.append((coeff, v))

        final_obj_str = ""
        for i, (coeff, var) in enumerate(obj_terms):
            if i == 0:
                if coeff < 0:
                    final_obj_str += f"- {abs(coeff)} {var}"
                else:
                    final_obj_str += f"{coeff} {var}"
            else:
                if coeff < 0:
                    final_obj_str += f" - {abs(coeff)} {var}"
                else:
                    final_obj_str += f" + {coeff} {var}"
        out.write(final_obj_str.strip() + "\n")

        if constraints:
            out.write("Subject To\n")
            for cname, var_dict, sense, rhs in constraints:
                c_str = ""
                for i, v in enumerate(all_vars):
                    coeff = var_dict[v]
                    if i == 0:
                        if coeff < 0:
                            c_str += f"- {abs(coeff)} {v}"
                        else:
                            c_str += f"{coeff} {v}"
                    else:
                        if coeff < 0:
                            c_str += f" - {abs(coeff)} {v}"
                        else:
                            c_str += f" + {coeff} {v}"
                out.write(f" {cname}: {c_str} {sense} {rhs}\n")

        if bounds:
            out.write("Bounds\n")
            for b in bounds:
                out.write(b + "\n")

        if binaries:
            out.write("Binaries\n")
            out.write(" " + " ".join(binaries) + "\n")

        if generals:
            out.write("Generals\n")
            out.write(" " + " ".join(generals) + "\n")

        out.write("End\n")

    print(f"Transformed LP written to {output_file}")
