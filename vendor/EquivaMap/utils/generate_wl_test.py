import os
import re
import math
import networkx as nx

BASE_DIR = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

def parse_terms(expression):
    """
    Parse a linear expression into a dict {var: coeff}.
    """
    coeff_dict = {}
    term_pattern = re.compile(r"([+-]?)\s*(\d*(?:\.\d+)?)\s*([A-Za-z]\w*(?:\[\d+\])?)")
    matches = term_pattern.findall(expression)
    for sign, number_str, var_name in matches:
        s = -1.0 if sign == '-' else 1.0
        coeff = float(number_str) if number_str else 1.0
        coeff *= s
        coeff_dict[var_name] = coeff_dict.get(var_name, 0.0) + coeff
    return coeff_dict

def parse_lp_file(lp_path):
    with open(lp_path, 'r') as f:
        content = f.read()

    sense_obj = re.search(r"(Maximize|Minimize)", content, re.IGNORECASE)
    obj_sense = 'min' if sense_obj and sense_obj.group(1).lower() == 'minimize' else 'max'

    obj_section = re.search(r"(Maximize|Minimize)\s+(.*?)Subject To", content, re.DOTALL|re.IGNORECASE)
    obj_coeffs = {}
    if obj_section:
        obj_text = obj_section.group(2).strip()
        obj_lines = obj_text.split('\n')
        obj_line = " ".join(l.strip() for l in obj_lines)
        obj_coeffs = parse_terms(obj_line)

    constraints_section = re.search(r"Subject To\s+(.*?)(Bounds|Binary|Binaries|General|Generals|End)", content, re.DOTALL|re.IGNORECASE)
    constraints = []
    if constraints_section:
        constr_text = constraints_section.group(1)
        lines = [l.strip() for l in constr_text.split('\n') if l.strip()!='']
        for line in lines:
            if ':' in line:
                cname, cexpr = line.split(':', 1)
                cname = cname.strip()
                cexpr = cexpr.strip()
                m = re.search(r"(<=|>=|=)\s*([+-]?\d+(\.\d+)?)", cexpr)
                if m:
                    csense_str = m.group(1)
                    rhs = float(m.group(2))
                else:
                    continue

                if csense_str == '<=':
                    sense_c = 0
                elif csense_str == '=':
                    sense_c = 1
                else:
                    sense_c = 2

                left_side = re.split(r"(<=|>=|=)", cexpr)[0].strip()
                coeff_dict = parse_terms(left_side)
                constraints.append((cname, coeff_dict, sense_c, rhs))

    var_bounds = {}
    bounds_section = re.search(r"Bounds\s+(.*?)(Binary|Binaries|General|Generals|End)", content, re.DOTALL|re.IGNORECASE)
    if bounds_section:
        btext = bounds_section.group(1)
        blines = [l.strip() for l in btext.split('\n') if l.strip()!='']
        for line in blines:
            parts = line.split()
            if len(parts)==3:
                if parts[1]=='<=':
                    if parts[0].replace('.','',1).isdigit():
                        # num <= var
                        lb = float(parts[0])
                        var = parts[2]
                        var_bounds[var] = (lb, math.inf)
                    else:
                        # var <= num
                        var = parts[0]
                        ub = float(parts[2])
                        lb, _ = var_bounds.get(var,(0,math.inf))
                        var_bounds[var] = (lb, ub)
                elif parts[1]=='>=':
                    # var >= num
                    if parts[0].isalpha():
                        var = parts[0]
                        lb = float(parts[2])
                        _, ub = var_bounds.get(var,(0,math.inf))
                        var_bounds[var] = (lb, ub)
            elif len(parts)==5:
                # "0 <= x <= 10"
                if parts[1]=='<=' and parts[3]=='<=':
                    lb_val = float(parts[0])
                    var = parts[2]
                    ub_val = float(parts[4])
                    var_bounds[var] = (lb_val, ub_val)

    binaries_section = re.search(r"(Binary|Binaries)\s+(.*?)(General|Generals|End)", content, re.DOTALL|re.IGNORECASE)
    binaries = set()
    if binaries_section:
        bin_text = binaries_section.group(2)
        bin_vars = re.findall(r"(\w+(\[\d+\])?)", bin_text)
        for bv_tuple in bin_vars:
            bv = bv_tuple[0]
            binaries.add(bv)

    generals_section = re.search(r"(General|Generals)\s+(.*?)(Binary|Binaries|End)", content, re.DOTALL|re.IGNORECASE)
    generals = set()
    if generals_section:
        gen_text = generals_section.group(2)
        gen_vars = re.findall(r"(\w+(\[\d+\])?)", gen_text)
        for gv_tuple in gen_vars:
            gv = gv_tuple[0]
            generals.add(gv)

    vars_all = set(obj_coeffs.keys())
    for _, coeff_dict, _, _ in constraints:
        vars_all.update(coeff_dict.keys())

    # Default bounds
    for v in vars_all:
        if v not in var_bounds:
            var_bounds[v] = (0.0, math.inf)

    # Adjust bounds for binaries
    for v in binaries:
        var_bounds[v] = (0.0, 1.0)

    # Determine integrality
    var_types = {}
    for v in vars_all:
        if v in binaries:
            var_types[v] = 1
        else:
            var_types[v] = 0

    # If var not in obj_coeff, cost=0
    for v in vars_all:
        if v not in obj_coeffs:
            obj_coeffs[v] = 0.0

    return obj_sense, obj_coeffs, constraints, var_bounds, var_types

def update_var_types_from_optimus_code(optimus_path, var_types):
    """
    Parse the optimus-code.py file to find the addVar statements and update var_types accordingly.
    If vtype is BINARY or INTEGER, set is_integer = 1, else 0.
    If vtype is not specified, assume continuous (is_integer = 0).
    """
    if not os.path.exists(optimus_path):
        return var_types

    # Regex to capture lines with addVar, extracting vtype and name
    # Pattern tries to capture something like:
    # varname = model.addVar(... vtype=GRB.INTEGER, ... name="w")
    line_pattern = re.compile(r"(\w+)\s*=\s*model\.addVar\s*\((.*?)\)")

    # We'll look inside the arguments for vtype and name
    vtype_pattern = re.compile(r"vtype\s*=\s*GRB\.([A-Z]+)")
    name_pattern = re.compile(r'name\s*=\s*"([^"]+)"')

    with open(optimus_path, 'r') as f:
        for line in f:
            line_match = line_pattern.search(line)
            if line_match:
                var_python_name = line_match.group(1)  # the Python variable name on LHS
                args_str = line_match.group(2)
                vtype_match = vtype_pattern.search(args_str)
                name_match = name_pattern.search(args_str)

                var_name = None
                if name_match:
                    var_name = name_match.group(1)
                else:
                    # If no name attribute, assume var_name = var_python_name
                    var_name = var_python_name

                vtype = "CONTINUOUS" # default
                if vtype_match:
                    vtype = vtype_match.group(1)

                # Update var_types based on vtype
                # BINARY or INTEGER means is_integer = 1
                # Else (CONTINUOUS) means is_integer = 0
                if var_name in var_types:
                    if vtype == "BINARY" or vtype == "INTEGER":
                        var_types[var_name] = 1
                    else:
                        var_types[var_name] = 0

    return var_types

def generate_wltest_py(lp_path, obj_sense, obj_coeffs, constraints, var_bounds, var_types):
    dir_name = os.path.dirname(lp_path)
    wltest_path = os.path.join(dir_name, "wltest.py")

    code_lines = []
    code_lines.append("import networkx as nx")
    code_lines.append("")
    code_lines.append("# Constructing MILP graph from model.lp")
    code_lines.append("G = nx.Graph()")

    # Add constraint nodes
    for (cname, coeff_dict, sense_c, rhs) in constraints:
        code_lines.append(f"G.add_node('{cname}', rhs={rhs}, sense={sense_c})")

    # Add variable nodes
    for v in sorted(obj_coeffs.keys()):
        cost = obj_coeffs[v]
        lb, ub = var_bounds[v]
        ub_str = "float('inf')" if math.isinf(ub) else str(ub)
        is_int = var_types[v]
        code_lines.append(f"G.add_node('{v}', cost={cost}, lb={lb}, ub={ub_str}, is_integer={is_int})")

    # Add edges for all variables in each constraint
    for (cname, coeff_dict, sense_c, rhs) in constraints:
        for var, coeff in coeff_dict.items():
            code_lines.append(f"G.add_edge('{cname}', '{var}', weight={coeff})")

    code_lines.append("")
    code_lines.append("# Prepare labels for WL")
    code_lines.append("for n, data in G.nodes(data=True):")
    code_lines.append("    if 'sense' in data and 'rhs' in data:")
    code_lines.append("        # Constraint node")
    code_lines.append("        G.nodes[n]['label'] = f\"{data['rhs']}_{data['sense']}\"")
    code_lines.append("    else:")
    code_lines.append("        # Variable node")
    code_lines.append("        c = data['cost']")
    code_lines.append("        lb = data['lb']")
    code_lines.append("        ub = data['ub']")
    code_lines.append("        isint = data['is_integer']")
    code_lines.append("        G.nodes[n]['label'] = f\"{c}_{lb}_{ub}_{isint}\"")
    code_lines.append("")
    code_lines.append("for u,v,data in G.edges(data=True):")
    code_lines.append("    G[u][v]['label'] = str(data['weight'])")
    code_lines.append("")
    code_lines.append("wl_hash = nx.weisfeiler_lehman_graph_hash(G, node_attr='label', edge_attr='label', iterations=2)")
    code_lines.append("print('WL Hash:', wl_hash)")

    with open(wltest_path, 'w') as f:
        f.write("\n".join(code_lines))

    print(f"wltest.py generated at: {wltest_path}")

def main():
    for root, dirs, files in os.walk(BASE_DIR):
        if not root.endswith("_i"):
            continue
        for file in files:
            if file == "model.lp":
                lp_path = os.path.join(root, file)
                optimus_path = os.path.join(root, "optimus-code.py") 
                print("Processing:", lp_path)
                obj_sense, obj_coeffs, constraints, var_bounds, var_types = parse_lp_file(lp_path)

                # Update var_types based on optimus-code.py
                var_types = update_var_types_from_optimus_code(optimus_path, var_types)

                generate_wltest_py(lp_path, obj_sense, obj_coeffs, constraints, var_bounds, var_types)


if __name__ == "__main__":
    main()
