import gurobipy as gp
import json
import glob
import os

# Use glob to find all paths matching the pattern "X/X_c/model_updated.lp"
# For example, if you have directories like:
# 1/1_c/model_updated.lp, 2/2_c/model_updated.lp, etc.
# Adjust the pattern as needed.
model_paths = glob.glob("*/*_c/model_updated.lp")

for model_path in model_paths:
    # Determine the corresponding output JSON path by replacing "model_updated.lp" with "model_data.json"
    output_json = model_path.replace("model_updated.lp", "model_data.json")
    
    # Read the model
    model = gp.read(model_path)
    vars = model.getVars()
    constrs = model.getConstrs()

    # Extract data
    var_names = [v.VarName for v in vars]
    c = [v.Obj for v in vars]
    senses = model.getAttr("Sense", constrs)
    b = model.getAttr("RHS", constrs)

    # Construct A matrix
    A = []
    for constr in constrs:
        row_coeffs = [model.getCoeff(constr, v) for v in vars]
        A.append(row_coeffs)

    # Create readable constraints
    constraint_strings = []
    for i, constr in enumerate(constrs):
        terms = []
        for var_idx, v in enumerate(vars):
            coeff = A[i][var_idx]
            if abs(coeff) > 1e-15:
                # Determine sign prefix
                if len(terms) > 0:
                    sign_str = " + " if coeff > 0 else " - "
                else:
                    sign_str = "- " if coeff < 0 else ""

                abs_coeff = abs(coeff)
                if abs_coeff == 1:
                    # If coeff is ±1, omit the '1'
                    term_str = f"{sign_str}{v.VarName}".strip()
                else:
                    term_str = f"{sign_str}{abs_coeff:g}*{v.VarName}".strip()
                terms.append(term_str)

        lhs_str = " ".join(terms) if terms else "0"
        lhs_str = lhs_str.replace("  ", " ")

        # Convert sense
        sense = senses[i]
        if sense == '<':
            sense_str = "<="
        elif sense == '>':
            sense_str = ">="
        else:
            sense_str = "="

        rhs_val = b[i]
        if abs(rhs_val) < 1e-15:
            rhs_val = 0.0

        constraint_str = f"{lhs_str} {sense_str} {rhs_val:g}"
        constraint_strings.append(constraint_str)

    # Prepare a dictionary to store all data
    data = {
        "variables": var_names,
        "objective_coeffs": c,
        "A": A,
        "b": list(b),
        "constraints_readable": constraint_strings
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Write the dictionary to the JSON file
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Model data saved to {output_json}")
