import sys
import subprocess
import gurobipy as gp
from gurobipy import GRB
import os

def fractional_cut_callback(model, where):
    # This is a generic callback that checks if we are at a MIP node and can add a cut.
    if where == GRB.Callback.MIPNODE:
        # Get candidate solution values
        node_solution = model.cbGetNodeRel(model.getVars())
        # Identify fractional variables
        fractional_vars = []
        for i, v in enumerate(model.getVars()):
            val = node_solution[i]
            if abs(val - round(val)) > 1e-5:
                fractional_vars.append((v, val))
        
        if fractional_vars:
            # Example: add a trivial cut that tries to 'round up' 
            # (This is just a dummy example. In practice, you'd derive a meaningful cut.)
            # For instance, sum of these fractional variables >= 1
            # This is not necessarily a valid or useful cut; it's just illustrative.
            expr = gp.LinExpr()
            for (var, val) in fractional_vars:
                expr.add(var, 1.0)
            # Add the cut: sum of fractional_vars >= 1
            model.cbCut(expr >= 1.0)

            # Store the constraint for later output.
            # We'll store it in the model's _added_cuts attribute
            # For a real scenario, you'd store something more meaningful.
            if not hasattr(model, '_added_cuts'):
                model._added_cuts = []
            model._added_cuts.append((fractional_vars, "sum fractional_vars >= 1"))

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_with_cut.py <input_gurobi_script> <output_cut_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Example:
    # input_file = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/1/1_a/optimus-code.py"
    # output_file = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/1/1_e/cut.py"
    
    # Run the input gurobi script using subprocess. We assume it creates a model file named "model.lp"
    # in the same directory, or somewhere known.
    # You might need to modify 'optimus-code.py' so that it writes out a model file or a scenario
    # that can be loaded.
    
    run_dir = os.path.dirname(input_file)
    model_lp_path = os.path.join(run_dir, "model.lp")

    # Run the input script
    subprocess.run(["python", input_file], check=True)

    # Check if model.lp exists
    if not os.path.isfile(model_lp_path):
        print("No model.lp file found. Ensure that optimus-code.py generates a model.lp file.")
        sys.exit(1)

    # Load the model from the LP file
    model = gp.read(model_lp_path)

    # Set up the model parameters if needed
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output if desired

    # Optimize with our callback
    model.optimize(fractional_cut_callback)

    # After optimization, if we added cuts, we can output them
    if hasattr(model, '_added_cuts') and model._added_cuts:
        # Write the last added cut (or all added cuts) to the output file as Python code.
        # For demonstration, we'll just write a Python snippet that defines a constraint.
        # In a real scenario, you might want a more systematic format.
        
        with open(output_file, "w") as f:
            f.write("# This file contains a cutting-plane constraint discovered by the callback.\n\n")
            # Just take the last constraint added:
            fractional_vars, constraint_desc = model._added_cuts[-1]
            var_names = [v.VarName for (v, val) in fractional_vars]
            
            # Write a Python snippet that defines a constraint in a future model
            # For example: sum of those var_names >= 1
            f.write("def add_cut(model):\n")
            f.write("    # Automatically generated cut\n")
            f.write("    expr = 0\n")
            for name in var_names:
                f.write(f"    expr += model.getVarByName('{name}')\n")
            f.write("    model.addConstr(expr >= 1, name='fractional_cut')\n")
            f.write(f"    # Description: {constraint_desc}\n")
    else:
        # No cuts added
        with open(output_file, "w") as f:
            f.write("# No cutting-plane constraints were added.\n")

if __name__ == "__main__":
    main()
