import os
import re

# Adjust to your own root directory
BASE_DIR = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# Regex to match and capture:
#   1) any leading indentation/characters + 'model.setObjective('
#   2) the objective expression (greedy, i.e., can contain commas, parentheses, etc.)
#   3) the comma, whitespace, GRB.<SENSE> and the closing parenthesis
#
# Example line matched:
#   model.setObjective(quicksum([i, u]), GRB.MINIMIZE)
# Captures:
#   group(1) -> 'model.setObjective('
#   group(2) -> 'quicksum([i, u])'
#   group(3) -> ', GRB.MINIMIZE)'
pattern = re.compile(r'^(.*model\.setObjective\()(.+)(,\s*GRB\.\w+\).*)$')

for root, dirs, files in os.walk(BASE_DIR):
    # Only process directories ending with '_i'
    if root.endswith('_i'):
        for filename in files:
            if filename == "optimus-code.py":
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    match = pattern.match(line)
                    if match:
                        # group(1): up to '('
                        # group(2): expression inside setObjective
                        # group(3): from the comma before GRB up through the end
                        before = match.group(1)
                        expr   = match.group(2)
                        after  = match.group(3)
                        
                        # Insert 2*(...) around the expression
                        new_line = f"{before}2*({expr}){after}"
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
