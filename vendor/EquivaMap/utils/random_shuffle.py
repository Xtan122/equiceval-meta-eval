import os
import json
import random
import shutil

# Define your base directory
base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# Function to shuffle JSON content
def shuffle_json_content(data):
    # Shuffle parameters, variables, and constraints as in the previous example
    parameters_items = list(data.get("parameters", {}).items())
    variables_items = list(data.get("variables", {}).items())
    constraints_items = data.get("constraints", [])

    random.shuffle(parameters_items)
    random.shuffle(variables_items)
    random.shuffle(constraints_items)

    # Assign shuffled data back to the dictionary
    data["parameters"] = dict(parameters_items)
    data["variables"] = dict(variables_items)
    data["constraints"] = constraints_items

    return data

# Walk through directories to locate each `problem_info.json`
for root, dirs, files in os.walk(base_dir):
    if root.endswith("_b"):
        # Create the new directory path
        new_dir = root.replace("_b", "_c")
        os.makedirs(new_dir, exist_ok=True)
        
        for file in files:
            # Process problem_info.json to shuffle and save
            if file == "problem_info.json":
                original_path = os.path.join(root, file)
                with open(original_path, "r") as f:
                    data = json.load(f)
                
                # Shuffle the content
                shuffled_data = shuffle_json_content(data)

                # Define the new file path and save the shuffled data
                new_path = os.path.join(new_dir, file)
                with open(new_path, "w") as f:
                    json.dump(shuffled_data, f, indent=4)
                
                print(f"Processed and saved: {new_path}")
            
            # Copy additional files directly
            elif file in ["optimus-code.py", "parameters.json"]:
                source_path = os.path.join(root, file)
                dest_path = os.path.join(new_dir, file)
                shutil.copyfile(source_path, dest_path)
                print(f"Copied: {source_path} to {dest_path}")