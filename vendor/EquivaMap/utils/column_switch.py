import json
import re
import random
import glob
import os

def swap_terms_in_formulation(formulation_str):
    # Pattern to match terms separated by '+' or '\\times'
    pattern = r"(\b[A-Za-z0-9]+\b)(\s*\+\s*\b[A-Za-z0-9]+\b)+"
    pattern_complex = r"(\b[A-Za-z0-9]+\b\s*\\times\s*\b[A-Za-z0-9]+\b)(\s*\+\s*\b[A-Za-z0-9]+\b\s*\\times\s*\b[A-Za-z0-9]+\b)+"

    # Check for complex pattern (with \\times)
    match_complex = re.search(pattern_complex, formulation_str)
    if match_complex:
        # Split the terms by '+'
        terms = re.split(r"\s*\+\s*", match_complex.group())
        # Shuffle the terms randomly
        random.shuffle(terms)
        # Join the terms back with '+'
        swapped = " + ".join(terms)
        return formulation_str[:match_complex.start()] + swapped + formulation_str[match_complex.end():]

    # Check for simple pattern
    match = re.search(pattern, formulation_str)
    if match:
        terms = re.split(r"\s*\+\s*", match.group())
        random.shuffle(terms)
        swapped = " + ".join(terms)
        return formulation_str[:match.start()] + swapped + formulation_str[match.end():]

    return formulation_str

def swap_terms_in_code(code_str):
    # For code strings, similar patterns:
    #   "y + x"
    #   "Z * y + D * x"
    pattern_complex_code = r"([A-Za-z0-9]+)\s*\*\s*([A-Za-z0-9]+)\s*\+\s*([A-Za-z0-9]+)\s*\*\s*([A-Za-z0-9]+)"
    pattern_simple_code = r"(\b[A-Za-z0-9]+\b)\s*\+\s*(\b[A-Za-z0-9]+\b)"

    match_complex = re.search(pattern_complex_code, code_str)
    if match_complex:
        # "Z * y + D * x" -> "D * x + Z * y"
        swapped = f"{match_complex.group(3)} * {match_complex.group(4)} + {match_complex.group(1)} * {match_complex.group(2)}"
        return code_str[:match_complex.start()] + swapped + code_str[match_complex.end():]

    match_simple = re.search(pattern_simple_code, code_str)
    if match_simple:
        # "y + x" -> "x + y"
        swapped = f"{match_simple.group(2)} + {match_simple.group(1)}"
        return code_str[:match_simple.start()] + swapped + code_str[match_simple.end():]

    return code_str

# Directory that contains all the data. Adjust this as needed.
base_directory = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/"

# Find all directories that end with "_a"
pattern = os.path.join(base_directory, "**", "*_a", "problem_info.json")
input_files = glob.glob(pattern, recursive=True)

for input_file in input_files:
    # Construct the output file by replacing "_a" with "_b"
    output_file = input_file.replace("_a", "_b")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load the JSON from the input file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process constraints
    if "constraints" in data:
        for constraint in data["constraints"]:
            if "formulation" in constraint:
                constraint["formulation"] = swap_terms_in_formulation(constraint["formulation"])

    # Process objective
    if "objective" in data and "formulation" in data["objective"]:
        data["objective"]["formulation"] = swap_terms_in_formulation(data["objective"]["formulation"])

    # Write the modified JSON to the output file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print("Term swapping complete. Modified data written to:", output_file)
