import os
import json
import re
from openai import OpenAI

# Set your OpenAI API key
client = OpenAI(api_key='your-api-key')

# -----------------------------------
# 1. Helper Functions
# -----------------------------------

def is_problem_lp(problem_dir):
    """
    Determines if the problem is an LP or MIP by reading the third line
    of optimus-code.py. Returns 'LP', 'MIP', or None if it cannot determine.
    """
    base_name = os.path.basename(problem_dir)
    i_subdir = os.path.join(problem_dir, f"{base_name}_j")
    code_file = os.path.join(i_subdir, "optimus-code.py")

    if not os.path.isfile(code_file):
        return None

    with open(code_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    # The third line should be something like: "# Problem type: LP"
    problem_type_line = lines[2].strip()
    if "# Problem type: LP" in problem_type_line:
        return "LP"
    elif "# Problem type: MIP" in problem_type_line:
        return "MIP"
    else:
        return None

def load_problem_info(json_path):
    """
    Loads the problem_info.json content. Returns a dict (or None if fails).
    """
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)

def create_equivalence_prompt(problem_info_1, problem_info_2, problem_type):
    """
    Create a prompt that asks GPT to decide whether problem_info_1
    and problem_info_2 represent equivalent formulations.
    The `problem_type` can be 'LP', 'MIP', or None for custom logic.
    """
    text_1 = json.dumps(problem_info_1, indent=2)
    text_2 = json.dumps(problem_info_2, indent=2)
    
    prompt = f"""
You are given two optimization problem formulations (both declared as {problem_type if problem_type else "Unknown Type"}).
Decide if they are equivalent formulations.

First problem formulation (Problem A):
{text_1}

Second problem formulation (Problem B):
{text_2}

Based on the data, please respond with exactly one of the following:
- "Equivalent" if these two are the same formulation. Be rigorous in your reasoning. 
- "Not Equivalent" if they are different. When you are not sure, say "Not Equivalent".

Briefly explain the reasoning in 1-2 sentences, then end with the word "Equivalent" or "Not Equivalent" on its own line.
"""
    print(prompt)
    return prompt

def ask_gpt_equivalence(problem_info_1, problem_info_2, problem_type):
    """
    Calls the OpenAI GPT model to decide whether two problem formulations 
    are equivalent. Returns (bool_equivalent, raw_gpt_response).
    """
    prompt = create_equivalence_prompt(problem_info_1, problem_info_2, problem_type)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in mathematical optimization problems. "
                        "You decide if two given formulations represent the same problem."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content.strip()
        # Extract the final word in the response for definitive judgment
        last_line = content.splitlines()[-1].strip().lower()
        
        if last_line == "equivalent":
            return True, content
        elif last_line == "not equivalent":
            return False, content
        else:
            # If the final line isn't clear, treat as "Not Equivalent" by default
            return False, content

    except Exception as e:
        print(f"[ERROR] OpenAI request failed: {e}")
        return False, str(e)


# -----------------------------------
# 2. Main Processing Function
# -----------------------------------

def main():
    base_dir = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"
    
    # We'll store the results in dictionaries keyed by problem type.
    results = {
        "LP": {
            "equivalent": [],
            "nonequivalent": []
        },
        "MIP": {
            "equivalent": [],
            "nonequivalent": []
        }
    }
    
    # List all items in base_dir; keep only those that are directories and numeric
    candidates = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    
    for problem_dir in candidates:
        problem_path = os.path.join(base_dir, problem_dir)
        
        # Determine if LP or MIP
        problem_type = is_problem_lp(problem_path)  # returns 'LP', 'MIP', or None
        if problem_type not in ["LP", "MIP"]:
            # Skip if it's not recognized or has no optimus-code.py
            print(f"Skipping '{problem_dir}' because it's neither recognized as LP nor MIP.")
            continue
        
        # Now handle LP or MIP in the same pipeline:
        
        # Original problem_info.json
        original_info_path = os.path.join(problem_path, "problem_info.json")
        if not os.path.isfile(original_info_path):
            print(f"Skipping '{problem_dir}' - no problem_info.json found.")
            continue
        
        # Subdirectory with _g (e.g., "243_g")
        c_subdir_name = f"{problem_dir}_l"
        c_subdir_path = os.path.join(problem_path, c_subdir_name)
        new_info_path = os.path.join(c_subdir_path, "problem_info.json")
        
        if not os.path.isdir(c_subdir_path) or not os.path.isfile(new_info_path):
            print(f"Skipping '{problem_dir}' - no '{c_subdir_name}' subdirectory or problem_info.json.")
            continue
        
        # Load JSON data
        original_data = load_problem_info(original_info_path)
        new_data = load_problem_info(new_info_path)
        
        if not (original_data and new_data):
            print(f"Skipping '{problem_dir}' - failed to load JSON data.")
            continue
        
        # Ask GPT if they are equivalent
        is_equiv, gpt_response = ask_gpt_equivalence(original_data, new_data, problem_type)
        
        if is_equiv:
            results[problem_type]["equivalent"].append(problem_dir)
            label = "Equivalent"
        else:
            results[problem_type]["nonequivalent"].append(problem_dir)
            label = "Not Equivalent"
        
        # Optionally, print the GPT's short decision for debugging
        print(f"Problem {problem_dir} ({problem_type}): {label}")
        print("GPT reasoning (truncated):")
        print(gpt_response[:300] + "...\n")  # show partial response
    
    # Final summary
    print("\n-- Summary of Equivalence by Problem Type --")
    for pt in ["LP", "MIP"]:
        eq_list = results[pt]["equivalent"]
        neq_list = results[pt]["nonequivalent"]
        print(f"Problem Type: {pt}")
        print("  Equivalent Directories:")
        print("   ", eq_list)
        print(f"  Count: {len(eq_list)}\n")
        print("  Not Equivalent Directories:")
        print("   ", neq_list)
        print(f"  Count: {len(neq_list)}\n")

if __name__ == "__main__":
    main()
