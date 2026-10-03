import json
from openai import OpenAI
import os
import re

# Set your OpenAI API key
client = OpenAI(api_key='your-api-key')


# Base directory containing all the problems
base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy/'

# Function to load data from a JSON file
def load_problem_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        variables = data.get('variables', {})
        constraints = data.get('constraints', [])
        objective = data.get('objective', {})
        return variables, constraints, objective


# Function to extract constraints involving a specific variable
def get_constraints_involving_variable(variable_name, constraints):
    involved_constraints = []
    for constraint in constraints:
        formulation = constraint.get('formulation', '')
        description = constraint.get('description', '')
        if variable_name in formulation:
            involved_constraints.append({
                'description': description,
                'formulation': formulation
            })
    return involved_constraints

# Function to prepare the prompt for GPT using CoT approach
# Function to prepare the prompt for GPT
def create_prompt(var_name1, var_info1, variables2, constraints1, objective1, constraints2, objective2):
    # Gather information for the variable from Problem 1
    var_desc1 = var_info1.get('description', '')
    constraints_involving_var1 = get_constraints_involving_variable(var_name1, constraints1)
    objective_formulation1 = objective1.get('formulation', '')
    var_in_objective1 = var_name1 in objective_formulation1

    # Start constructing the prompt
    prompt = f"""
You are an AI language model assisting in mapping variables between two optimization problems by analyzing their roles in constraints and the objective function.

**Variable from Problem 1:**
- **Name:** {var_name1}
- **Description:** {var_desc1}
- **Constraints involving {var_name1}:**
"""
    for constraint in constraints_involving_var1:
        prompt += f"  - Description: {constraint['description']}\n"
        prompt += f"    Formulation: {constraint['formulation']}\n"

    prompt += f"- **In Objective Function:** {'Yes' if var_in_objective1 else 'No'}\n"

    # Provide information about variables from Problem 2
    prompt += "\n**Variables from Problem 2:**\n"

    for var_name2, var_info2 in variables2.items():
        var_desc2 = var_info2.get('description', '')
        constraints_involving_var2 = get_constraints_involving_variable(var_name2, constraints2)
        objective_formulation2 = objective2.get('formulation', '')
        var_in_objective2 = var_name2 in objective_formulation2

        prompt += f"- **Name:** {var_name2}\n"
        prompt += f"  **Description:** {var_desc2}\n"
        prompt += f"  **Constraints involving {var_name2}:**\n"

        for constraint in constraints_involving_var2:
            prompt += f"    - Description: {constraint['description']}\n"
            prompt += f"      Formulation: {constraint['formulation']}\n"

        prompt += f"  **In Objective Function:** {'Yes' if var_in_objective2 else 'No'}\n\n"

    # Final instruction with explicit formatting requirement
    prompt += f"""
Based on the above information, find the best mapping from variables in Problem 2 for the variable '{var_name1}' from Problem 1. The mapping can be a linear combination of variables from Problem 2, possibly with constant multipliers. Your goal is to express '{var_name1}' in terms of variables from Problem 2, as accurately as possible, based on their roles in the constraints and objective functions.

**Important Instructions:**

- **Provide only the mapping for '{var_name1}' as a JSON object.**
- **Do not include any additional text, explanations, or formatting.**
- **The JSON object must follow this exact structure:**

{{
  "{var_name1}": [
    {{
      "constant": constant_value_1,
      "variable": "variable_name_1"
    }},
    {{
      "constant": constant_value_2,
      "variable": "variable_name_2"
    }},
    ...
  ]
}}

- **If there is only one term in the mapping, the list should contain a single object.**
- **Use numerical values for constants (decimals), and enclose variable names in double quotes ("").**

**Examples:**

1. If the best mapping is '0.1*a', your response should be:

{{
  "{var_name1}": [
    {{
      "constant": 0.1,
      "variable": "a"
    }}
  ]
}}

2. If the best mapping is '0.1*a + 0.01*b', your response should be:

{{
  "{var_name1}": [
    {{
      "constant": 0.1,
      "variable": "a"
    }},
    {{
      "constant": 0.01,
      "variable": "b"
    }}
  ]
}}

3. If the best mapping is a single variable 'a' with a coefficient of 1, your response should be:

{{
  "{var_name1}": [
    {{
      "constant": 1,
      "variable": "a"
    }}
  ]
}}

4. If there is no direct mapping, your response should be:

{{
  "{var_name1}": [
    {{
      "constant": "none",
      "variable": "none"
    }}
  ]
}}

Please ensure your response is a valid JSON object that can be parsed by standard JSON parsers. 
"""
    return prompt



# Function to get the mapping using OpenAI ChatCompletion API
import json
import re

# Function to get the mapping using OpenAI ChatCompletion API
def get_variable_mapping(variables1, variables2, constraints1, objective1, constraints2, objective2):
    mappings = {}
    for var_name1, var_info1 in variables1.items():
        prompt = create_prompt(var_name1, var_info1, variables2, constraints1, objective1, constraints2, objective2)
        try:
            response = client.chat.completions.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": "You are an expert in optimization problems and variable mappings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # More deterministic output
            )
            content = response.choices[0].message.content.strip()
            print(f"GPT response for variable '{var_name1}':\n{content}\n")

            # Extract JSON object from the response using regex
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                json_str = match.group(0)
                mapping_json = json.loads(json_str)
                mapping_terms = mapping_json.get(var_name1)
                if not mapping_terms:
                    print(f"Warning: No mapping found for variable '{var_name1}' in the GPT response.")
                    mappings[var_name1] = None
                    continue
            else:
                print(f"Error: Could not find JSON object in GPT response for variable '{var_name1}'.")
                mappings[var_name1] = None
                continue

            # Validate each term in the mapping
            valid = True
            for term in mapping_terms:
                constant = term.get('constant')
                variable = term.get('variable')
                if constant is None or variable is None:
                    print(f"Warning: Missing 'constant' or 'variable' in term {term} for variable '{var_name1}'.")
                    valid = False
                    break
                if not isinstance(constant, (int, float)):
                    print(f"Warning: 'constant' must be a number in term {term} for variable '{var_name1}'.")
                    valid = False
                    break
                if variable not in variables2:
                    print(f"Warning: The variable '{variable}' in the mapping is not in Problem 2 variables.")
                    valid = False
                    break
            if valid:
                mappings[var_name1] = mapping_terms
            else:
                mappings[var_name1] = None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for variable '{var_name1}': {e}")
            mappings[var_name1] = None
        except Exception as e:
            print(f"Error mapping variable '{var_name1}': {e}")
            mappings[var_name1] = None
    return mappings


# Function to process all problem directories, focusing only on subdirectories ending with '_0'
def process_all_problems(base_dir):
    # Iterate over all directories in the base directory
    for item in os.listdir(base_dir):
        problem_dir = os.path.join(base_dir, item)
        # Check if it's a directory
        if os.path.isdir(problem_dir):
            main_problem_file = os.path.join(problem_dir, 'problem_info.json')
            # Check if the main problem_info.json exists
            if os.path.isfile(main_problem_file):
                # Load the main problem data
                variables1, constraints1, objective1 = load_problem_data(main_problem_file)
                # Now, find subdirectories that end with '_0' (e.g., 1_0, 2_0, etc.)
                for sub_item in os.listdir(problem_dir):
                    sub_dir = os.path.join(problem_dir, sub_item)
                    if os.path.isdir(sub_dir) and sub_item.endswith('_d'):
                        sub_problem_file = os.path.join(sub_dir, 'problem_info.json')
                        # Check if the subproblem problem_info.json exists
                        if os.path.isfile(sub_problem_file):
                            # Load the subproblem data
                            variables2, constraints2, objective2 = load_problem_data(sub_problem_file)
                            # Get the variable mappings
                            variable_mappings = get_variable_mapping(
                                variables1, variables2, constraints1, objective1, constraints2, objective2
                            )
                            # Print the mappings
                            print(f"Variable Mappings for {problem_dir} and {sub_dir}:")
                            for var1, var2 in variable_mappings.items():
                                print(f"{var1} --> {var2}")
                            # Output the mappings to a JSON file
                            output_file = os.path.join(sub_dir, 'variable_mappings.json')
                            with open(output_file, 'w') as file:
                                json.dump(variable_mappings, file, indent=4)
                            print(f"Mappings saved to {output_file}\n")
                        else:
                            print(f"No problem_info.json found in {sub_dir}")
            else:
                print(f"No problem_info.json found in {problem_dir}")

# Run the processing function
process_all_problems(base_dir)