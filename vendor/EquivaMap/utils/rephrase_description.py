import json
from openai import OpenAI
import time
import os
import re

# Set your OpenAI API key
client = OpenAI(api_key='your-api-key')

# Define the root directory where your directories are located
root_directory = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

def paraphrase(text):
    prompt = f"Paraphrase the following text while keeping its original meaning:\n\n\"{text}\"\n\nParaphrased:"
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are an expert in paraphrasing."},
            {"role": "user", "content": prompt}
        ]
    )
        paraphrased_text = response.choices[0].message.content
        return clean_text(paraphrased_text)
    except Exception as e:
        print(f"Error paraphrasing text: {e}")
        return text  # Return the original text if there's an error

def clean_text(text):
    # Remove leading/trailing quotes, backslashes, and extra whitespace
    text = text.strip().strip('"').strip("'")  # Remove surrounding quotes
    text = text.replace("\\", "")  # Remove any backslashes
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces into a single space
    text = re.sub(r'\(\s+', '(', text)  # Remove spaces after opening parentheses
    text = re.sub(r'\s+\)', ')', text)  # Remove spaces before closing parentheses
    return text

def update_descriptions(data):
    # Update descriptions in parameters
    for param in data.get('parameters', {}).values():
        original_desc = param['description']
        print(f"Original parameter description: {original_desc}")
        paraphrased_desc = paraphrase(original_desc)
        print(f"Paraphrased parameter description: {paraphrased_desc}\n")
        param['description'] = paraphrased_desc
    # Update descriptions in variables
    for var in data.get('variables', {}).values():
        original_desc = var['description']
        print(f"Original variable description: {original_desc}")
        paraphrased_desc = paraphrase(original_desc)
        print(f"Paraphrased variable description: {paraphrased_desc}\n")
        var['description'] = paraphrased_desc
    # Update descriptions in constraints
    for constraint in data.get('constraints', []):
        original_desc = constraint['description']
        print(f"Original constraint description: {original_desc}")
        paraphrased_desc = paraphrase(original_desc)
        print(f"Paraphrased constraint description: {paraphrased_desc}\n")
        constraint['description'] = paraphrased_desc
    # Update description in objective
    if 'objective' in data and 'description' in data['objective']:
        original_desc = data['objective']['description']
        print(f"Original objective description: {original_desc}")
        paraphrased_desc = paraphrase(original_desc)
        print(f"Paraphrased objective description: {paraphrased_desc}\n")
        data['objective']['description'] = paraphrased_desc
    return data

def save_updated_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Traverse the directory structure
for subdir, _, files in os.walk(root_directory):
    # Process only directories that end with '_0' and contain problem_info.json
    if subdir.endswith('_a') and 'problem_info.json' in files:
        problem_info_path = os.path.join(subdir, 'problem_info.json')
        print(f'Processing: {problem_info_path}')
        
        # Load the JSON data
        with open(problem_info_path, 'r') as f:
            data = json.load(f)
        
        # Update the descriptions
        updated_data = update_descriptions(data)
        
        # Save the updated JSON data
        save_updated_json(updated_data, problem_info_path)
        
        print(f'Updated descriptions in: {problem_info_path}\n')