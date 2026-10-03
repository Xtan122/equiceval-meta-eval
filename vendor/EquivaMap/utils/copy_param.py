import os
import shutil

def copy_all_parameters(base_path, source_suffix, target_suffix):
    """
    Copies all parameters.json files from directories ending with source_suffix to corresponding directories ending with target_suffix.
    
    :param base_path: The base directory to start searching.
    :param source_suffix: The suffix of the source directories (e.g., "_0").
    :param target_suffix: The suffix of the target directories (e.g., "_2").
    """
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.endswith(source_suffix):
                source_dir = os.path.join(root, dir_name)
                target_dir = source_dir.replace(source_suffix, target_suffix)
                
                source_file = os.path.join(source_dir, "parameters.json")
                target_file = os.path.join(target_dir, "parameters.json")
                
                # Ensure the source file exists
                if not os.path.isfile(source_file):
                    print(f"Source file does not exist: {source_file}")
                    continue
                
                # Create the target directory if it doesn't exist
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy the file
                shutil.copy(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")

# Define the base path containing all subdirectories
base_path = "/Users/stevenzhai/Desktop/MILP_data/sample-data-easy"

# Define the suffixes for source and target directories
source_suffix = "_c"
target_suffix = "_i"

# Execute the process
copy_all_parameters(base_path, source_suffix, target_suffix)
