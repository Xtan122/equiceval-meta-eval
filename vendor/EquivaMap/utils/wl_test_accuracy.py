import glob
import os
import subprocess

def get_problem_type(optimus_path):
    """
    Reads the 'optimus_code.py' (or 'optimus-code.py') and returns 'LP' or 'MIP' if found.
    """
    with open(optimus_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Problem type:"):
                # e.g. "# Problem type: LP" or "# Problem type: MIP"
                return line.split(":", 1)[1].strip()
    return None

def get_wl_hash(wltest_path):
    """
    Runs `python wltest.py` at the given path and returns the hash printed
    (the part after 'WL Hash:').
    """
    try:
        result = subprocess.check_output(["python", wltest_path], text=True)
    except Exception as e:
        print(f"Could not run wltest.py at {wltest_path}: {e}")
        return None
    
    # Look for a line like: "WL Hash: abc123xyz"
    for line in result.splitlines():
        if line.startswith("WL Hash:"):
            return line.split(":", 1)[1].strip()
    return None

def main():
    base_dir = '/Users/stevenzhai/Desktop/MILP_data/sample-data-easy'

    # Pattern to find every optimus_code.py under something like: <id>_c/optimus_code.py
    pattern = os.path.join(base_dir, '*', '*_c', 'optimus-code.py')
    
    # Dictionary to hold result data:
    # {
    #   problem_id: {
    #       'type': 'LP' or 'MIP',
    #       'c_hash': <hash or None>,
    #       'i_hash': <hash or None>
    #   },
    #   ...
    # }
    problem_data = {}

    # We'll also track how many times each WL hash appears across all files
    hash_frequency = {}

    # We'll track total LP, total MIP, and how many matched (c_hash == i_hash)
    lp_count, lp_matches = 0, 0
    mip_count, mip_matches = 0, 0

    # Find all matching optimus_code.py files in _c folders
    c_optimus_paths = glob.glob(pattern)
    for c_optimus_path in c_optimus_paths:
        # Example c_optimus_path:
        #   /Users/.../266/266_c/optimus_code.py
        
        # _c directory
        c_dir = os.path.dirname(c_optimus_path)  
        # base problem directory
        problem_dir = os.path.dirname(c_dir)     
        # c_dirname is something like "266_c"
        c_dirname = os.path.basename(c_dir)      
        
        # Extract the "problem_id" from "266_c" by stripping the trailing "_c"
        if c_dirname.endswith("_c"):
            problem_id = c_dirname[:-2]  # remove "_c"
        else:
            # fallback if something doesn't match the pattern
            continue
        
        # Read the problem type from the c_optimus_path
        problem_type = get_problem_type(c_optimus_path)

        # Now find the matching i_dir, e.g. "266_i"
        i_dirname = f"{problem_id}_i"
        i_dir = os.path.join(problem_dir, i_dirname)

        # We'll look for wltest.py in both c_dir and i_dir
        wltest_c = os.path.join(c_dir, 'wltest.py')
        wltest_i = os.path.join(i_dir, 'wltest.py')

        # Get the WL hashes if files exist
        c_hash = get_wl_hash(wltest_c) if os.path.isfile(wltest_c) else None
        i_hash = get_wl_hash(wltest_i) if os.path.isfile(wltest_i) else None

        # Store in our dictionary
        problem_data[problem_id] = {
            'type': problem_type,
            'c_hash': c_hash,
            'i_hash': i_hash
        }

        # Update global frequency counts
        for hsh in (c_hash, i_hash):
            if hsh is not None:
                hash_frequency[hsh] = hash_frequency.get(hsh, 0) + 1

        # If we know it's LP or MIP, increment counters
        if problem_type == "LP":
            lp_count += 1
            if c_hash and i_hash and c_hash == i_hash:
                lp_matches += 1
        elif problem_type == "MIP":
            mip_count += 1
            if c_hash and i_hash and c_hash == i_hash:
                mip_matches += 1

    # Print a summary
    print("=== Summary of WL Hashes by Problem ===")
    print(f"{'Problem':>8} | {'Type':>3} | {'c_hash':>32} | {'i_hash':>32}")
    # Sort problem IDs as integers if numeric
    for pid in sorted(problem_data.keys(), key=lambda x: int(x) if x.isdigit() else x):
        ptype = problem_data[pid]['type']
        c_hash = problem_data[pid]['c_hash'] or '-'
        i_hash = problem_data[pid]['i_hash'] or '-'
        print(f"{pid:>8} | {ptype:>3} | {c_hash:>32} | {i_hash:>32}")

    print("\n=== WL Hash Frequency Across All Files ===")
    # Sort by frequency descending
    for hsh, freq in sorted(hash_frequency.items(), key=lambda x: x[1], reverse=True):
        print(f"{hsh} : {freq}")

    # Final matches summary
    print("\n=== Final LP / MIP Matches ===")
    if lp_count > 0:
        print(f"LP : {lp_matches}/{lp_count} match")
    else:
        print("LP : no LP problems found")

    if mip_count > 0:
        print(f"MIP: {mip_matches}/{mip_count} match")
    else:
        print("MIP: no MIP problems found")

if __name__ == "__main__":
    main()
