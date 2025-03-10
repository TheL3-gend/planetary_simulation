#!/usr/bin/env python3
# scale_factor_fix.py - Script to check and fix SCALE_FACTOR in constants.py

import os
import sys
import re

def fix_scale_factor():
    """Check and fix the SCALE_FACTOR in constants.py"""
    # Path to constants.py
    constants_path = "constants.py"
    
    # Check if file exists
    if not os.path.exists(constants_path):
        print(f"Error: {constants_path} not found!")
        return False
    
    # Read the file
    try:
        with open(constants_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {constants_path}: {e}")
        return False
    
    # Look for SCALE_FACTOR definition
    scale_factor_pattern = r'SCALE_FACTOR\s*=\s*([0-9.e+-]+)'
    match = re.search(scale_factor_pattern, content)
    
    if not match:
        print(f"Could not find SCALE_FACTOR in {constants_path}")
        return False
    
    # Get current value
    current_value = match.group(1)
    print(f"Current SCALE_FACTOR = {current_value}")
    
    # Suggest a better value
    suggestion = "1e8"  # More reasonable scale factor
    
    # Ask for confirmation
    print("\nThe SCALE_FACTOR value might be causing your black screen issue.")
    print(f"Would you like to change it from {current_value} to {suggestion}?")
    response = input("Change SCALE_FACTOR? (y/n): ").strip().lower()
    
    if response != 'y':
        print("No changes made.")
        return False
    
    # Replace the value
    new_content = re.sub(
        scale_factor_pattern,
        f'SCALE_FACTOR = {suggestion}',
        content
    )
    
    # Write the modified file
    try:
        with open(constants_path, 'w') as f:
            f.write(new_content)
        print(f"SCALE_FACTOR updated to {suggestion} in {constants_path}")
        return True
    except Exception as e:
        print(f"Error writing to {constants_path}: {e}")
        return False

def main():
    print("=== SCALE_FACTOR Fix Utility ===")
    print("This script will check and potentially fix the SCALE_FACTOR in constants.py")
    print("A large SCALE_FACTOR can make planets too small to see, resulting in a black screen.")
    
    success = fix_scale_factor()
    
    if success:
        print("\nFix applied successfully!")
        print("Now try running your simulation again with:")
        print("  python simplified_main.py")
        print("\nIf this works, you can try your original main.py again.")
    else:
        print("\nCould not apply fix automatically.")
        print("Consider manually editing constants.py and changing SCALE_FACTOR to 1e8.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())