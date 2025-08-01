#!/usr/bin/env python3
"""
Script to merge missing JSON files with main manual JSON files
for both concentration (conc) and description (desc) data.
"""

import json
import os
from typing import Dict, Any

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def save_json_file(data: Dict[str, Any], filepath: str) -> None:
    """Save data to a JSON file."""
    print(f"Saving to {filepath}...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(data)} entries to {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")

def merge_json_data(main_data: Dict[str, Any], missing_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge missing data into main data.
    Missing data entries will be added to main data.
    If there are conflicts, missing data will take precedence.
    """
    merged_data = main_data.copy()
    
    print(f"Main data has {len(main_data)} entries")
    print(f"Missing data has {len(missing_data)} entries")
    
    # Add all missing entries to the merged data
    for key, value in missing_data.items():
        if key in merged_data:
            print(f"Warning: Key '{key}' exists in both files. Using missing data version.")
        merged_data[key] = value
    
    print(f"Merged data has {len(merged_data)} entries")
    return merged_data

def main():
    """Main function to merge both concentration and description files."""
    data_dir = "output/ct_rate"
    
    # File paths
    files_to_merge = [
        {
            "main_file": os.path.join(data_dir, "conc_info_manual_v5.json"),
            "missing_file": os.path.join(data_dir, "decomposed_conc_missing.json"),
            "output_file": os.path.join(data_dir, "conc_info_manual_merged.json"),
            "type": "concentration"
        },
        {
            "main_file": os.path.join(data_dir, "desc_info_manual_v5.json"),
            "missing_file": os.path.join(data_dir, "decomposed_desc_missing.json"),
            "output_file": os.path.join(data_dir, "desc_info_manual_merged.json"),
            "type": "description"
        }
    ]
    
    for file_set in files_to_merge:
        print(f"\n{'='*50}")
        print(f"Processing {file_set['type']} files...")
        print(f"{'='*50}")
        
        # Load the files
        main_data = load_json_file(file_set["main_file"])
        missing_data = load_json_file(file_set["missing_file"])
        
        if not main_data and not missing_data:
            print(f"No data found for {file_set['type']}. Skipping...")
            continue
        
        # Merge the data
        merged_data = merge_json_data(main_data, missing_data)
        
        # Save the merged data
        save_json_file(merged_data, file_set["output_file"])
        
        print(f"✓ Completed merging {file_set['type']} files")
    
    print(f"\n{'='*50}")
    print("All merging operations completed!")
    print("Output files:")
    print("- output/ct_rate/conc_info_manual_merged.json")
    print("- output/ct_rate/desc_info_manual_merged.json")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 