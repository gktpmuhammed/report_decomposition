import json

def find_patients_with_key(input_file, key_to_find, output_file):
    """
    Finds patient IDs in a JSON file that have a specific non-empty key
    and not more than one other field that is not 'Conclusion' or 'Findings'.

    Args:
        input_file (str): Path to the input JSON file.
        key_to_find (str): The key to look for (e.g., "Conclusion" or "Findings").
        output_file (str): Path to the output file for patient IDs.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    found_ids = set()
    for pid, patient_data in data.items():
        if isinstance(patient_data, dict) and patient_data.get(key_to_find):
            # Check for other fields that are not "Conclusion" or "Findings"
            other_keys = set(patient_data.keys()) - {"Conclusion", "Findings"}
            if len(other_keys) == 0:
                found_ids.add(pid)

    sorted_ids = sorted(list(found_ids))

    with open(output_file, 'w') as f:
        for patient_id in sorted_ids:
            f.write(f"{patient_id}\n")

    print(f"Found {len(sorted_ids)} patients meeting the criteria in {input_file}.")
    print(f"Patient IDs saved to {output_file}")


if __name__ == "__main__":
    # Process conclusions file
    conc_json_file = "data/conc_info_manual_v5.json"
    conc_output_file = "patients_with_conclusion.txt"
    find_patients_with_key(conc_json_file, "Conclusion", conc_output_file)

    print("-" * 20)

    # Process descriptions/findings file
    desc_json_file = "data/desc_info_manual_v5.json"
    desc_output_file = "patients_with_findings.txt"
    find_patients_with_key(desc_json_file, "Findings", desc_output_file) 