import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Explicitly load the .env file from the current directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Add the radextract directory to the Python path to locate the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../radextract')))

try:
    # Import the core class from the radextract library
    from structure_report import RadiologyReportStructurer
except ImportError as e:
    print(f"Error importing RadiologyReportStructurer: {e}")
    print("Please ensure the 'radextract' repository is cloned in the same directory as this script.")
    sys.exit(1)

# Retrieve the Gemini API key from environment variables
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

def decompose_section(section_text: str, model_id: str = 'gemini-1.5-pro') -> dict:
    """
    Decomposes a single section of a radiology report (e.g., findings or impression).
    Args:
        section_text: The unstructured text of the report section.
        model_id: The Gemini model to use for the decomposition.
    Returns:
        A dictionary containing the structured section data.
    """
    if not isinstance(section_text, str) or not section_text.strip():
        print("Skipping empty, non-string, or whitespace-only report section.")
        return {}
    try:
        # Initialize the structurer with your API key and desired model
        structurer = RadiologyReportStructurer(api_key=API_KEY, model_id=model_id)
        # Process the report section to get the structured output
        structured_result = structurer.predict(section_text)
        return structured_result
    except Exception as e:
        print(f"An error occurred while processing the report section: {e}")
        return {"error": str(e)}

def process_reports_from_csv(input_path: str, findings_output_path: str, impressions_output_path: str):
    """
    Reads radiology reports from a CSV file, processes each patient once, and saves
    the results into separate files for findings and impressions.
    Args:
        input_path: Path to the input CSV file.
        findings_output_path: Path for the decomposed findings JSON output.
        impressions_output_path: Path for the decomposed impressions JSON output.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    decomposed_findings = {}
    decomposed_impressions = {}
    processed_patient_ids = set()

    for index, row in df.iterrows():
        volume_name = row.get('VolumeName')
        if not volume_name:
            continue
        
        # Extract patient ID, e.g., 'train_1_a' from 'train_1_a_1.nii.gz'
        patient_id = "_".join(volume_name.split('_')[:-1])

        if patient_id in processed_patient_ids:
            print(f"Skipping already processed patient: {patient_id}")
            continue

        findings = row.get('Findings_EN', '')
        impressions = row.get('Impressions_EN', '')

        print(f"Decomposing findings for patient {patient_id}...")
        decomposed_findings[patient_id] = decompose_section(f"FINDINGS: {findings}")
        
        print(f"Decomposing impressions for patient {patient_id}...")
        decomposed_impressions[patient_id] = decompose_section(f"IMPRESSION: {impressions}")
        
        processed_patient_ids.add(patient_id)

        # Save the results every 5 new patients
        if len(processed_patient_ids) % 5 == 0:
            with open(findings_output_path, 'w', encoding='utf-8') as f:
                json.dump(decomposed_findings, f, indent=4)
            print(f"Saved intermediate findings to {findings_output_path}")

            with open(impressions_output_path, 'w', encoding='utf-8') as f:
                json.dump(decomposed_impressions, f, indent=4)
            print(f"Saved intermediate impressions to {impressions_output_path}")

    # Save the final results
    with open(findings_output_path, 'w', encoding='utf-8') as f:
        json.dump(decomposed_findings, f, indent=4)
    print(f"Decomposition complete. Final findings saved to {findings_output_path}")

    with open(impressions_output_path, 'w', encoding='utf-8') as f:
        json.dump(decomposed_impressions, f, indent=4)
    print(f"Decomposition complete. Final impressions saved to {impressions_output_path}")

if __name__ == '__main__':
    # Define the input and output file paths
    INPUT_FILE = 'data/train_reports.csv'
    FINDINGS_OUTPUT_FILE = 'output/ct_rate/radextract_findings_output.json'
    IMPRESSIONS_OUTPUT_FILE = 'output/ct_rate/radextract_impressions_output.json'
    
    # Start the decomposition process
    process_reports_from_csv(INPUT_FILE, FINDINGS_OUTPUT_FILE, IMPRESSIONS_OUTPUT_FILE)
