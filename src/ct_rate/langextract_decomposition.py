import pandas as pd
import json
import langextract as lx
import textwrap
from pathlib import Path
from tqdm import tqdm
import os

# --- Anatomy Grouping from Table 6 ---
ANATOMY_MAPPING = {
    "Face": ["Face"],
    "Brain": ["Brain"],
    "Esophagus": ["Esophagus"],
    "Trachea": ["Trachea"],
    "Lung": ["Lung", "Lung upper lobe left", "Lung lower lobe left", "Lung upper lobe right", "Lung middle lobe right", "Lung lower lobe right"],
    "Heart": ["Heart", "Heart myocardium", "Heart atrium left", "Heart atrium right", "Heart ventricle left", "Heart ventricle right"],
    "Adrenal gland": ["Adrenal gland", "Adrenal gland left", "Adrenal gland right"],
    "Kidney": ["Kidney", "Kidney right", "Kidney left"],
    "Stomach": ["Stomach"],
    "Liver": ["Liver"],
    "Gallbladder": ["Gallbladder"],
    "Pancreas": ["Pancreas"],
    "Spleen": ["Spleen"],
    "Colon": ["Colon"],
    "Small bowel": ["Small bowel", "Duodenum"],
    "Urinary bladder": ["Urinary bladder"],
    "Aorta": ["Aorta"],
    "Inferior vena cava": ["Inferior vena cava"],
    "Portal vein and splenic vein": ["Portal vein and splenic vein"],
    "Pulmonary artery": ["Pulmonary artery"],
    "Iliac artery": ["Iliac artery", "Iliac artery right", "Iliac artery left"],
    "Iliac vena": ["Iliac vena", "Iliac vena right", "Iliac vena left"],
    "Lumbar vertebrae": ["Vertebrae L1-L4", "Lumbar vertebrae"],
    "Thoracic vertebrae": ["Vertebrae T1-T12", "Thoracic vertebrae"],
    "Cervical vertebrae": ["Vertebrae C1-C7", "Cervical vertebrae"],
    "Rib": ["Rib", "Rib right 1-12", "Rib left 1-12"],
    "Humerus": ["Humerus", "Humerus right", "Humerus left"],
    "Scapula": ["Scapula", "Scapula right", "Scapula left"],
    "Clavicula": ["Clavicula", "Clavicula right", "Clavicula left"],
    "Femur": ["Femur", "Femur right", "Femur left"],
    "Hip": ["Hip", "Hip right", "Hip left"],
    "Sacrum": ["Sacrum"],
    "Gluteus": ["Gluteus", "Gluteus maximus right", "Gluteus medius right", "Gluteus medius left", "Gluteus minimus left", "Gluteus minimus right"],
    "Iliopsoas": ["Iliopsoas", "Iliopsoas right", "Iliopsoas left"],
    "Autochthon": ["Autochthon", "Autochthon right", "Autochthon left"],
}

GROUPED_ANATOMIES = list(ANATOMY_MAPPING.keys())

# Configuration
BATCH_SIZE = 10  # Process this many reports per batch
MODEL_ID = "gemini-2.5-pro"  # Can be changed to gemini-1.5-pro for better quality

def get_extraction_task():
    """Defines the langextract prompt and examples for organ decomposition."""
    prompt = textwrap.dedent(f"""\
        Extract anatomical information from the provided medical report text.
        Identify all mentioned organs from the list and extract the complete, exact sentences describing each one.
        The list of possible organs is: {', '.join(GROUPED_ANATOMIES)}
        Only extract information for organs present in the text.
        Do not paraphrase, summarize, or omit any details from the original text.
    """)

    examples = [
        lx.data.ExampleData(
            text="""Findings: Liver is normal in size and contour. Spleen is enlarged.
                     Both kidneys demonstrate simple cysts. The heart is unremarkable.""",
            extractions=[
                lx.data.Extraction(
                    extraction_class="Liver",
                    extraction_text="Liver is normal in size and contour."
                ),
                lx.data.Extraction(
                    extraction_class="Spleen",
                    extraction_text="Spleen is enlarged."
                ),
                lx.data.Extraction(
                    extraction_class="Kidney",
                    extraction_text="Both kidneys demonstrate simple cysts."
                ),
                lx.data.Extraction(
                    extraction_class="Heart",
                    extraction_text="The heart is unremarkable."
                )
            ]
        )
    ]
    return prompt, examples

def process_report(row, prompt, examples):
    """Processes a single report row to extract findings and impressions."""
    volume_name = row.get('VolumeName', 'UNKNOWN')
    patient_id = '_'.join(str(volume_name).split('_')[:3])
    
    findings_text = str(row.get('Findings_EN', ''))
    impressions_text = str(row.get('Impressions_EN', ''))

    decomposed_data = {"patient_id": patient_id, "findings": {}, "impressions": {}}

    # Process findings
    if findings_text and findings_text != 'nan':
        try:
            result = lx.extract(
                text_or_documents=findings_text,
                prompt_description=prompt,
                examples=examples,
                model_id=MODEL_ID,
            )
            for extraction in result.extractions:
                if extraction.extraction_class in GROUPED_ANATOMIES:
                    decomposed_data["findings"][extraction.extraction_class] = extraction.extraction_text
        except Exception as e:
            print(f"Error processing findings for {patient_id}: {e}")

    # Process impressions
    if impressions_text and impressions_text != 'nan':
        try:
            result = lx.extract(
                text_or_documents=impressions_text,
                prompt_description=prompt,
                examples=examples,
                model_id=MODEL_ID,
            )
            for extraction in result.extractions:
                if extraction.extraction_class in GROUPED_ANATOMIES:
                    decomposed_data["impressions"][extraction.extraction_class] = extraction.extraction_text
        except Exception as e:
            print(f"Error processing impressions for {patient_id}: {e}")

    return decomposed_data

def load_existing_results(output_file):
    """Load existing results if the output file exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                processed_patients = {item.get('patient_id') for item in existing_data if item.get('patient_id')}
                print(f"Loaded {len(existing_data)} existing results. Processed patients: {len(processed_patients)}")
                return existing_data, processed_patients
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading existing results: {e}. Starting fresh.")
            return [], set()
    return [], set()

def save_batch_results(all_results, output_file):
    """Save current results to the output file."""
    try:
        # Save with temporary file to prevent corruption
        temp_file = output_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        os.replace(temp_file, output_file)
        print(f"Saved {len(all_results)} results to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to run the decomposition using langextract with batch processing."""
    input_file = 'data/train_reports.csv'
    output_file = 'output/ct_rate/langextract_decomposition_output.json'

    print(f"Reading input data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    # Load existing results
    all_results, processed_patients = load_existing_results(output_file)
    
    prompt, examples = get_extraction_task()
    
    print(f"Processing {len(df)} reports in batches of {BATCH_SIZE}...")
    print(f"Using model: {MODEL_ID}")
    
    batch_count = 0
    new_results_count = 0
    
    # Process in batches
    for start_idx in range(0, len(df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        batch_count += 1
        
        print(f"\nProcessing batch {batch_count} (rows {start_idx}-{end_idx-1})...")
        
        batch_results = []
        for index, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_count}"):
            volume_name = row.get('VolumeName', 'UNKNOWN')
            patient_id = '_'.join(str(volume_name).split('_')[:3])
            
            # Skip if already processed
            if patient_id in processed_patients:
                print(f"Skipping already processed patient: {patient_id}")
                continue
            
            decomposed_report = process_report(row, prompt, examples)
            batch_results.append(decomposed_report)
            processed_patients.add(patient_id)
            new_results_count += 1
        
        # Add batch results to all results
        all_results.extend(batch_results)
        
        # Save after each batch
        if batch_results:
            save_batch_results(all_results, output_file)
            print(f"Batch {batch_count} complete. Added {len(batch_results)} new results.")
        else:
            print(f"Batch {batch_count} complete. No new results (all patients already processed).")
    
    print(f"\nDecomposition complete!")
    print(f"Total results: {len(all_results)}")
    print(f"New results processed: {new_results_count}")
    print(f"Final output saved to: {output_file}")

if __name__ == '__main__':
    main() 