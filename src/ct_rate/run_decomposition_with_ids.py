import pandas as pd
from pathlib import Path
from report_decomposition import MedicalReportDecomposer, setup_logging
from config import MODEL_CONFIG

def load_patient_ids(file_path):
    """Loads patient IDs from a text file."""
    with open(file_path, 'r') as f:
        return {line.strip() for line in f}

def main():
    """Main function to run the decomposition on a filtered set of reports."""
    model_name = MODEL_CONFIG["name"]
    save_interval = MODEL_CONFIG["save_interval"]

    # Setup logging
    logger = setup_logging(f"{model_name}_filtered")
    logger.info("Starting decomposition with filtered patient IDs.")

    # Load the patient IDs
    conc_patient_ids = load_patient_ids('output/ct_rate/patients_with_conclusion.txt')
    desc_patient_ids = load_patient_ids('output/ct_rate/patients_with_findings.txt')
    logger.info(f"Loaded {len(conc_patient_ids)} conclusion patient IDs and {len(desc_patient_ids)} findings patient IDs.")

    # Load the main dataset
    input_csv = "data/train_reports.csv"
    logger.info(f"Loading main dataset from {input_csv}")
    df = pd.read_csv(input_csv)

    # Filter the DataFrame
    df['patient_id'] = df['VolumeName'].str.extract(r'^(train_\d+_[a-zA-Z])')
    
    # Filter for findings
    desc_patient_df = df[df['patient_id'].isin(desc_patient_ids)].copy()
    desc_patient_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    logger.info(f"Filtered dataset to {len(desc_patient_df)} reports for findings processing.")

    # Filter for conclusions
    conc_patient_df = df[df['patient_id'].isin(conc_patient_ids)].copy()
    conc_patient_df.drop_duplicates(subset=['patient_id'], keep='first', inplace=True)
    logger.info(f"Filtered dataset to {len(conc_patient_df)} reports for conclusion processing.")

    # Initialize decomposer
    decomposer = MedicalReportDecomposer(
        use_ollama=True,
        ollama_model=model_name,
        max_workers=MODEL_CONFIG["max_workers"],
        logger=logger
    )

    # Define new output files
    output_dir = Path('output/ct_rate')
    findings_output_path = output_dir / "decomposed_desc_missing.json"
    impressions_output_path = output_dir / "decomposed_conc_missing.json"

    # Process reports for findings
    logger.info("Processing reports for findings.")
    decomposer.process_reports(
        desc_patient_df,
        output_dir='output/ct_rate',
        save_interval=save_interval,
        findings_path=findings_output_path,
        impressions_path=impressions_output_path,
        processing_mode='findings'
    )
    logger.info("Findings processing complete.")

    # Process reports for conclusions
    logger.info("Processing reports for conclusions.")
    decomposer.process_reports(
        conc_patient_df,
        output_dir='output/ct_rate',
        save_interval=save_interval,
        findings_path=findings_output_path,
        impressions_path=impressions_output_path,
        processing_mode='impressions'
    )
    logger.info("Conclusions processing complete.")


if __name__ == '__main__':
    main() 