import json
import requests
import time
import pandas as pd
import re
from typing import Dict, List, Any

def clean_text(text: str) -> str:
    """
    Clean text by removing all escape characters and converting to single line
    """
    if not text:
        return ""
    
    # Replace literal escape sequences
    text = text.replace('\\r', ' ')
    text = text.replace('\\n', ' ')
    text = text.replace('\\f', ' ')
    text = text.replace('\\t', ' ')
    
    # Replace actual control characters
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\f', ' ')
    text = text.replace('\t', ' ')
    
    # Remove any other control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    
    # Normalize spaces
    text = ' '.join(text.split())
    
    return text.strip()

def translate_text(text: str) -> str:
    """
    Translate text from German to English using Ollama's Mistral model
    """
    if not text or text.strip() == "":
        return ""
    
    # Text is already cleaned when reading from JSON
    prompt = f"""Translate the following German medical report text to English. 
    Maintain medical terminology and professional tone. 
    Provide a concise single-line translation without any line breaks.
    
    German text:
    {text}
    
    English translation:"""
    
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               "model": "mistral",
                               "prompt": prompt,
                               "stream": False
                           })
    
    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        print(f"Error translating text: {response.status_code}")
        return text

def clean_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean all text fields in the report
    """
    cleaned = {}
    for key, value in report.items():
        if isinstance(value, str):
            cleaned[key] = clean_text(value)
        else:
            cleaned[key] = value
    return cleaned

def translate_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate specific fields of a medical report from German to English
    """
    # Fields that need translation
    fields_to_translate = [
        'station', 'untersuchung', 'klinische_angaben', 
        'fragestellung', 'voruntersuchung', 'befund', 'beurteilung'
    ]
    
    translated_report = report.copy()
    
    for field in fields_to_translate:
        if field in report and report[field]:
            print(f"Translating {field}...")
            translated_report[field] = translate_text(report[field])
            # Add a small delay to avoid overwhelming the API
            time.sleep(1)
    
    return translated_report

def format_volume_name(patid: str, accnr: str, date: str) -> str:
    """
    Format the volume name to match the expected format: patid_accnr_date
    """
    return f"{patid}_{accnr}_{date}"

def convert_to_csv(reports: List[Dict[str, Any]]) -> None:
    """
    Convert the translated reports to CSV format matching train_reports.csv structure
    """
    # Convert to DataFrame
    df = pd.DataFrame(reports)
    
    # Create VolumeName from patid, accnr, and date
    df['VolumeName'] = df.apply(lambda row: format_volume_name(
        row['patid'], '1', '1'
    ), axis=1)
    
    # Rename columns to match train_reports.csv format
    column_mapping = {
        'befund': 'Findings_EN',
        'beurteilung': 'Impressions_EN'
    }
    
    # Rename only the columns that exist
    existing_columns = set(df.columns) & set(column_mapping.keys())
    rename_dict = {col: column_mapping[col] for col in existing_columns}
    df = df.rename(columns=rename_dict)
    
    # Select and reorder columns
    columns_to_keep = ['VolumeName', 'Findings_EN', 'Impressions_EN']
    df = df[columns_to_keep]
    
    # Save to CSV
    output_file = 'data/reports_english.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nConverted to CSV: {output_file}")
    
    # Display the first few rows and column names
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

def main():
    # Read the German reports
    with open('data/reports_german.json', 'r', encoding='utf-8') as f:
        reports = json.load(f)
    
    # Clean all reports before translation
    cleaned_reports = [clean_report(report) for report in reports]
    
    # Translate each report
    translated_reports = []
    for i, report in enumerate(cleaned_reports, 1):
        print(f"\nTranslating report {i} of {len(reports)}...")
        translated_report = translate_report(report)
        translated_reports.append(translated_report)
    
    # Save the translated reports as JSON
    with open('data/reports_english.json', 'w', encoding='utf-8') as f:
        json.dump(translated_reports, f, ensure_ascii=False, indent=2)
    
    print("\nTranslation completed! Translated reports saved to 'data/reports_english.json'")
    
    # Convert to CSV format
    print("\nConverting to CSV format...")
    convert_to_csv(translated_reports)

if __name__ == "__main__":
    main()
