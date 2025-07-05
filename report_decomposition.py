import pandas as pd
import json
import re
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from datetime import datetime

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

def setup_logging(model_name: str) -> logging.Logger:
    """Setup logging with timestamped filename."""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/decomposition_{model_name.replace(':', '_')}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger('MedicalReportDecomposer')
    logger.info(f"Logging started for model: {model_name}")
    logger.info(f"Log file: {log_filename}")
    
    return logger

class OllamaProvider:
    """LLM provider using Ollama (local LLM)."""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.base_url = base_url
        self._session = requests.Session()  # Reuse connections
        self.logger = logger or logging.getLogger('OllamaProvider')
        self.request_count = 0
        
    def call(self, prompt: str, system_message: Optional[str] = None, enable_thinking: bool = False) -> str:
        """Call Ollama API with optimized settings and optional system message."""
        self.request_count += 1
        
        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Zero temperature for maximum consistency
                    "top_p": 0.1,       # Very focused sampling
                    "num_predict": 2048, # Limit response length
                    "enable_thinking": enable_thinking
                },
                "enable_thinking": enable_thinking
            }
            
            # Add system message if provided
            if system_message:
                request_data["system"] = system_message
            
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=60
            )
            
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json["response"].strip()
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return ""

class MedicalReportDecomposer:
    """Decomposes medical reports into structured components using an LLM."""

    def __init__(self, use_ollama: bool = True, ollama_model: str = "llama3.1", max_workers: int = 4, logger: Optional[logging.Logger] = None):
        """Initializes the decomposer."""
        self.logger = logger or logging.getLogger('MedicalReportDecomposer')
        
        if use_ollama:
            self.llm = OllamaProvider(model_name=ollama_model, logger=self.logger)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.extraction_count = 0
        
        self.logger.info(f"Initialized with model: {ollama_model}, workers: {max_workers}")

    def _call_llm_for_json(self, text: str, patient_id: str = "UNKNOWN") -> Dict[str, str]:
        """
        Calls the LLM with a system message and prompt to extract all mentioned anatomies and their
        descriptions in a single JSON response.
        """
        self.extraction_count += 1
        
        anatomy_list = ", ".join(f'"{a}"' for a in GROUPED_ANATOMIES)
        
        system_message = f"""You are a professional radiologist tasked with information extraction.

Your role is to identify anatomical structures mentioned in CT reports and extract their complete descriptions.

ANATOMIES TO LOOK FOR: [{anatomy_list}]

INSTRUCTIONS:
- Identify every anatomy mentioned that is on the provided list
- Extract the complete description for each mentioned anatomy from the text
- Output MUST be a single, valid JSON object
- Keys should be the anatomy names from the list (exact match)
- Values should be their corresponding extracted descriptions
- If an anatomy from the list is not mentioned in the text, do not include it in the JSON object
- Provide only the JSON object in your response, no additional text, formatting, or thinking blocks
- Do not use <think> tags, explanatory text, or any markdown formatting
- Do not include reasoning, analysis, or commentary
- Return only valid JSON without any prefix or suffix

EXAMPLE:
Input CT Report: "Multiple venous collaterals are present in the anterior left chest wall and are associated with the anterior jugular vein at the level of the right sternoclavicular junction. Left subclavian vein collapsed (chronic occlusion pathology?). Trachea, both main bronchi are open. Calcific plaques are observed in the aortic arch. Other mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process. The left kidney partially entering the section is atrophic. The right kidney could not be evaluated because it did not enter the section. Other upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. There are osteophytes with anterior extension in the thoracic vertebrae."

Expected Output:
{{
    "Findings": "Multiple venous collaterals are present in the anterior left chest wall and are associated with the anterior jugular vein at the level of the right sternoclavicular junction. Left subclavian vein collapsed (chronic occlusion pathology?). Trachea, both main bronchi are open. Calcific plaques are observed in the aortic arch. Other mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process. The left kidney partially entering the section is atrophic. The right kidney could not be evaluated because it did not enter the section. Other upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. There are osteophytes with anterior extension in the thoracic vertebrae.",
    "Lung": "Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process.",
    "Heart": "Heart contour, size are normal. Pericardial effusion-thickening was not observed.",
    "Esophagus": "Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected.",
    "Aorta": "Calcific plaques are observed in the aortic arch. Thoracic aorta diameter is normal."
}}"""

        prompt = f"""Extract anatomical information from this CT report text:

---
{text}
---

Return the JSON object:"""
        
        response_str = self.llm.call(prompt, system_message=system_message, enable_thinking=False)
        
        # Clean the response to extract only the JSON part
        try:
            # Remove thinking blocks if present
            cleaned_response = re.sub(r'<think>.*?</think>', '', response_str, flags=re.DOTALL)
            
            json_str = None
            
            # Try to find JSON in markdown code blocks first
            match = re.search(r'```json\n({.*?})\n```', cleaned_response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Look for JSON object in the cleaned response
                match = re.search(r'\{[^}]*(?:\{[^}]*\}[^}]*)*\}', cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    # Fallback: try to extract anything between first { and last }
                    start = cleaned_response.find('{')
                    end = cleaned_response.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        json_str = cleaned_response[start:end+1]
                    else:
                        self.logger.error(f"Patient {patient_id}: No JSON found in response: {response_str}")
                        return {}
            
            parsed_json = json.loads(json_str)
            return parsed_json
            
        except (json.JSONDecodeError, IndexError) as e:
            self.logger.error(f"Patient {patient_id}: Failed to parse JSON: {e}. Response: {response_str}")
            return {}

    def decompose_report(self, row: pd.Series) -> Dict[str, Dict[str, str]]:
        """Decomposes a single report row into findings and impressions."""
        volume_name = row.get('VolumeName', 'UNKNOWN')
        patient_id = '_'.join(str(volume_name).split('_')[:3])
        
        findings_text = row.get('Findings_EN', '')
        impressions_text = row.get('Impressions_EN', '')

        # Ensure text is a string
        findings_text = str(findings_text) if findings_text is not None and pd.notna(findings_text) else ""
        impressions_text = str(impressions_text) if impressions_text is not None and pd.notna(impressions_text) else ""

        decomposed_data = {"findings": {}, "impressions": {}}

        # Process findings
        if findings_text:
            decomposed_data["findings"]["Findings"] = findings_text
            extracted_findings = self._call_llm_for_json(findings_text, patient_id)
            
            for anatomy, desc in extracted_findings.items():
                # Case-insensitive check for recognized anatomy
                anatomy_title = anatomy.title()  # Convert to title case for comparison
                if anatomy_title in GROUPED_ANATOMIES:
                    decomposed_data["findings"][anatomy.lower()] = desc

        # Process impressions
        if impressions_text:
            decomposed_data["impressions"]["Impressions"] = impressions_text
            extracted_impressions = self._call_llm_for_json(impressions_text, patient_id)
            
            for anatomy, desc in extracted_impressions.items():
                # Case-insensitive check for recognized anatomy
                anatomy_title = anatomy.title()  # Convert to title case for comparison
                if anatomy_title in GROUPED_ANATOMIES:
                    decomposed_data["impressions"][anatomy.lower()] = desc
        
        return decomposed_data

    def _process_single_report(self, row_data: Tuple) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """Process a single report (for parallel processing)."""
        idx, row = row_data
        volume_name = row['VolumeName']
        patient_id = '_'.join(str(volume_name).split('_')[:3])
        
        try:
            decomposed_data = self.decompose_report(row)
            return patient_id, decomposed_data
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            return patient_id, {"findings": {}, "impressions": {}}

    def process_reports(self, csv_path: str, output_dir: str = "data", sample_size: Optional[int] = None):
        """Process all reports in the CSV file with parallel processing."""
        df = pd.read_csv(csv_path)
        
        # Allow configurable sample size
        if sample_size:
            df = df.head(sample_size)
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_findings = {}
        all_impressions = {}
        processed_patients = set()
        
        print(f"Processing {len(df)} reports with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self._process_single_report, (idx, row)): idx 
                for idx, row in df.iterrows()
            }
            
            # Process results as they complete
            with tqdm(total=len(df)) as pbar:
                for future in as_completed(future_to_row):
                    try:
                        patient_id, decomposed_data = future.result()
                        
                        # Thread-safe update of results
                        with self._lock:
                            if patient_id not in processed_patients:
                                if decomposed_data["findings"]:
                                    all_findings[patient_id] = decomposed_data["findings"]
                                if decomposed_data["impressions"]:
                                    all_impressions[patient_id] = decomposed_data["impressions"]
                                processed_patients.add(patient_id)
                        
                    except Exception as e:
                        print(f"Error in future result: {e}")
                    
                    pbar.update(1)
        
        # Save results
        findings_path = os.path.join(output_dir, 'desc_info_manual.json')
        with open(findings_path, 'w', encoding='utf-8') as f:
            json.dump(all_findings, f, indent=2, ensure_ascii=False)
        print(f"Saved findings to {findings_path}")

        impressions_path = os.path.join(output_dir, 'conc_info_manual.json')
        with open(impressions_path, 'w', encoding='utf-8') as f:
            json.dump(all_impressions, f, indent=2, ensure_ascii=False)
        print(f"Saved impressions to {impressions_path}")
        
        print(f"Decomposition complete! Processed {len(processed_patients)} unique patients.")
        self.logger.info(f"Completed processing {len(processed_patients)} patients")

def main():
    """Main function to run the decomposition."""
    model_name = "qwen3:8b"
    sample_size = 10
    
    # Setup logging
    logger = setup_logging(model_name)
    logger.info(f"Starting decomposition with model: {model_name}, sample size: {sample_size}")
    
    # Using qwen3:8b with parallel processing and logging
    decomposer = MedicalReportDecomposer(
        use_ollama=True, 
        ollama_model=model_name,
        max_workers=4,  # Adjust based on your system
        logger=logger
    )
    
    # Start with a small sample to test performance
    decomposer.process_reports('data/train_reports.csv', sample_size=sample_size)

if __name__ == '__main__':
    main() 