import pandas as pd
import json
import re
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import time
import requests
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from datetime import datetime
from collections import defaultdict

# Use the same anatomy mapping from the original script
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

@dataclass
class AnatomyDescription:
    """Represents a description for a specific anatomy."""
    anatomy: str
    description: str
    section: str  # 'findings' or 'impressions'

class OllamaProvider:
    """LLM provider using Ollama (local LLM)."""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logger or logging.getLogger('OllamaProvider')
        self.request_count = 0
        
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=2,
            pool_block=False
        )
        self._session = requests.Session()
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
        
    def call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Call Ollama API with optimized settings and optional system message."""
        self.request_count += 1
        max_retries = 2
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.1,
                        "num_predict": 2048
                    }
                }
                
                if system_message:
                    request_data["system"] = system_message
                
                response = self._session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=90
                )
                
                response.raise_for_status()
                return response.json()["response"].strip()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request timed out, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error("Max retries reached, giving up.")
                    return ""
                    
            except Exception as e:
                self.logger.error(f"API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return ""
        
        return ""

class MedicalReportDecomposerKV:
    """Decomposes medical reports into key-value pairs using an LLM."""

    def __init__(self, use_ollama: bool = True, ollama_model: str = "llama3.1", max_workers: int = 4, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger('MedicalReportDecomposerKV')
        
        if use_ollama:
            self.llm = OllamaProvider(model_name=ollama_model, logger=self.logger)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self.extraction_count = 0
        
        # Performance monitoring
        self.successful_calls = 0
        self.failed_calls = 0
        self.timeout_errors = 0
        self.processing_times = []
        self.start_time = None
        
        # Rate limiting
        self.request_semaphore = threading.Semaphore(max_workers)
        self.min_request_interval = 0.5
        self.last_request_time = time.time()
        
        self.logger.info(f"Initialized with model: {ollama_model}, workers: {max_workers}")

    def _extract_kv_pairs(self, text: str, patient_id: str = "UNKNOWN") -> List[AnatomyDescription]:
        """
        Extracts anatomy descriptions using a key-value pair format instead of JSON.
        Returns a list of AnatomyDescription objects.
        """
        anatomy_list = "\n".join(f"- {a}" for a in GROUPED_ANATOMIES)
        
        system_message = f"""You are a professional radiologist tasked with information extraction.

Your task is to extract anatomical findings from medical reports in a simple key-value pair format.

ANATOMY LIST:
{anatomy_list}

OUTPUT FORMAT:
- Use exactly "ANATOMY:" prefix for each anatomy
- Use exactly "DESCRIPTION:" prefix for each description
- Separate each anatomy-description pair with "---"
- Only include anatomies from the list that are mentioned
- Descriptions should be complete and detailed
- Do not add any other text or formatting

EXAMPLE INPUT:
"Multiple venous collaterals are present in the anterior left chest wall and are associated with the anterior jugular vein at the level of the right sternoclavicular junction. Left subclavian vein collapsed (chronic occlusion pathology?). Trachea, both main bronchi are open. Calcific plaques are observed in the aortic arch. Other mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process. The left kidney partially entering the section is atrophic. The right kidney could not be evaluated because it did not enter the section. Other upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. There are osteophytes with anterior extension in the thoracic vertebrae.",
        
EXAMPLE OUTPUT:
ANATOMY: Heart
DESCRIPTION: Heart contour, size are normal. Pericardial effusion-thickening was not observed.
---
ANATOMY: Aorta
DESCRIPTION: Calcific plaques are observed in the aortic arch. Thoracic aorta diameter is normal.
---
ANATOMY: Esophagus
DESCRIPTION: Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected.
---
ANATOMY: Lung
DESCRIPTION: Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places.
---
ANATOMY: Kidney
DESCRIPTION: The left kidney partially entering the section is atrophic. The right kidney could not be evaluated because it did not enter the section.
---
ANATOMY: Liver
DESCRIPTION: No space-occupying lesion was detected in the liver that entered the cross-sectional area.
---
ANATOMY: Adrenal gland
DESCRIPTION: Bilateral adrenal glands were normal and no space-occupying lesion was detected.
---
ANATOMY: Vertebrae
DESCRIPTION: There are osteophytes with anterior extension in the thoracic vertebrae.
---
ANATOMY: Rib
DESCRIPTION: No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected.


IMPORTANT:
- Only output in the exact format shown
- No additional text or explanations
- No markdown or other formatting
- Only include mentioned anatomies
- Keep descriptions complete and accurate"""

        prompt = f"""Extract anatomical information from this medical report text:

{text}

Output the findings in key-value pairs:"""
        
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        with self.request_semaphore:
            response = self.llm.call(prompt, system_message=system_message)
            
            # If empty response and text exists, retry once
            if not response and text.strip():
                time.sleep(1)
                response = self.llm.call(prompt, system_message=system_message)
        
        # Parse the response into AnatomyDescription objects
        results = []
        current_anatomy = None
        current_description = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line or line == '---':
                if current_anatomy and current_description:
                    results.append(AnatomyDescription(
                        anatomy=current_anatomy,
                        description=current_description,
                        section='findings'  # Will be updated by caller
                    ))
                current_anatomy = None
                current_description = None
            elif line.startswith('ANATOMY:'):
                current_anatomy = line[8:].strip()
            elif line.startswith('DESCRIPTION:'):
                current_description = line[12:].strip()
        
        # Add the last pair if exists
        if current_anatomy and current_description:
            results.append(AnatomyDescription(
                anatomy=current_anatomy,
                description=current_description,
                section='findings'  # Will be updated by caller
            ))
        
        return results

    def decompose_report(self, row: pd.Series) -> Dict[str, Dict[str, str]]:
        """Decomposes a single report row into findings and impressions."""
        volume_name = row.get('VolumeName', 'UNKNOWN')
        patient_id = '_'.join(str(volume_name).split('_')[:3])
        
        findings_text = str(row.Findings_EN) if pd.notnull(row.Findings_EN) else ""
        impressions_text = str(row.Impressions_EN) if pd.notnull(row.Impressions_EN) else ""

        decomposed_data = {"findings": {}, "impressions": {}}

        # Process findings
        if findings_text:
            decomposed_data["findings"]["Findings"] = findings_text
            findings_results = self._extract_kv_pairs(findings_text, patient_id)
            
            for result in findings_results:
                result.section = 'findings'
                if result.anatomy.title() in GROUPED_ANATOMIES:
                    decomposed_data["findings"][result.anatomy.lower()] = result.description

        # Process impressions
        if impressions_text:
            decomposed_data["impressions"]["Conclusion"] = impressions_text
            impressions_results = self._extract_kv_pairs(impressions_text, patient_id)
            
            for result in impressions_results:
                result.section = 'impressions'
                if result.anatomy.title() in GROUPED_ANATOMIES:
                    decomposed_data["impressions"][result.anatomy.lower()] = result.description

        return decomposed_data

    def _process_single_report(self, row_data: Tuple) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """Process a single report (for parallel processing)."""
        idx, row = row_data
        start_time = time.time()
        try:
            decomposed_data = self.decompose_report(row)
            processing_time = time.time() - start_time
            
            with self._lock:
                self.processing_times.append(processing_time)
                self.successful_calls += 1
            
            volume_name = row.get('VolumeName', 'UNKNOWN')
            patient_id = '_'.join(str(volume_name).split('_')[:3])
            return patient_id, decomposed_data
            
        except Exception as e:
            with self._lock:
                self.failed_calls += 1
                if "timeout" in str(e).lower():
                    self.timeout_errors += 1
            
            self.logger.error(f"Error processing report: {e}")
            return "UNKNOWN", {"findings": {}, "impressions": {}}

    def process_reports(self, input_data, output_dir: str = "data"):
        """Process reports from DataFrame or CSV with parallel processing."""
        if isinstance(input_data, str):
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise ValueError("input_data must be either a CSV path string or a pandas DataFrame")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_findings = {}
        all_impressions = {}
        processed_patients = set()
        
        self.start_time = time.time()
        print(f"Processing {len(df)} reports with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_row = {
                executor.submit(self._process_single_report, (idx, row)): idx 
                for idx, row in df.iterrows()
            }
            
            with tqdm(total=len(df)) as pbar:
                for future in as_completed(future_to_row):
                    try:
                        patient_id, decomposed_data = future.result()
                        
                        with self._lock:
                            if patient_id not in processed_patients:
                                if decomposed_data["findings"]:
                                    all_findings[patient_id] = decomposed_data["findings"]
                                if decomposed_data["impressions"]:
                                    all_impressions[patient_id] = decomposed_data["impressions"]
                                processed_patients.add(patient_id)
                        
                    except Exception as e:
                        self.logger.error(f"Error in future result: {e}")
                    
                    pbar.update(1)
        
        # Save results
        findings_path = os.path.join(output_dir, 'desc_info_manual_kv_v1.json')
        with open(findings_path, 'w', encoding='utf-8') as f:
            json.dump(all_findings, f, indent=2, ensure_ascii=False)
        print(f"Saved findings to {findings_path}")

        impressions_path = os.path.join(output_dir, 'conc_info_manual_kv_v1.json')
        with open(impressions_path, 'w', encoding='utf-8') as f:
            json.dump(all_impressions, f, indent=2, ensure_ascii=False)
        print(f"Saved impressions to {impressions_path}")
        
        print(f"Decomposition complete! Processed {len(processed_patients)} unique patients.")

def main():
    """Main function to run the decomposition."""
    model_name = "qwen3:8b"
    batch_size = 40
    
    # Setup logging
    logger = logging.getLogger('MedicalReportDecomposerKV')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    logger.info(f"Starting decomposition with model: {model_name}, batch_size: {batch_size}")
    
    decomposer = MedicalReportDecomposerKV(
        use_ollama=True, 
        ollama_model=model_name,
        max_workers=4,
        logger=logger
    )
    
    try:
        df = pd.read_csv('data/train_reports.csv')
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_df = df.iloc[i:i+batch_size].copy()
            decomposer.process_reports(batch_df, output_dir='data')
            break
            if i + batch_size < len(df):
                logger.info("Cooling down between batches...")
                time.sleep(10)
                
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == '__main__':
    main() 