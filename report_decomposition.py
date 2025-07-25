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
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from datetime import datetime
from config import (
    MODEL_CONFIG,
    PROCESSING_CONFIG,
    LOGGING_CONFIG,
    INPUT_FILES,
    OUTPUT_FILES,
    LOGS_DIR
)

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
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = LOGS_DIR / f"decomposition_{model_name.replace(':', '_')}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
        ]
    )
    
    logger = logging.getLogger('MedicalReportDecomposer')
    logger.info(f"Logging started for model: {model_name}")
    logger.info(f"Log file: {log_filename}")
    
    return logger

class OllamaProvider:
    """LLM provider using Ollama (local LLM)."""
    
    def __init__(self, model_name: str = MODEL_CONFIG["name"], base_url: str = MODEL_CONFIG["base_url"], logger: Optional[logging.Logger] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logger or logging.getLogger('OllamaProvider')
        self.request_count = 0
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=20,    # Increase from default 10
            pool_maxsize=20,        # Match with max_workers
            max_retries=PROCESSING_CONFIG["retry"]["max_attempts"],
            pool_block=False        # Don't block when pool is full
        )
        self._session = requests.Session()
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
        
    def call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Call Ollama API with optimized settings and optional system message."""
        self.request_count += 1
        max_retries = 2
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Zero temperature for maximum consistency
                        "top_p": 0.1,       # Very focused sampling
                        "num_predict": 20000  # Limit response length
                    }
                }
                
                # Add system message if provided
                if system_message:
                    request_data["system"] = system_message
                
                response = self._session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=150  # Increased timeout
                )
                
                response.raise_for_status()
                response_json = response.json()
                response_text = response_json["response"].strip()
                
                return response_text
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Request timed out, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
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
        
        return ""  # Ensure we always return a string

class MedicalReportDecomposer:
    """Decomposes medical reports into structured components using an LLM."""

    def __init__(self, use_ollama: bool = True, ollama_model: str = MODEL_CONFIG["name"], 
                 max_workers: int = MODEL_CONFIG["max_workers"], logger: Optional[logging.Logger] = None):
        """Initializes the decomposer."""
        self.logger = logger or logging.getLogger('MedicalReportDecomposer')
        
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
        self.min_request_interval = 0.5  # Minimum time between requests in seconds
        self.last_request_time = time.time()
        
        self.logger.info(f"Initialized with model: {ollama_model}, workers: {max_workers}")

    def _call_llm_for_json(self, text: str, patient_id: str = "UNKNOWN") -> Dict[str, str]:
        """
        Calls the LLM with a system message and prompt to extract all mentioned anatomies and their
        descriptions in a single JSON response.
        """
        self.extraction_count += 1
        
        anatomy_list = ", ".join(f'"{a}"' for a in GROUPED_ANATOMIES)
        
        system_message = f"""You are a professional radiologist tasked with information extraction.

STRICT OUTPUT REQUIREMENTS:
1. Respond with ONLY a valid JSON object
2. NO text before or after the JSON
3. NO markdown, code blocks, or formatting
4. NO comments or explanations
5. NO thinking blocks or tags

ANATOMY LIST: [{anatomy_list}]

JSON FORMAT RULES:
1. Use exact anatomy names from the list as keys
2. Each value must be a complete description
3. Only include mentioned anatomies
4. Must be valid JSON with proper quotes and braces
5. Keys must match anatomy names exactly

EXAMPLE:
Input CT Report: "Multiple venous collaterals are present in the anterior left chest wall and are associated with the anterior jugular vein at the level of the right sternoclavicular junction. Left subclavian vein collapsed (chronic occlusion pathology?). Trachea, both main bronchi are open. Calcific plaques are observed in the aortic arch. Other mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process. The left kidney partially entering the section is atrophic. The right kidney could not be evaluated because it did not enter the section. Other upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. There are osteophytes with anterior extension in the thoracic vertebrae."

Expected Output:
{{
    "Lung": "Linear atelectasis is present in both lung parenchyma. Subsegmental atelectasis is observed in the right middle lobe. Thickening of the bronchial wall and peribronchial budding tree-like reticulonodular densities are observed in the bilateral lower lobes. Peribronchial minimal consolidation is seen in the lower lobes in places. The findings were evaluated primarily in favor of the infectious process.",
    "Heart": "Heart contour, size are normal. Pericardial effusion-thickening was not observed.",
    "Esophagus": "Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected.",
    "Aorta": "Calcific plaques are observed in the aortic arch. Thoracic aorta diameter is normal."
}}

INVALID OUTPUTS:
❌ Here's the JSON: {{"Heart": "normal"}}
❌ ```{{"Heart": "normal"}}```
❌ Let me analyze... {{"Heart": "normal"}}
❌ {{"heart": "normal"}} (wrong capitalization)
❌ {{"Heart": "normal", "Unknown": "data"}} (invalid anatomy)"""

        prompt = f"""Extract anatomical information from this CT report text:

{text}

Return ONLY the JSON object:"""
        
        response_str = self.llm.call(prompt, system_message=system_message)
        
        # Clean the response to extract only the JSON part
        try:
            # Step 1: Remove any non-JSON content
            def clean_response(text: str) -> str:
                """Extracts the last JSON object from a string that might contain other text."""
                try:
                    # Find the start of the last potential JSON object
                    start = text.rfind('{')
                    if start == -1:
                        return ""
                    
                    # Find the end of the last potential JSON object
                    end = text.rfind('}')
                    if end == -1 or end < start:
                        return ""
                    
                    return text[start : end + 1]
                except ValueError:
                    return ""
            
            # Step 2: Extract and validate JSON
            cleaned_text = clean_response(response_str)
            
            # If we have a valid JSON structure
            if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                try:
                    parsed_json = json.loads(cleaned_text)
                    
                    # Step 3: Validate anatomy names
                    validated_json = {}
                    for key, value in parsed_json.items():
                        # Convert to title case for comparison
                        key_title = key.title()
                        if key_title in GROUPED_ANATOMIES:
                            validated_json[key_title] = value
                    
                    return validated_json
                except json.JSONDecodeError:
                    self.logger.error(f"Patient {patient_id}: Invalid JSON structure after cleaning: {cleaned_text}")
                    return {}
            else:
                self.logger.error(f"Patient {patient_id}: No valid JSON structure found in response: {response_str}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Patient {patient_id}: Failed to parse response: {e}. Response: {response_str}")
            return {}

    def _rate_limited_call(self, text: str, patient_id: str) -> Dict[str, str]:
        """Rate-limited LLM call with validation."""
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            self.last_request_time = time.time()
        
        with self.request_semaphore:
            result = self._call_llm_for_json(text, patient_id)
            
            # Validate result
            if not result and text.strip():
                # Retry once with backoff
                time.sleep(1)
                result = self._call_llm_for_json(text, patient_id)
                
            return result

    def decompose_report(self, row: pd.Series, processing_mode: str = 'all') -> Dict[str, Dict[str, str]]:
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
        if processing_mode in ['all', 'findings'] and findings_text:
            decomposed_data["findings"]["Findings"] = findings_text
            extracted_findings = self._rate_limited_call(findings_text, patient_id)
            
            # Validate findings extraction
            if not extracted_findings and len(findings_text.split()) > 20:
                self.logger.warning(f"Failed to extract findings for {patient_id} with substantial text")
            
            for anatomy, desc in extracted_findings.items():
                anatomy_title = anatomy.title()
                if anatomy_title in GROUPED_ANATOMIES:
                    decomposed_data["findings"][anatomy.lower()] = desc

        # Process impressions
        if processing_mode in ['all', 'impressions'] and impressions_text:
            decomposed_data["impressions"]["Conclusion"] = impressions_text
            extracted_impressions = self._rate_limited_call(impressions_text, patient_id)
            
            # Validate impressions extraction
            if not extracted_impressions and len(impressions_text.split()) > 20:
                self.logger.warning(f"Failed to extract impressions for {patient_id} with substantial text")
            
            for anatomy, desc in extracted_impressions.items():
                anatomy_title = anatomy.title()
                if anatomy_title in GROUPED_ANATOMIES:
                    decomposed_data["impressions"][anatomy.lower()] = desc
        
        # Validate overall extraction
        if findings_text and impressions_text:
            if not decomposed_data["findings"] and not decomposed_data["impressions"]:
                self.logger.error(f"Complete extraction failure for {patient_id}")
            elif bool(decomposed_data["findings"]) != bool(decomposed_data["impressions"]):
                self.logger.warning(f"Partial extraction for {patient_id}: findings={bool(decomposed_data['findings'])}, impressions={bool(decomposed_data['impressions'])}")
        
        return decomposed_data

    def _process_single_report(self, row_data: Tuple, processing_mode: str = 'all') -> Tuple[str, Dict[str, Dict[str, str]]]:
        """Process a single report (for parallel processing)."""
        idx, row = row_data
        volume_name = row['VolumeName']
        patient_id = '_'.join(str(volume_name).split('_')[:3])
        
        start_time = time.time()
        try:
            self.logger.debug(f"Starting processing for patient {patient_id}")
            decomposed_data = self.decompose_report(row, processing_mode=processing_mode)
            processing_time = time.time() - start_time
            
            with self._lock:
                self.processing_times.append(processing_time)
                self.successful_calls += 1
                
                # Log progress every 10 successful calls
                if self.successful_calls % 10 == 0:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    success_rate = (self.successful_calls / (self.successful_calls + self.failed_calls)) * 100 if (self.successful_calls + self.failed_calls) > 0 else 0
                    self.logger.info(
                        f"Progress Update:\n"
                        f"- Successful: {self.successful_calls}\n"
                        f"- Failed: {self.failed_calls}\n"
                        f"- Timeouts: {self.timeout_errors}\n"
                        f"- Success rate: {success_rate:.1f}%\n"
                        f"- Avg processing time: {avg_time:.2f}s"
                    )
            
            self.logger.debug(f"Completed processing for patient {patient_id} in {processing_time:.2f}s")
            return patient_id, decomposed_data
            
        except Exception as e:
            with self._lock:
                self.failed_calls += 1
                if "timeout" in str(e).lower():
                    self.timeout_errors += 1
                    error_type = "Timeout"
                else:
                    error_type = "Error"
            
            self.logger.error(f"{error_type} processing {patient_id}: {str(e)}")
            return patient_id, {"findings": {}, "impressions": {}}

    def process_reports(self, input_data, output_dir: str = "data", sample_size: Optional[int] = None, save_interval: int = 100, findings_path: Optional[Path] = None, impressions_path: Optional[Path] = None, processing_mode: str = 'all'):
        """Process all reports in the CSV file or DataFrame with parallel processing and incremental saving."""
        # Handle input data
        if isinstance(input_data, str):
            self.logger.info(f"Reading input data from CSV file: {input_data}")
            df = pd.read_csv(input_data)
            if sample_size:
                df = df.head(sample_size)
                self.logger.info(f"Using sample size of {sample_size} reports")
        elif isinstance(input_data, pd.DataFrame):
            self.logger.info(f"Using provided DataFrame with {len(input_data)} rows")
            df = input_data
        else:
            error_msg = "input_data must be either a CSV path string or a pandas DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing results if they exist
        final_findings_path = findings_path if findings_path else OUTPUT_FILES["findings"]
        final_impressions_path = impressions_path if impressions_path else OUTPUT_FILES["impressions"]
        
        all_findings = {}
        all_impressions = {}
        processed_patients = set()
        
        if os.path.exists(final_findings_path):
            self.logger.info(f"Loading existing findings from {final_findings_path}")
            with open(final_findings_path, 'r', encoding='utf-8') as f:
                all_findings = json.load(f)
                processed_patients.update(all_findings.keys())
                self.logger.info(f"Loaded {len(all_findings)} existing findings")
        
        if os.path.exists(final_impressions_path):
            self.logger.info(f"Loading existing impressions from {final_impressions_path}")
            with open(final_impressions_path, 'r', encoding='utf-8') as f:
                all_impressions = json.load(f)
                self.logger.info(f"Loaded {len(all_impressions)} existing impressions")
        
        self.start_time = time.time()
        self.logger.info(f"Starting processing of {len(df)} reports with {self.max_workers} workers")
        print(f"Processing {len(df)} reports with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(self._process_single_report, (idx, row), processing_mode=processing_mode): idx 
                for idx, row in df.iterrows()
            }
            self.logger.info(f"Submitted {len(future_to_row)} tasks to thread pool")
            
            # Process results as they complete
            with tqdm(total=len(df)) as pbar:
                completed_since_last_save = 0
                
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
                                
                                completed_since_last_save += 1
                                
                                # Save results periodically
                                if completed_since_last_save >= save_interval:
                                    self.logger.info(f"Saving intermediate results after {completed_since_last_save} new completions")
                                    try:
                                        self._save_results(all_findings, all_impressions, final_findings_path, final_impressions_path, processing_mode=processing_mode)
                                        self.logger.info(f"Successfully saved intermediate results. Total processed: {len(processed_patients)}")
                                    except Exception as save_error:
                                        self.logger.error(f"Failed to save intermediate results: {save_error}")
                                    completed_since_last_save = 0
                        
                    except Exception as e:
                        self.logger.error(f"Error processing future result for patient {patient_id}: {e}")
                        print(f"Error in future result: {e}")
                    
                    pbar.update(1)
        
        # Final save
        self.logger.info("Performing final save of results")
        try:
            self._save_results(all_findings, all_impressions, final_findings_path, final_impressions_path, processing_mode=processing_mode)
            self.logger.info("Final save completed successfully")
        except Exception as final_save_error:
            self.logger.error(f"Failed to perform final save: {final_save_error}")
        
        # Final performance report
        total_time = time.time() - self.start_time
        success_rate = (self.successful_calls / (self.successful_calls + self.failed_calls)) * 100 if (self.successful_calls + self.failed_calls) > 0 else 0
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        performance_summary = f"""
=== Performance Summary ===
Total processing time: {total_time:.2f}s
Average processing time per report: {avg_time:.2f}s
Success rate: {success_rate:.1f}%
Successful calls: {self.successful_calls}
Failed calls: {self.failed_calls}
Timeout errors: {self.timeout_errors}
Processed patients: {len(processed_patients)}
====================="""
        
        self.logger.info(performance_summary)
        print(f"Decomposition complete! Processed {len(processed_patients)} unique patients.")
        self.logger.info(f"Completed processing {len(processed_patients)} patients")

    def _save_results(self, findings: Dict, impressions: Dict, findings_path: Path, impressions_path: Path, processing_mode: str = 'all'):
        """Helper method to save results to files."""
        # Save with temporary files to prevent corruption
        findings_temp = str(findings_path) + '.tmp'
        impressions_temp = str(impressions_path) + '.tmp'
        
        try:
            if processing_mode in ['all', 'findings']:
                with open(findings_temp, 'w', encoding='utf-8') as f:
                    json.dump(findings, f, indent=2, ensure_ascii=False)
                os.replace(findings_temp, findings_path)
            
            if processing_mode in ['all', 'impressions']:
                with open(impressions_temp, 'w', encoding='utf-8') as f:
                    json.dump(impressions, f, indent=2, ensure_ascii=False)
                os.replace(impressions_temp, impressions_path)
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            # Clean up temp files if they exist
            for temp_file in [findings_temp, impressions_temp]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise

def main():
    """Main function to run the decomposition."""
    model_name = MODEL_CONFIG["name"]
    batch_size = MODEL_CONFIG["batch_size"]
    save_interval = MODEL_CONFIG["save_interval"]
    
    # Setup logging
    logger = setup_logging(model_name)
    logger.info(f"Starting decomposition with model: {model_name}, batch_size: {batch_size}")
    
    # Using configured model with conservative parallel processing and logging
    decomposer = MedicalReportDecomposer(
        use_ollama=True, 
        ollama_model=model_name,
        max_workers=MODEL_CONFIG["max_workers"],
        logger=logger
    )
    
    # Read and process the dataset in chunks to manage memory
    try:
        # Use pandas chunk reading for large CSV
        chunk_size = MODEL_CONFIG["chunk_size"]
        for chunk_num, chunk_df in enumerate(pd.read_csv(INPUT_FILES["train"], chunksize=chunk_size)):
            logger.info(f"Processing chunk {chunk_num + 1}")
            
            # Process each chunk in batches
            for i in range(0, len(chunk_df), batch_size):
                batch_num = i // batch_size + 1
                logger.info(f"Processing batch {batch_num} of chunk {chunk_num + 1}")
                
                batch_df = chunk_df.iloc[i:i+batch_size].copy()
                decomposer.process_reports(
                    batch_df,
                    output_dir='data',
                    save_interval=save_interval
                )
                
                # Add cool-down period between batches
                if i + batch_size < len(chunk_df):
                    logger.info("Cooling down between batches...")
                    time.sleep(PROCESSING_CONFIG["cooldown"]["between_batches"])
            
            # Longer cool-down between chunks
            logger.info("Cooling down between chunks...")
            time.sleep(PROCESSING_CONFIG["cooldown"]["between_chunks"])
                
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == '__main__':
    main() 