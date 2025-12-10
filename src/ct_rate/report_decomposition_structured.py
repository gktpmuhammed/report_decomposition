import pandas as pd
import json
import os
import time
import logging
import argparse
import requests
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter

# --- CONFIGURATION ---
# Use 70b if you have the VRAM (48GB+), otherwise 8b is fine but strict prompting is harder.
MODEL_NAME = "llama3.1:8b" 
OLLAMA_URL = "http://localhost:11434"
MAX_WORKERS = 4  # Increase this if you have multiple GPUs serving Ollama
MAX_RETRIES = 3
TIMEOUT = 400 

# --- 1. DEFINE EXACT SCHEMA ---
# This matches your ALL_TARGET_KEYS list in train.py
class OrganReport(BaseModel):
    Conclusion: Optional[str] = Field(description="Full text of the Impression/Conclusion section.")
    
    # Thorax
    Lung: Optional[str] = Field(description="Lungs, lobes, parenchyma, nodules, opacities, atelectasis.")
    Heart: Optional[str] = Field(description="Heart size, cardiomegaly, pericardium, chambers.")
    Aorta: Optional[str] = Field(description="Aorta, arch, atherosclerosis, calcifications.")
    Pulmonary_Vein: Optional[str] = Field(description="Pulmonary veins, pulmonary vasculature.")
    Vena_Cava: Optional[str] = Field(description="SVC, IVC, superior/inferior vena cava.")
    Trachea: Optional[str] = Field(description="Trachea, main bronchi, carina.")
    Esophagus: Optional[str] = Field(description="Esophagus, hiatal hernia.")
    Thyroid: Optional[str] = Field(description="Thyroid gland, nodules, goiter.")
    
    # Bones
    Spine: Optional[str] = Field(description="Vertebrae (Cervical/Thoracic/Lumbar), degenerative changes, spondylosis.")
    Rib: Optional[str] = Field(description="Ribs, fractures, cage.")
    Sternum: Optional[str] = Field(description="Sternum, xiphoid.")
    Clavicula: Optional[str] = Field(description="Clavicle.")
    Scapula: Optional[str] = Field(description="Scapula, shoulder blade.")
    Humerus: Optional[str] = Field(description="Humerus head, shoulder joint.")
    
    # Abdomen
    Liver: Optional[str] = Field(description="Liver parenchyma, cysts, steatosis.")
    Gallbladder: Optional[str] = Field(description="Gallbladder, stones, sludge, cholecystectomy.")
    Stomach: Optional[str] = Field(description="Stomach, gastric wall.")
    Pancreas: Optional[str] = Field(description="Pancreas, duct.")
    Spleen: Optional[str] = Field(description="Spleen, splenomegaly.")
    Kidney: Optional[str] = Field(description="Kidneys, renal cysts, stones, hydronephrosis.")
    Adrenal: Optional[str] = Field(description="Adrenal glands, nodules.")
    Colon: Optional[str] = Field(description="Colon, large intestine, sigmoid, rectum.")
    Small_Bowel: Optional[str] = Field(description="Small intestine, duodenum, ileum, bowel loops.")
    
    # Muscles
    Iliopsoas: Optional[str] = Field(description="Psoas muscles.")
    Autochthon: Optional[str] = Field(description="Paraspinal muscles, back muscles.")

# --- 2. OLLAMA CLIENT ---
class OllamaClient:
    def __init__(self, base_url, model):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        # Allow more connections for parallel processing
        adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
        self.session.mount('http://', adapter)

    def generate_structured(self, text: str) -> Dict[str, Any]:
        """
        Forces the LLM to output JSON matching the OrganReport schema.
        """
        schema_json = OrganReport.model_json_schema()
        
        # Strict System Prompt
        system_prompt = f"""You are a strict data extraction engine for radiology reports.
        
        RULES:
        1. Extract findings into the exact JSON structure provided.
        2. Do NOT invent new keys (like 'Mediastinum' or 'Bone'). Use only the keys in the schema.
        3. If an organ is NOT mentioned in the text, set the value to null.
        4. Copy the text EXACTLY from the report. Do not summarize.
        5. 'Conclusion' key must contain the full Impression section.
        6. If a sentence mentions multiple organs (e.g. "Heart and lungs are normal"), include that sentence in BOTH the 'Heart' and 'Lung' fields.
        
        SCHEMA:
        {json.dumps(schema_json, indent=2)}
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "format": "json",  # Force JSON output
            "stream": False,
            "options": {
                "temperature": 0.0, # Deterministic (Critical for extraction)
                "num_ctx": 8192     # Ensure context is large enough
            }
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/chat", 
                    json=payload, 
                    timeout=TIMEOUT
                )
                response.raise_for_status()
                result = response.json()
                
                content = result['message']['content']
                data = json.loads(content)
                
                # Cleanup: Convert None to empty string or drop
                # And lowercase keys to match your training script expectations
                clean_data = {}
                for k, v in data.items():
                    if v and isinstance(v, str) and v.lower() != "null":
                        clean_data[k.lower()] = v.strip()
                
                return clean_data

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logging.error(f"Failed to process report: {e}")
                    return {}
                time.sleep(2)
        return {}

# --- 3. PROCESSING PIPELINE ---
def get_column_data(row, possible_names):
    """Helper to find data regardless of CSV column naming"""
    for name in possible_names:
        if name in row and pd.notna(row[name]):
            return str(row[name])
    return ""

def process_row(args):
    idx, row, client = args
    
    # Flexible column handling
    findings = get_column_data(row, ['findings', 'Findings_EN', 'Findings'])
    impression = get_column_data(row, ['impressions', 'Impressions_EN', 'Impressions', 'Conclusion'])
    
    # Skip if report is essentially empty
    if len(findings) < 5 and len(impression) < 5:
        return None, None

    full_text = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"
    
    # ID Handling
    # Try finding 'scan_id', 'VolumeName', 'patient_id'
    pid = get_column_data(row, ['scan_id'])

    # Call LLM
    extracted_data = client.generate_structured(full_text)
    
    if not extracted_data:
        return pid, None
        
    return pid, extracted_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_json', type=str, default='desc_info_structured.json')
    parser.add_argument('--sample', type=int, default=None, help="Test on N rows")
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("decomposition.log"), logging.StreamHandler()]
    )

    # Load Data
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        logging.error(f"Could not read CSV: {e}")
        return

    if args.sample:
        df = df.head(args.sample)
    
    logging.info(f"Loaded {len(df)} reports. Starting decomposition with {MODEL_NAME}...")

    # Initialize Client
    client = OllamaClient(OLLAMA_URL, MODEL_NAME)
    
    results = {}
    
    # Process
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, (idx, row, client)) for idx, row in df.iterrows()]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                pid, data = future.result()
                if pid and data:
                    results[pid] = data
                    
                    # Periodic Save (Every 50)
                    if len(results) % 50 == 0:
                        with open(args.output_json, 'w') as f:
                            json.dump(results, f, indent=2)
            except Exception as e:
                logging.error(f"Worker error: {e}")

    # Final Save
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Done. Saved {len(results)} structured reports to {args.output_json}")

if __name__ == "__main__":
    main()