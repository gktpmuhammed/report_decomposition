import pandas as pd
import json
import argparse
import os
import logging
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
MODEL_ID = "google/medgemma-4b-it" 
TENSOR_PARALLEL_SIZE = 1 

# --- SCHEMAS (Used to define keys) ---
# We keep these to know what columns to create, but we won't feed the raw complex schema to the LLM anymore.

class EsophagusDiseases(BaseModel):
    hiatal_hernia: int = Field(..., description="Hiatal hernia")
    varicose_veins: int = Field(..., description="Esophageal varices")

class GallbladderDiseases(BaseModel):
    cholecystitis: int = Field(..., description="Cholecystitis")
    gallstone: int = Field(..., description="Gallstones, cholelithiasis")
    adenomyomatosis: int = Field(..., description="Adenomyomatosis")

class HeartDiseases(BaseModel):
    cardiomegaly: int = Field(..., description="Cardiomegaly, enlarged heart")
    pericardial_effusion: int = Field(..., description="Pericardial effusion")

class KidneyDiseases(BaseModel):
    atrophy: int = Field(..., description="Renal atrophy")
    cyst: int = Field(..., description="Renal cyst")
    hydronephrosis: int = Field(..., description="Hydronephrosis")
    calculi: int = Field(..., description="Renal calculi, stones")

class LiverDiseases(BaseModel):
    steatosis: int = Field(..., description="Steatosis, fatty liver")
    glissons_capsule_effusion: int = Field(..., description="Perihepatic fluid/effusion")
    metastasis: int = Field(..., description="Liver metastasis")
    intrahepatic_duct_dilatation: int = Field(..., description="Intrahepatic bile duct dilatation")
    cancer: int = Field(..., description="Primary liver cancer/HCC")
    cyst: int = Field(..., description="Liver cyst")
    abscess: int = Field(..., description="Liver abscess")
    cirrhosis: int = Field(..., description="Cirrhosis")

class LungDiseases(BaseModel):
    atelectasis: int = Field(..., description="Atelectasis")
    bronchiectasis: int = Field(..., description="Bronchiectasis")
    emphysema: int = Field(..., description="Emphysema")
    pneumonia: int = Field(..., description="Pneumonia, consolidation, infiltrate")
    pleural_effusion: int = Field(..., description="Pleural effusion")

class PancreasDiseases(BaseModel):
    pancreatic_cancer: int = Field(..., description="Pancreatic cancer/mass")
    atrophy: int = Field(..., description="Pancreatic atrophy")
    pancreatitis: int = Field(..., description="Pancreatitis")
    duct_dilatation: int = Field(..., description="Pancreatic duct dilatation")
    steatosis: int = Field(..., description="Lipomatosis/steatosis of pancreas")

class SpleenDiseases(BaseModel):
    hemangioma: int = Field(..., description="Hemangioma")
    infarction: int = Field(..., description="Splenic infarction")
    splenomegaly: int = Field(..., description="Splenomegaly")

class StomachDiseases(BaseModel):
    wall_thickening: int = Field(..., description="Gastric wall thickening")
    cancer: int = Field(..., description="Stomach/Gastric cancer")

# Mapping Keys
ORGAN_TO_SCHEMA = {
    "esophagus": EsophagusDiseases,
    "gallbladder": GallbladderDiseases,
    "heart": HeartDiseases,
    "kidney": KidneyDiseases,
    "liver": LiverDiseases,
    "lung": LungDiseases,
    "pancreas": PancreasDiseases,
    "spleen": SpleenDiseases,
    "stomach": StomachDiseases,
}

def load_json_data(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def clean_json_output(text: str):
    """
    Robustly extract JSON.
    """
    text = text.strip()
    
    # 1. Try to find the first {...} block
    # Use non-greedy .*? to avoid capturing multiple JSONs or trailing text
    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except:
            pass # Regex failed, try brute force cleanup below

    # 2. Brute force cleanup
    clean_text = text.replace("```json", "").replace("```", "").strip()
    # Sometimes models output "Output: {...}"
    if "{" in clean_text:
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        clean_text = clean_text[start:end]

    return json.loads(clean_text)

def construct_prompt(organ: str, text: str, schema_model: BaseModel) -> str:
    """
    Simplified Prompt: explicitly lists fields and requests a template fill.
    This prevents the 'Schema Echo' error.
    """
    # Create a list of fields and their descriptions for the prompt
    fields_desc = []
    template_dict = {}
    
    for name, field in schema_model.model_fields.items():
        desc = field.description
        fields_desc.append(f"- {name}: {desc}")
        template_dict[name] = 0 # Example value
        
    fields_block = "\n".join(fields_desc)
    template_str = json.dumps(template_dict, indent=4)

    return (
        f"<start_of_turn>user\n"
        f"You are a medical coding assistant. Analyze the radiology report for the {organ}.\n"
        f"For each pathology below, output 1 if present, 0 if absent/normal.\n\n"
        f"Pathologies to check:\n{fields_desc}\n\n"
        f"Report Text:\n{text}\n\n"
        f"Output ONLY a JSON object exactly like this template (fill with 0 or 1):\n"
        f"{template_str}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--findings_json', type=str, required=True)
    parser.add_argument('--impressions_json', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='disease_labels.csv')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading decomposed data...")
    findings_data = load_json_data(args.findings_json)
    impressions_data = load_json_data(args.impressions_json)

    all_pids = sorted(list(set(findings_data.keys()) | set(impressions_data.keys())))
    logging.info(f"Found {len(all_pids)} unique patients.")

    prompts = []
    metadata = []
    final_results = {pid: {} for pid in all_pids}

    # Prepare Prompt Batch
    for pid in all_pids:
        f_organs = findings_data.get(pid, {})
        i_organs = impressions_data.get(pid, {})
        
        for organ, schema_model in ORGAN_TO_SCHEMA.items():
            f_text = f_organs.get(organ)
            i_text = i_organs.get(organ)
            
            combined_text = ""
            if f_text: combined_text += f"{f_text} "
            if i_text: combined_text += f"{i_text}"
            combined_text = combined_text.strip()

            # If no text, default to 0
            if not combined_text:
                defaults = {f"{organ}_{field}": 0 for field in schema_model.model_fields.keys()}
                final_results[pid].update(defaults)
            else:
                p = construct_prompt(organ, combined_text, schema_model)
                prompts.append(p)
                metadata.append((pid, organ))

    logging.info(f"Total queries: {len(prompts)}")

    if prompts:
        logging.info(f"Initializing {MODEL_ID}...")
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="bfloat16"
        )

        # Batch by organ for cleaner logging/processing (optional but good practice)
        organ_groups = {org: [] for org in ORGAN_TO_SCHEMA.keys()}
        for i, (pid, organ) in enumerate(metadata):
            organ_groups[organ].append({
                "prompt": prompts[i],
                "pid": pid
            })

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["<end_of_turn>"]
        )

        for organ, items in organ_groups.items():
            if not items: continue
            
            logging.info(f"Processing {organ} ({len(items)} samples)...")
            batch_prompts = [x["prompt"] for x in items]
            
            outputs = llm.generate(batch_prompts, sampling_params)
            
            for item, output in zip(items, outputs):
                pid = item["pid"]
                generated_text = output.outputs[0].text
                
                try:
                    data = clean_json_output(generated_text)
                    
                    # Validation: Check if it's a hallucinated schema
                    if "properties" in data or "type" in data:
                        raise ValueError("Model returned a schema definition, not data.")

                    # Flatten keys
                    flat_data = {f"{organ}_{k}": int(v) for k, v in data.items() if k in ORGAN_TO_SCHEMA[organ].model_fields}
                    final_results[pid].update(flat_data)
                except Exception as e:
                    # Logging the first few chars helps debug
                    logging.error(f"Error {pid} {organ}: {e} | Output start: {generated_text[:50]}")
                    
                    # Fallback to zeros
                    defaults = {f"{organ}_{field}": 0 for field in ORGAN_TO_SCHEMA[organ].model_fields.keys()}
                    final_results[pid].update(defaults)

    logging.info("Saving CSV...")
    df = pd.DataFrame.from_dict(final_results, orient='index')
    df.index.name = 'PatientID'
    df.reset_index(inplace=True)
    
    # Fill any missing NaNs with 0 (in case some keys were missed in partial failures)
    df = df.fillna(0)

    # Sort columns
    cols = ['PatientID'] + sorted([c for c in df.columns if c != 'PatientID'])
    df = df[cols]
    
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Saved to {args.output_csv}")

if __name__ == "__main__":
    main()