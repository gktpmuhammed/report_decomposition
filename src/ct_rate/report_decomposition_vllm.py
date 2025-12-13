import pandas as pd
import json
import argparse
import os
import logging
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
# Use the official Meta weights or the AWQ version. 
# With 4x A100s, you can likely run the full unquantized BF16 model for max accuracy.
# If OOM, switch to "kosbu/Llama-3.3-70B-Instruct-AWQ"
MODEL_ID = "kosbu/Llama-3.3-70B-Instruct-AWQ" 
TENSOR_PARALLEL_SIZE = 2  

# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# TENSOR_PARALLEL_SIZE = 1

# --- 1. THE SCHEMA ---
# We define a sub-model for the organ mapping
class AnatomyMap(BaseModel):
    # Thorax
    lung: str | None = Field(None, description="Lungs, parenchyma, nodules, opacities, pleura.")
    heart: str | None = Field(None, description="Heart size, pericardium, chambers.")
    aorta: str | None = Field(None, description="Aorta, arch, calcifications.")
    pulmonary_vein: str | None = Field(None, description="Pulmonary veins, vasculature.")
    vena_cava: str | None = Field(None, description="SVC, IVC.")
    trachea: str | None = Field(None, description="Trachea, main bronchi.")
    esophagus: str | None = Field(None, description="Esophagus, hiatal hernia.")
    thyroid: str | None = Field(None, description="Thyroid gland.")
    
    # Bones
    spine: str | None = Field(None, description="Vertebrae, degenerative changes, spondylosis.")
    rib: str | None = Field(None, description="Ribs, fractures.")
    sternum: str | None = Field(None, description="Sternum.")
    clavicula: str | None = Field(None, description="Clavicle.")
    scapula: str | None = Field(None, description="Scapula.")
    humerus: str | None = Field(None, description="Humerus, shoulder joint.")
    
    # Abdomen
    liver: str | None = Field(None, description="Liver parenchyma, steatosis.")
    gallbladder: str | None = Field(None, description="Gallbladder, stones.")
    stomach: str | None = Field(None, description="Stomach.")
    pancreas: str | None = Field(None, description="Pancreas.")
    spleen: str | None = Field(None, description="Spleen.")
    kidney: str | None = Field(None, description="Kidneys, cysts, stones.")
    adrenal: str | None = Field(None, description="Adrenal glands.")
    colon: str | None = Field(None, description="Colon, large intestine.")
    small_bowel: str | None = Field(None, description="Small intestine, duodenum.")
    
    # Muscles
    iliopsoas: str | None = Field(None, description="Psoas muscles.")
    autochthon: str | None = Field(None, description="Paraspinal muscles.")

# This is the master schema that separates Findings and Impressions
class MasterDecomposition(BaseModel):
    findings_anatomy: AnatomyMap
    impressions_anatomy: AnatomyMap

# --- HELPERS ---
def clean_patient_id(vol_name: str) -> str:
    """train_1_a_1.nii.gz -> train_1_a"""
    base = vol_name.replace('.nii.gz', '').replace('.nii', '')
    parts = base.split('_')
    # Take first 3 parts (e.g., train, 1, a)
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return base

def parse_json_output(text: str):
    """Robustly find and parse JSON from LLM output (handles markdown)."""
    try:
        # Attempt direct parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Try stripping markdown blocks ```json ... ```
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Last resort: Regex search for { ... }
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./decomposed_data')
    parser.add_argument('--sample', type=int, default=None, help="Process only N samples per file for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    # Initialize vLLM
    logging.info(f"Initializing vLLM on {TENSOR_PARALLEL_SIZE} GPUs...")
    llm = LLM(
        model=MODEL_ID, 
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True
    )

    # 1. Get JSON Schema string for the prompt
    schema_json = json.dumps(MasterDecomposition.model_json_schema(), indent=2)

    # 2. Define Sampling Params (REMOVED guided_json to fix version error)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        stop=["<|im_end|>", "<|eot_id|>"] # Stop tokens
    )

    # Process each CSV
    for csv_path in [args.train_csv, args.val_csv]:
        split_name = "train" if "train" in csv_path.lower() else "val"
        logging.info(f"Processing {split_name} split...")
        
        df = pd.read_csv(csv_path)
        
        # --- SAMPLE LOGIC ---
        if args.sample:
            logging.info(f"Subsampling first {args.sample} rows...")
            df = df.head(args.sample)
        # --------------------
        
        prompts = []
        pids = []
        
        for _, row in df.iterrows():
            f_text = str(row.get('Findings_EN', ''))
            i_text = str(row.get('Impressions_EN', ''))
            pid = clean_patient_id(str(row.get('VolumeName', '')))
            
            if len(f_text) < 5 and len(i_text) < 5: continue

            # Updated prompt with explicit schema instruction
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a radiologist. Decompose the report into organ-specific findings based on this schema:\n"
                f"{schema_json}\n"
                f"Separate the extraction for 'findings_anatomy' and 'impressions_anatomy'. "
                f"Use null if an organ is not mentioned. Do not summarize. Return valid JSON only.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"FINDINGS SECTION:\n{f_text}\n\n"
                f"IMPRESSION SECTION:\n{i_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prompts.append(prompt)
            pids.append(pid)

        # Batch Generate
        outputs = llm.generate(prompts, sampling_params)

        # Result Accumulators
        findings_json = {}
        impressions_json = {}

        for pid, output in zip(pids, outputs):
            generated_text = output.outputs[0].text
            data = parse_json_output(generated_text)
            
            if data:
                try:
                    # Findings (Extract sub-dict and filter nulls)
                    f_data = {k: v for k, v in data.get('findings_anatomy', {}).items() if v}
                    findings_json[pid] = f_data
                    
                    # Impressions
                    i_data = {k: v for k, v in data.get('impressions_anatomy', {}).items() if v}
                    impressions_json[pid] = i_data
                except Exception as e:
                    logging.error(f"Error processing keys for {pid}: {e}")
            else:
                logging.error(f"Failed to parse JSON for {pid}")

        # Save separate files per split
        with open(os.path.join(args.output_dir, f"{split_name}_findings.json"), 'w') as f:
            json.dump(findings_json, f, indent=2)
        with open(os.path.join(args.output_dir, f"{split_name}_impressions.json"), 'w') as f:
            json.dump(impressions_json, f, indent=2)

    logging.info("Decomposition complete.")

if __name__ == "__main__":
    main()