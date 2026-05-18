# Radiology Report Decomposition with MedGemma and vLLM

Structured extraction pipeline for decomposing CT radiology reports into organ-specific findings and impressions.

This project uses a schema-guided LLM prompt to turn free-text reports into machine-readable JSON. It was built as a supporting tool for medical vision-language model experiments, where report text needs to be aligned with anatomical regions and downstream evaluation pipelines.

## What It Does

Given a CSV of CT reports, the pipeline:

- Reads the `Findings_EN` and `Impressions_EN` sections.
- Prompts a medical instruction model with a Pydantic-generated JSON schema.
- Separates findings from impressions.
- Maps each section to anatomical keys such as `lung`, `heart`, `aorta`, `liver`, `kidney`, `gallbladder`, and `bladder`.
- Runs batched inference through vLLM.
- Saves clean JSON files for the train and validation splits.

The goal is not to summarize reports. The goal is to preserve clinically relevant organ-specific statements in a structured format.

## Why It Matters

Radiology reports are dense, variable, and often mix multiple organs in one paragraph. For medical AI workflows, that makes it difficult to:

- build organ-level supervision,
- evaluate generated reports by anatomical region,
- inspect model failures,
- connect visual tokens or attention maps with report content.

This repository demonstrates a practical LLM-based preprocessing step for that problem.

## Pipeline

```text
CSV reports
  -> Findings_EN / Impressions_EN extraction
  -> Pydantic anatomy schema
  -> MedGemma prompt
  -> vLLM batched generation
  -> robust JSON parsing
  -> train_findings.json / train_impressions.json
  -> val_findings.json / val_impressions.json
```

## Repository Structure

```text
.
├── data/
│   ├── train_reports.csv
│   ├── reports_english.csv
│   ├── reports_english.json
│   ├── reports_german.json
│   ├── conc_info.json
│   └── desc_info.json
└── src/
    └── ct_rate/
        └── report_decomposition_vllm.py
```

## Input Format

The main script expects CSV files with these columns:

```text
VolumeName,ClinicalInformation_EN,Technique_EN,Findings_EN,Impressions_EN
```

`VolumeName` is converted into a patient or volume identifier. For example:

```text
train_1_a_1.nii.gz -> train_1_a
```

## Output Format

The script writes separate JSON files for findings and impressions:

```text
output/ct_rate/train_findings.json
output/ct_rate/train_impressions.json
output/ct_rate/val_findings.json
output/ct_rate/val_impressions.json
```

Example shape:

```json
{
  "train_1_a": {
    "lung": "Linear atelectasis is present in both lung parenchyma...",
    "aorta": "Calcific plaques are observed in the aortic arch.",
    "kidney": "The left kidney partially entering the section is atrophic."
  }
}
```

Only mentioned organs are kept in the final output. Missing organs are omitted after parsing.

## Model and Runtime

Default model:

```text
google/medgemma-4b-it
```

The model can be changed through environment variables:

```bash
export REPORT_DECOMP_MODEL_ID="google/medgemma-4b-it"
export REPORT_DECOMP_TP_SIZE=1
```

`REPORT_DECOMP_TP_SIZE` controls vLLM tensor parallelism.

## Setup

Create an environment with Python 3.10 or newer, then install the required packages:

```bash
pip install pandas pydantic vllm
```

For GPU inference, install a vLLM build compatible with your CUDA and PyTorch versions.

## Usage

Run a small smoke test first:

```bash
PYTHONPATH=src python src/ct_rate/report_decomposition_vllm.py \
  --train_csv data/train_reports.csv \
  --val_csv data/reports_english.csv \
  --output_dir output/ct_rate \
  --sample 5
```

Run the full decomposition:

```bash
PYTHONPATH=src python src/ct_rate/report_decomposition_vllm.py \
  --train_csv data/train_reports.csv \
  --val_csv data/reports_english.csv \
  --output_dir output/ct_rate
```

## Implementation Notes

- The anatomy schema is defined with Pydantic models.
- The JSON schema is injected directly into the system prompt.
- vLLM handles batched generation for higher throughput.
- `temperature=0.0` is used for deterministic extraction.
- The parser handles direct JSON, fenced JSON blocks, and JSON embedded in surrounding text.

## What to Notice

This is a compact applied-LLM pipeline rather than a notebook experiment. The important engineering pieces are:

- schema-grounded prompting,
- structured output design,
- batched local inference,
- robust output parsing,
- reusable train and validation split processing,
- medically meaningful organ-level decomposition.

## Limitations

- The current script validates parseability but does not yet enforce the Pydantic schema after generation.
- There is no extraction quality benchmark in this repository yet.
- Output quality depends on the selected instruction model.
- This project is for research and engineering demonstration only, not clinical use.

## Next Improvements

- Add schema validation and automatic retry for invalid generations.
- Add a small labeled evaluation set.
- Report organ-level precision, recall, and exact-match metrics.
- Add Docker or a reproducible environment file for GPU servers.
