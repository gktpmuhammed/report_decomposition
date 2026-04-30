# Smoke Tests

```bash
conda activate ct-rate
python -c "from vllm import LLM; import pandas, pydantic"
python src/ct_rate/report_decomposition_vllm.py \
  --train_csv /path/to/train_reports.csv \
  --val_csv /path/to/validation_reports.csv \
  --output_dir output/ct_rate \
  --sample 2
```
