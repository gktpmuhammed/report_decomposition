import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- MAPPING CONFIGURATION ---
# Key = Column name in Ground Truth
# Value = Column name in Your Predictions
DISEASE_MAPPING = {
    "Hiatal hernia": "esophagus_hiatal_hernia",
    "Cardiomegaly": "heart_cardiomegaly",
    "Pericardial effusion": "heart_pericardial_effusion",
    "Atelectasis": "lung_atelectasis",
    "Emphysema": "lung_emphysema",
    "Pleural effusion": "lung_pleural_effusion",
    "Bronchiectasis": "lung_bronchiectasis"
}

def clean_gt_id(vol_name: str) -> str:
    """
    Converts 'valid_1_a_1.nii.gz' -> 'valid_1_a'
    Adjust this logic if your predicted IDs look different.
    """
    # Remove extension
    base = vol_name.replace('.nii.gz', '').replace('.nii', '')
    
    # Heuristic: split by '_' and assume the last part is the sub-volume number (1, 2, etc.)
    # valid_1_a_1 -> valid_1_a
    # If your predictions use the full name, change this logic.
    parts = base.split('_')
    if len(parts) > 3: 
        # Rejoin everything except the last part
        return "_".join(parts[:-1])
    return base

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_csv', type=str, required=True, help="Your generated disease_labels.csv")
    parser.add_argument('--gt_csv', type=str, required=True, help="The ground truth CSV file")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading files...")
    df_pred = pd.read_csv(args.pred_csv)
    df_gt = pd.read_csv(args.gt_csv)

    # 2. Normalize IDs to ensure they match
    # Predictions usually have 'PatientID' like 'valid_1_a'
    # GT usually has 'VolumeName' like 'valid_1_a_1.nii.gz'
    
    # Ensure ID columns exist
    if 'PatientID' not in df_pred.columns:
        # Fallback if your pred csv doesn't have a header for the index
        df_pred.rename(columns={df_pred.columns[0]: 'PatientID'}, inplace=True)
    
    df_gt['CleanID'] = df_gt['VolumeName'].apply(clean_gt_id)
    df_pred['CleanID'] = df_pred['PatientID'].astype(str)

    # 3. Merge Dataframes on ID
    # We use inner join to only compare patients that exist in BOTH files
    merged = pd.merge(df_gt, df_pred, on='CleanID', suffixes=('_gt', '_pred'))
    
    print(f"\n--- Data Alignment ---")
    print(f"Ground Truth Rows: {len(df_gt)}")
    print(f"Predicted Rows:    {len(df_pred)}")
    print(f"Overlapping IDs:   {len(merged)}")

    if len(merged) == 0:
        print("ERROR: No matching IDs found. Check your ID formatting logic in 'clean_gt_id'.")
        print(f"Example GT ID before clean: {df_gt['VolumeName'].iloc[0]}")
        print(f"Example GT ID after clean:  {df_gt['CleanID'].iloc[0]}")
        print(f"Example Pred ID:            {df_pred['CleanID'].iloc[0]}")
        return

    # 4. Calculate Metrics
    results = []

    for gt_col, pred_col in DISEASE_MAPPING.items():
        if gt_col not in merged.columns or pred_col not in merged.columns:
            print(f"Skipping {gt_col}: Column missing in merged data.")
            continue

        y_true = merged[gt_col]
        y_pred = merged[pred_col]

        # Basic Stats
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Support (Positive cases in GT)
        support = y_true.sum()

        results.append({
            "Disease": gt_col,
            "Accuracy": round(acc, 3),
            "F1 Score": round(f1, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "GT Positives": support,
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    # 5. Print Table
    results_df = pd.DataFrame(results)
    
    # Calculate Macro Averages
    if not results_df.empty:
        print("\n--- Evaluation Results ---")
        # Adjust print width for better readability
        print(results_df.to_string(index=False))
        
        avg_f1 = results_df['F1 Score'].mean()
        avg_acc = results_df['Accuracy'].mean()
        print("-" * 60)
        print(f"MACRO AVERAGE ACCURACY: {avg_acc:.3f}")
        print(f"MACRO AVERAGE F1 SCORE: {avg_f1:.3f}")
        
        # Save detailed report
        output_file = "evaluation_metrics.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed metrics saved to {output_file}")
    else:
        print("No matching columns found to evaluate.")

if __name__ == "__main__":
    main()