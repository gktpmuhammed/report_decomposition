import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the files
def load_json(filename):
    with open(filename) as f:
        return json.load(f)

# Load all data
ground_truth_conc = load_json('data/conc_info.json')
ground_truth_desc = load_json('data/desc_info.json')
generated_conc = load_json('output/ct_rate/conc_info_manual_merged.json')
generated_desc = load_json('output/ct_rate/desc_info_manual_merged.json')

# Initialize counters
desc_organ_matches = defaultdict(lambda: {'correct': 0, 'total': 0})
conc_organ_matches = defaultdict(lambda: {'correct': 0, 'total': 0})
# New counters for reverse direction
desc_organ_matches_reverse = defaultdict(lambda: {'correct': 0, 'total': 0})
conc_organ_matches_reverse = defaultdict(lambda: {'correct': 0, 'total': 0})

# Specific organs to focus on
target_organs = ['heart', 'aorta', 'lung', 'esophagus']

report_type_counts = {
    'ground_truth': {
        'impression_only': {'count': 0, 'total': 0},
        'findings_only': {'count': 0, 'total': 0}
    },
    'generated': {
        'impression_only': {'count': 0, 'total': 0},
        'findings_only': {'count': 0, 'total': 0}
    },
    'matching': {
        'impression_only': 0,
        'findings_only': 0
    }
}

# Compare organ findings in descriptions - ground truth in generated
for report_id in generated_desc:
    if report_id in ground_truth_desc:  # Check if this report exists in ground truth
        gt_organs = set(ground_truth_desc[report_id].keys())
        gen_organs = set(generated_desc[report_id].keys())
        
        # Check each organ in ground truth
        for organ in gt_organs:
            if organ != "Findings" and organ in target_organs:  # Only check target organs
                desc_organ_matches[organ]['total'] += 1
                if organ in gen_organs:
                    desc_organ_matches[organ]['correct'] += 1

        # Check each organ in generated (reverse direction)
        for organ in gen_organs:
            if organ != "Findings" and organ in target_organs:  # Only check target organs
                desc_organ_matches_reverse[organ]['total'] += 1
                if organ in gt_organs:
                    desc_organ_matches_reverse[organ]['correct'] += 1

# Compare organ findings in conclusions - ground truth in generated
for report_id in generated_conc:
    if report_id in ground_truth_conc:  # Check if this report exists in ground truth
        gt_organs = set(ground_truth_conc[report_id].keys())
        gen_organs = set(generated_conc[report_id].keys())
        
        # Check each organ in ground truth
        for organ in gt_organs:
            if organ not in ["Impressions", "Conclusion"] and organ in target_organs:  # Only check target organs
                conc_organ_matches[organ]['total'] += 1
                if organ in gen_organs:
                    conc_organ_matches[organ]['correct'] += 1

        # Check each organ in generated (reverse direction)
        for organ in gen_organs:
            if organ not in ["Impressions", "Conclusion"] and organ in target_organs:  # Only check target organs
                conc_organ_matches_reverse[organ]['total'] += 1
                if organ in gt_organs:
                    conc_organ_matches_reverse[organ]['correct'] += 1

# Check for impression-only and findings-only reports
def is_impression_only(conc_data, report_id):
    if report_id not in conc_data:
        return False
    return list(conc_data[report_id].keys()) == ["Conclusion"]

def is_findings_only(desc_data, report_id):
    if report_id not in desc_data:
        return False
    return list(desc_data[report_id].keys()) == ["Findings"]

# Analyze reports that exist in generated results
for report_id in generated_conc:
    if report_id in ground_truth_conc:
        # Count impression-only reports
        report_type_counts['ground_truth']['impression_only']['total'] += 1
        report_type_counts['generated']['impression_only']['total'] += 1
        
        gt_impression_only = is_impression_only(ground_truth_conc, report_id)
        gen_impression_only = is_impression_only(generated_conc, report_id)
        
        if gt_impression_only:
            report_type_counts['ground_truth']['impression_only']['count'] += 1
        if gen_impression_only:
            report_type_counts['generated']['impression_only']['count'] += 1
        if gt_impression_only and gen_impression_only:
            report_type_counts['matching']['impression_only'] += 1

for report_id in generated_desc:
    if report_id in ground_truth_desc:
        # Count findings-only reports
        report_type_counts['ground_truth']['findings_only']['total'] += 1
        report_type_counts['generated']['findings_only']['total'] += 1
        
        gt_findings_only = is_findings_only(ground_truth_desc, report_id)
        gen_findings_only = is_findings_only(generated_desc, report_id)
        
        if gt_findings_only:
            report_type_counts['ground_truth']['findings_only']['count'] += 1
        if gen_findings_only:
            report_type_counts['generated']['findings_only']['count'] += 1
        if gt_findings_only and gen_findings_only:
            report_type_counts['matching']['findings_only'] += 1

def create_comparison_plots():
    """Create comprehensive comparison plots between ground truth and generated results."""
    
    # Create figure with subplots for display
    fig_display, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig_display.suptitle('Ground Truth vs Generated Results: Organ Detection Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Organ Detection Accuracy (Ground Truth -> Generated)
    organs = []
    desc_accuracies = []
    conc_accuracies = []
    combined_accuracies = []
    
    for organ in target_organs:
        organs.append(organ.capitalize())
        
        # Description accuracy
        desc_acc = (desc_organ_matches[organ]['correct'] / desc_organ_matches[organ]['total'] * 100) if desc_organ_matches[organ]['total'] > 0 else 0
        desc_accuracies.append(desc_acc)
        
        # Conclusion accuracy
        conc_acc = (conc_organ_matches[organ]['correct'] / conc_organ_matches[organ]['total'] * 100) if conc_organ_matches[organ]['total'] > 0 else 0
        conc_accuracies.append(conc_acc)
        
        # Combined accuracy
        total_correct = desc_organ_matches[organ]['correct'] + conc_organ_matches[organ]['correct']
        total_cases = desc_organ_matches[organ]['total'] + conc_organ_matches[organ]['total']
        combined_acc = (total_correct / total_cases * 100) if total_cases > 0 else 0
        combined_accuracies.append(combined_acc)
    
    x_pos = np.arange(len(organs))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, desc_accuracies, width, label='Descriptions', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos, conc_accuracies, width, label='Conclusions', alpha=0.8, color='lightcoral')
    bars3 = ax1.bar(x_pos + width, combined_accuracies, width, label='Combined', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Organs', fontweight='bold')
    ax1.set_ylabel('Detection Accuracy (%)', fontweight='bold')
    ax1.set_title('Organ Detection Accuracy (Ground Truth → Generated)', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(organs)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Bidirectional Comparison
    gt_to_gen_acc = []
    gen_to_gt_acc = []
    
    for organ in target_organs:
        # GT -> Generated
        total_correct = desc_organ_matches[organ]['correct'] + conc_organ_matches[organ]['correct']
        total_cases = desc_organ_matches[organ]['total'] + conc_organ_matches[organ]['total']
        gt_gen_acc = (total_correct / total_cases * 100) if total_cases > 0 else 0
        gt_to_gen_acc.append(gt_gen_acc)
        
        # Generated -> GT
        total_correct_rev = desc_organ_matches_reverse[organ]['correct'] + conc_organ_matches_reverse[organ]['correct']
        total_cases_rev = desc_organ_matches_reverse[organ]['total'] + conc_organ_matches_reverse[organ]['total']
        gen_gt_acc = (total_correct_rev / total_cases_rev * 100) if total_cases_rev > 0 else 0
        gen_to_gt_acc.append(gen_gt_acc)
    
    x_pos = np.arange(len(organs))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, gt_to_gen_acc, width, label='Ground Truth → Generated', alpha=0.8, color='skyblue')
    bars2 = ax2.bar(x_pos + width/2, gen_to_gt_acc, width, label='Generated → Ground Truth', alpha=0.8, color='lightcoral')
    
    ax2.set_xlabel('Organs', fontweight='bold')
    ax2.set_ylabel('Detection Accuracy (%)', fontweight='bold')
    ax2.set_title('Bidirectional Organ Detection Comparison', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(organs)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Report Type Distribution
    categories = ['Conclusion-only', 'Findings-only']
    
    gt_percentages = [
        (report_type_counts['ground_truth']['impression_only']['count'] / report_type_counts['ground_truth']['impression_only']['total'] * 100) if report_type_counts['ground_truth']['impression_only']['total'] > 0 else 0,
        (report_type_counts['ground_truth']['findings_only']['count'] / report_type_counts['ground_truth']['findings_only']['total'] * 100) if report_type_counts['ground_truth']['findings_only']['total'] > 0 else 0
    ]
    
    gen_percentages = [
        (report_type_counts['generated']['impression_only']['count'] / report_type_counts['generated']['impression_only']['total'] * 100) if report_type_counts['generated']['impression_only']['total'] > 0 else 0,
        (report_type_counts['generated']['findings_only']['count'] / report_type_counts['generated']['findings_only']['total'] * 100) if report_type_counts['generated']['findings_only']['total'] > 0 else 0
    ]
    
    match_percentages = [
        (report_type_counts['matching']['impression_only'] / report_type_counts['generated']['impression_only']['total'] * 100) if report_type_counts['generated']['impression_only']['total'] > 0 else 0,
        (report_type_counts['matching']['findings_only'] / report_type_counts['generated']['findings_only']['total'] * 100) if report_type_counts['generated']['findings_only']['total'] > 0 else 0
    ]
    
    x_pos = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax3.bar(x_pos - width, gt_percentages, width, label='Ground Truth', alpha=0.8, color='skyblue')
    bars2 = ax3.bar(x_pos, gen_percentages, width, label='Generated', alpha=0.8, color='lightcoral')
    bars3 = ax3.bar(x_pos + width, match_percentages, width, label='Matching', alpha=0.8, color='lightgreen')
    
    ax3.set_xlabel('Report Types', fontweight='bold')
    ax3.set_ylabel('Percentage (%)', fontweight='bold')
    ax3.set_title('Report Type Distribution Comparison', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Case Count Comparison
    desc_gt_counts = [desc_organ_matches[organ]['total'] for organ in target_organs]
    desc_gen_counts = [desc_organ_matches_reverse[organ]['total'] for organ in target_organs]
    conc_gt_counts = [conc_organ_matches[organ]['total'] for organ in target_organs]
    conc_gen_counts = [conc_organ_matches_reverse[organ]['total'] for organ in target_organs]
    
    x_pos = np.arange(len(organs))
    width = 0.2
    
    ax4.bar(x_pos - 1.5*width, desc_gt_counts, width, label='Desc (GT)', alpha=0.8, color='skyblue')
    ax4.bar(x_pos - 0.5*width, desc_gen_counts, width, label='Desc (Gen)', alpha=0.8, color='lightblue')
    ax4.bar(x_pos + 0.5*width, conc_gt_counts, width, label='Conc (GT)', alpha=0.8, color='lightcoral')
    ax4.bar(x_pos + 1.5*width, conc_gen_counts, width, label='Conc (Gen)', alpha=0.8, color='mistyrose')
    
    ax4.set_xlabel('Organs', fontweight='bold')
    ax4.set_ylabel('Number of Cases', fontweight='bold')
    ax4.set_title('Case Count Comparison by Organ', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(organs)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create separate figure with only key plots for saving
    fig_save, (ax_save1, ax_save2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_save.suptitle('Ground Truth vs Generated Results: Key Comparisons', fontsize=16, fontweight='bold')
    
    # Recreate Plot 1: Organ Detection Accuracy
    bars1 = ax_save1.bar(x_pos - width, desc_accuracies, width, label='Descriptions', alpha=0.8, color='skyblue')
    bars2 = ax_save1.bar(x_pos, conc_accuracies, width, label='Conclusions', alpha=0.8, color='lightcoral')
    bars3 = ax_save1.bar(x_pos + width, combined_accuracies, width, label='Combined', alpha=0.8, color='lightgreen')
    
    ax_save1.set_xlabel('Organs', fontweight='bold')
    ax_save1.set_ylabel('Detection Accuracy (%)', fontweight='bold')
    ax_save1.set_title('Organ Detection Accuracy (Ground Truth → Generated)', fontweight='bold')
    ax_save1.set_xticks(x_pos)
    ax_save1.set_xticklabels(organs)
    ax_save1.legend()
    ax_save1.grid(True, alpha=0.3)
    ax_save1.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_save1.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Recreate Plot 2: Bidirectional Comparison
    x_pos_bi = np.arange(len(organs))
    width_bi = 0.35
    
    bars1 = ax_save2.bar(x_pos_bi - width_bi/2, gt_to_gen_acc, width_bi, label='Ground Truth → Generated', alpha=0.8, color='skyblue')
    bars2 = ax_save2.bar(x_pos_bi + width_bi/2, gen_to_gt_acc, width_bi, label='Generated → Ground Truth', alpha=0.8, color='lightcoral')
    
    ax_save2.set_xlabel('Organs', fontweight='bold')
    ax_save2.set_ylabel('Detection Accuracy (%)', fontweight='bold')
    ax_save2.set_title('Bidirectional Organ Detection Comparison', fontweight='bold')
    ax_save2.set_xticks(x_pos_bi)
    ax_save2.set_xticklabels(organs)
    ax_save2.legend()
    ax_save2.grid(True, alpha=0.3)
    ax_save2.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_save2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    return fig_display, fig_save

# Create plots
print("Creating comparison plots...")
fig_display, fig_save = create_comparison_plots()

# Save plots
output_file = "output/ct_rate/ground_truth_vs_generated_comparison.png"
fig_save.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plots saved to {output_file}")

# Print results
print("\n=== Analysis Summary ===")
print(f"Total reports analyzed in conclusions: {report_type_counts['generated']['impression_only']['total']}")
print(f"Total reports analyzed in descriptions: {report_type_counts['generated']['findings_only']['total']}")

print("\n=== Organ Detection Statistics (Descriptions) ===")
for organ, stats in desc_organ_matches.items():
    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"{organ}:")
    print(f"  - Found in {stats['correct']} out of {stats['total']} cases ({accuracy:.2f}%)")

print("\n=== Organ Detection Statistics (Conclusions) ===")
for organ, stats in conc_organ_matches.items():
    accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"{organ}:")
    print(f"  - Found in {stats['correct']} out of {stats['total']} cases ({accuracy:.2f}%)")

print("\n=== Combined Organ Detection Statistics (Both Directions) ===")
for organ in target_organs:
    print(f"\n{organ}:")
    
    # Ground Truth -> Generated direction
    desc_correct = desc_organ_matches[organ]['correct']
    desc_total = desc_organ_matches[organ]['total']
    conc_correct = conc_organ_matches[organ]['correct']
    conc_total = conc_organ_matches[organ]['total']
    
    total_correct = desc_correct + conc_correct
    total_cases = desc_total + conc_total
    accuracy = (total_correct / total_cases * 100) if total_cases > 0 else 0
    
    print(f"Ground Truth -> Generated:")
    print(f"  - Total: Found in {total_correct} out of {total_cases} cases ({accuracy:.2f}%)")
    print(f"  - In Descriptions: {desc_correct}/{desc_total}")
    print(f"  - In Conclusions: {conc_correct}/{conc_total}")
    
    # Generated -> Ground Truth direction
    desc_correct_rev = desc_organ_matches_reverse[organ]['correct']
    desc_total_rev = desc_organ_matches_reverse[organ]['total']
    conc_correct_rev = conc_organ_matches_reverse[organ]['correct']
    conc_total_rev = conc_organ_matches_reverse[organ]['total']
    
    total_correct_rev = desc_correct_rev + conc_correct_rev
    total_cases_rev = desc_total_rev + conc_total_rev
    accuracy_rev = (total_correct_rev / total_cases_rev * 100) if total_cases_rev > 0 else 0
    
    print(f"\nGenerated -> Ground Truth:")
    print(f"  - Total: Found in {total_correct_rev} out of {total_cases_rev} cases ({accuracy_rev:.2f}%)")
    print(f"  - In Descriptions: {desc_correct_rev}/{desc_total_rev}")
    print(f"  - In Conclusions: {conc_correct_rev}/{conc_total_rev}")

print("\n=== Report Type Statistics ===")
# Impression-only statistics
gt_imp_percentage = (report_type_counts['ground_truth']['impression_only']['count'] / report_type_counts['ground_truth']['impression_only']['total'] * 100) if report_type_counts['ground_truth']['impression_only']['total'] > 0 else 0
gen_imp_percentage = (report_type_counts['generated']['impression_only']['count'] / report_type_counts['generated']['impression_only']['total'] * 100) if report_type_counts['generated']['impression_only']['total'] > 0 else 0
match_imp_percentage = (report_type_counts['matching']['impression_only'] / report_type_counts['generated']['impression_only']['total'] * 100) if report_type_counts['generated']['impression_only']['total'] > 0 else 0

print("Conclusion-only Reports:")
print(f"Ground Truth: {report_type_counts['ground_truth']['impression_only']['count']} out of {report_type_counts['ground_truth']['impression_only']['total']} reports ({gt_imp_percentage:.2f}%)")
print(f"Generated Results: {report_type_counts['generated']['impression_only']['count']} out of {report_type_counts['generated']['impression_only']['total']} reports ({gen_imp_percentage:.2f}%)")
print(f"Matching Cases: {report_type_counts['matching']['impression_only']} reports correctly identified ({match_imp_percentage:.2f}%)")

# Findings-only statistics
gt_find_percentage = (report_type_counts['ground_truth']['findings_only']['count'] / report_type_counts['ground_truth']['findings_only']['total'] * 100) if report_type_counts['ground_truth']['findings_only']['total'] > 0 else 0
gen_find_percentage = (report_type_counts['generated']['findings_only']['count'] / report_type_counts['generated']['findings_only']['total'] * 100) if report_type_counts['generated']['findings_only']['total'] > 0 else 0
match_find_percentage = (report_type_counts['matching']['findings_only'] / report_type_counts['generated']['findings_only']['total'] * 100) if report_type_counts['generated']['findings_only']['total'] > 0 else 0

print("\nFindings-only Reports:")
print(f"Ground Truth: {report_type_counts['ground_truth']['findings_only']['count']} out of {report_type_counts['ground_truth']['findings_only']['total']} reports ({gt_find_percentage:.2f}%)")
print(f"Generated Results: {report_type_counts['generated']['findings_only']['count']} out of {report_type_counts['generated']['findings_only']['total']} reports ({gen_find_percentage:.2f}%)")
print(f"Matching Cases: {report_type_counts['matching']['findings_only']} reports correctly identified ({match_find_percentage:.2f}%)")

# Create and save data summary to CSV
summary_data = []
for organ in target_organs:
    # GT -> Generated direction
    desc_correct = desc_organ_matches[organ]['correct']
    desc_total = desc_organ_matches[organ]['total']
    conc_correct = conc_organ_matches[organ]['correct']
    conc_total = conc_organ_matches[organ]['total']
    
    desc_acc = (desc_correct / desc_total * 100) if desc_total > 0 else 0
    conc_acc = (conc_correct / conc_total * 100) if conc_total > 0 else 0
    
    total_correct = desc_correct + conc_correct
    total_cases = desc_total + conc_total
    combined_acc = (total_correct / total_cases * 100) if total_cases > 0 else 0
    
    # Generated -> GT direction
    desc_correct_rev = desc_organ_matches_reverse[organ]['correct']
    desc_total_rev = desc_organ_matches_reverse[organ]['total']
    conc_correct_rev = conc_organ_matches_reverse[organ]['correct']
    conc_total_rev = conc_organ_matches_reverse[organ]['total']
    
    total_correct_rev = desc_correct_rev + conc_correct_rev
    total_cases_rev = desc_total_rev + conc_total_rev
    combined_acc_rev = (total_correct_rev / total_cases_rev * 100) if total_cases_rev > 0 else 0
    
    summary_data.append({
        'Organ': organ.capitalize(),
        'GT_to_Gen_Desc_Accuracy': desc_acc,
        'GT_to_Gen_Conc_Accuracy': conc_acc,
        'GT_to_Gen_Combined_Accuracy': combined_acc,
        'Gen_to_GT_Combined_Accuracy': combined_acc_rev,
        'GT_Desc_Cases': desc_total,
        'GT_Conc_Cases': conc_total,
        'Gen_Desc_Cases': desc_total_rev,
        'Gen_Conc_Cases': conc_total_rev
    })

df_summary = pd.DataFrame(summary_data)
csv_file = "output/ct_rate/ground_truth_vs_generated_summary.csv"
df_summary.to_csv(csv_file, index=False)
print(f"\nSummary data saved to {csv_file}")

# Show plots
plt.show()

print("\nAnalysis and visualization complete!")
