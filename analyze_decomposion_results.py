import json
import pandas as pd
from collections import defaultdict

# Load the files
def load_json(filename):
    with open(f'data/{filename}') as f:
        return json.load(f)

# Load all data
ground_truth_conc = load_json('conc_info.json')
ground_truth_desc = load_json('desc_info.json')
generated_conc = load_json('conc_info_manual_v5.json')
generated_desc = load_json('desc_info_manual_v5.json')

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
