import json
import pandas as pd
from typing import Dict, Set, List
from collections import defaultdict

def load_json_file(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_coverage(data: Dict) -> Dict:
    """Analyze organ coverage in the data."""
    organ_counts = defaultdict(int)
    total_reports = len(data)
    
    for patient_data in data.values():
        for section in ['findings', 'impressions']:
            if section in patient_data:
                for key in patient_data[section].keys():
                    if key.lower() not in ['findings', 'conclusion']:
                        organ_counts[key.lower()] += 1
    
    return {
        'total_reports': total_reports,
        'organ_coverage': {k: v for k, v in organ_counts.items()},
        'avg_organs_per_report': sum(organ_counts.values()) / total_reports if total_reports > 0 else 0
    }

def compare_descriptions(json_v4: Dict, json_kv: Dict) -> Dict:
    """Compare descriptions between both versions."""
    common_patients = set(json_v4.keys()) & set(json_kv.keys())
    
    differences = {
        'total_patients_v4': len(json_v4),
        'total_patients_kv': len(json_kv),
        'common_patients': len(common_patients),
        'description_differences': 0,
        'organ_count_differences': 0,
        'examples': []
    }
    
    for patient_id in common_patients:
        v4_data = json_v4[patient_id]
        kv_data = json_kv[patient_id]
        
        # Compare organ counts
        v4_organs = set()
        kv_organs = set()
        
        for section in ['findings', 'impressions']:
            if section in v4_data:
                v4_organs.update(k.lower() for k in v4_data[section].keys() if k.lower() not in ['findings', 'conclusion'])
            if section in kv_data:
                kv_organs.update(k.lower() for k in kv_data[section].keys() if k.lower() not in ['findings', 'conclusion'])
        
        if len(v4_organs) != len(kv_organs):
            differences['organ_count_differences'] += 1
        
        # Compare descriptions
        for section in ['findings', 'impressions']:
            if section in v4_data and section in kv_data:
                common_organs = set(v4_data[section].keys()) & set(kv_data[section].keys())
                for organ in common_organs:
                    if v4_data[section][organ] != kv_data[section][organ]:
                        differences['description_differences'] += 1
                        if len(differences['examples']) < 5:  # Store up to 5 examples
                            differences['examples'].append({
                                'patient_id': patient_id,
                                'section': section,
                                'organ': organ,
                                'v4_desc': v4_data[section][organ],
                                'kv_desc': kv_data[section][organ]
                            })
    
    return differences

def main():
    # Load both JSON files
    json_v4 = load_json_file('data/conc_info_manual_v4.json')
    json_kv = load_json_file('data/conc_info_manual_kv_v1.json')
    
    # Analyze coverage
    v4_coverage = analyze_coverage(json_v4)
    kv_coverage = analyze_coverage(json_kv)
    
    # Compare descriptions
    comparison = compare_descriptions(json_v4, json_kv)
    
    # Print results
    print("\n=== Coverage Analysis ===")
    print(f"V4 Total Reports: {v4_coverage['total_reports']}")
    print(f"KV Total Reports: {kv_coverage['total_reports']}")
    print(f"\nV4 Avg Organs/Report: {v4_coverage['avg_organs_per_report']:.2f}")
    print(f"KV Avg Organs/Report: {kv_coverage['avg_organs_per_report']:.2f}")
    
    print("\n=== Organ Coverage Comparison ===")
    all_organs = set(v4_coverage['organ_coverage'].keys()) | set(kv_coverage['organ_coverage'].keys())
    for organ in sorted(all_organs):
        v4_count = v4_coverage['organ_coverage'].get(organ, 0)
        kv_count = kv_coverage['organ_coverage'].get(organ, 0)
        print(f"{organ}:")
        print(f"  V4: {v4_count} reports")
        print(f"  KV: {kv_count} reports")
        print(f"  Difference: {kv_count - v4_count:+d}")
    
    print("\n=== Description Comparison ===")
    print(f"Common patients: {comparison['common_patients']}")
    print(f"Description differences: {comparison['description_differences']}")
    print(f"Organ count differences: {comparison['organ_count_differences']}")
    
    if comparison['examples']:
        print("\n=== Example Differences ===")
        for i, example in enumerate(comparison['examples'], 1):
            print(f"\nExample {i}:")
            print(f"Patient: {example['patient_id']}")
            print(f"Section: {example['section']}")
            print(f"Organ: {example['organ']}")
            print(f"V4: {example['v4_desc']}")
            print(f"KV: {example['kv_desc']}")

if __name__ == '__main__':
    main() 