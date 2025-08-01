#!/usr/bin/env python3
"""
Analysis script to plot organ case distributions for decomposed medical reports.
Creates plots showing how many cases each organ has in both conclusion and description files.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_json_file(filepath):
    """Load JSON file and return the data."""
    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} cases from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_organ_counts(data, data_type="unknown"):
    """Extract organ counts from the decomposed data."""
    if data is None:
        return {}
    
    organ_counts = defaultdict(int)
    excluded_keys = {'Findings', 'Conclusion'}  # These are not organs
    
    print(f"Analyzing {data_type} data...")
    
    for case_id, case_data in data.items():
        if not isinstance(case_data, dict):
            continue
            
        # Count each organ that appears in this case
        for key in case_data.keys():
            if key not in excluded_keys and case_data[key]:  # Only count non-empty findings
                organ_counts[key] += 1
    
    print(f"Found {len(organ_counts)} organs in {data_type} data")
    return dict(organ_counts)

def create_comparison_plots(conc_counts, desc_counts):
    """Create comparison plots for organ distributions."""
    
    # Get all unique organs from both datasets
    all_organs = set(list(conc_counts.keys()) + list(desc_counts.keys()))
    
    # Prepare data for plotting
    organs = sorted(all_organs)
    conc_values = [conc_counts.get(organ, 0) for organ in organs]
    desc_values = [desc_counts.get(organ, 0) for organ in organs]
    
    # Create DataFrame for easier plotting
    df_data = []
    for organ in organs:
        df_data.append({'Organ': organ, 'Count': conc_counts.get(organ, 0), 'Type': 'Conclusion'})
        df_data.append({'Organ': organ, 'Count': desc_counts.get(organ, 0), 'Type': 'Description'})
    
    df = pd.DataFrame(df_data)
    
    # Create figure with all subplots for display
    fig_display, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig_display.suptitle('Organ Distribution Analysis in Decomposed Medical Reports', fontsize=16, fontweight='bold')
    
    # Plot 1: Side-by-side bar chart
    x_pos = np.arange(len(organs))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, conc_values, width, label='Conclusion', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, desc_values, width, label='Description', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Organs', fontweight='bold')
    ax1.set_ylabel('Number of Cases', fontweight='bold')
    ax1.set_title('Organ Case Counts: Conclusion vs Description', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(organs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(conc_values + desc_values)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(conc_values + desc_values)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Conclusion only (horizontal bar chart)
    conc_sorted = dict(sorted(conc_counts.items(), key=lambda x: x[1], reverse=True))
    ax2.barh(range(len(conc_sorted)), list(conc_sorted.values()), color='skyblue', alpha=0.8)
    ax2.set_yticks(range(len(conc_sorted)))
    ax2.set_yticklabels(list(conc_sorted.keys()))
    ax2.set_xlabel('Number of Cases', fontweight='bold')
    ax2.set_title('Conclusion Data: Organ Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(conc_sorted.values()):
        if v > 0:
            ax2.text(v + max(conc_sorted.values())*0.01, i, f'{v}', va='center', fontsize=9)
    
    # Plot 3: Description only (horizontal bar chart)
    desc_sorted = dict(sorted(desc_counts.items(), key=lambda x: x[1], reverse=True))
    ax3.barh(range(len(desc_sorted)), list(desc_sorted.values()), color='lightcoral', alpha=0.8)
    ax3.set_yticks(range(len(desc_sorted)))
    ax3.set_yticklabels(list(desc_sorted.keys()))
    ax3.set_xlabel('Number of Cases', fontweight='bold')
    ax3.set_title('Description Data: Organ Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(desc_sorted.values()):
        if v > 0:
            ax3.text(v + max(desc_sorted.values())*0.01, i, f'{v}', va='center', fontsize=9)
    
    # Plot 4: Stacked bar chart using seaborn
    sns.barplot(data=df, x='Organ', y='Count', hue='Type', ax=ax4)
    ax4.set_title('Organ Distribution: Stacked Comparison', fontweight='bold')
    ax4.set_xlabel('Organs', fontweight='bold')
    ax4.set_ylabel('Number of Cases', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create separate figure with only the middle two plots for saving
    fig_save, (ax_save1, ax_save2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_save.suptitle('Organ Distribution Analysis in Decomposed Medical Reports', fontsize=16, fontweight='bold')
    
    # Recreate Plot 2: Conclusion only (horizontal bar chart)
    ax_save1.barh(range(len(conc_sorted)), list(conc_sorted.values()), color='skyblue', alpha=0.8)
    ax_save1.set_yticks(range(len(conc_sorted)))
    ax_save1.set_yticklabels(list(conc_sorted.keys()))
    ax_save1.set_xlabel('Number of Cases', fontweight='bold')
    ax_save1.set_title('Conclusion Data: Organ Distribution', fontweight='bold')
    ax_save1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(conc_sorted.values()):
        if v > 0:
            ax_save1.text(v + max(conc_sorted.values())*0.01, i, f'{v}', va='center', fontsize=9)
    
    # Recreate Plot 3: Description only (horizontal bar chart)
    ax_save2.barh(range(len(desc_sorted)), list(desc_sorted.values()), color='lightcoral', alpha=0.8)
    ax_save2.set_yticks(range(len(desc_sorted)))
    ax_save2.set_yticklabels(list(desc_sorted.keys()))
    ax_save2.set_xlabel('Number of Cases', fontweight='bold')
    ax_save2.set_title('Description Data: Organ Distribution', fontweight='bold')
    ax_save2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(desc_sorted.values()):
        if v > 0:
            ax_save2.text(v + max(desc_sorted.values())*0.01, i, f'{v}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    return fig_display, fig_save, df

def create_detailed_analysis(conc_counts, desc_counts):
    """Create detailed analysis and statistics."""
    print("\n" + "="*60)
    print("DETAILED ORGAN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    all_organs = set(list(conc_counts.keys()) + list(desc_counts.keys()))
    
    print(f"\nTotal unique organs found: {len(all_organs)}")
    print(f"Organs in conclusion data: {len(conc_counts)}")
    print(f"Organs in description data: {len(desc_counts)}")
    
    # Create comparison table
    print(f"\n{'Organ':<15} {'Conclusion':<12} {'Description':<12} {'Difference':<12}")
    print("-" * 55)
    
    for organ in sorted(all_organs):
        conc_count = conc_counts.get(organ, 0)
        desc_count = desc_counts.get(organ, 0)
        diff = desc_count - conc_count
        
        print(f"{organ:<15} {conc_count:<12} {desc_count:<12} {diff:+<12}")
    
    # Summary statistics
    total_conc = sum(conc_counts.values())
    total_desc = sum(desc_counts.values())
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total organ mentions in conclusions: {total_conc}")
    print(f"Total organ mentions in descriptions: {total_desc}")
    print(f"Average mentions per organ (conclusion): {total_conc/len(conc_counts):.1f}")
    print(f"Average mentions per organ (description): {total_desc/len(desc_counts):.1f}")
    
    # Most and least common organs
    print(f"\nTOP 5 ORGANS (Conclusion):")
    for organ, count in sorted(conc_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {organ}: {count} cases")
    
    print(f"\nTOP 5 ORGANS (Description):")
    for organ, count in sorted(desc_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {organ}: {count} cases")

def main():
    """Main analysis function."""
    print("Starting organ distribution analysis...")
    
    # File paths
    conc_file = "output/ct_rate/conc_info_manual_merged.json"
    desc_file = "output/ct_rate/desc_info_manual_merged.json"
    
    # Check if files exist
    if not os.path.exists(conc_file):
        print(f"Error: {conc_file} not found!")
        return
    if not os.path.exists(desc_file):
        print(f"Error: {desc_file} not found!")
        return
    
    # Load data
    conc_data = load_json_file(conc_file)
    desc_data = load_json_file(desc_file)
    
    if conc_data is None or desc_data is None:
        print("Failed to load data files!")
        return
    
    # Extract organ counts
    conc_counts = extract_organ_counts(conc_data, "conclusion")
    desc_counts = extract_organ_counts(desc_data, "description")
    
    # Create detailed analysis
    create_detailed_analysis(conc_counts, desc_counts)
    
    # Create plots
    print("\nCreating plots...")
    fig_display, fig_save, df = create_comparison_plots(conc_counts, desc_counts)
    
    # Save only the middle two plots
    output_file = "output/ct_rate/organ_distribution_analysis.png"
    fig_save.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_file}")
    
    # Save data to CSV
    csv_file = "output/ct_rate/organ_distribution_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")
    
    # Show all plots for display
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 