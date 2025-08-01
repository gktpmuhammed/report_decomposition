#!/usr/bin/env python3
"""
Analysis script to plot organ case distributions for inspect decomposed medical reports.
Creates plots showing how many cases each organ has in the inspect impressions data.
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

def extract_organ_counts(data, data_type="inspect"):
    """Extract organ counts from the decomposed inspect data."""
    if data is None:
        return {}
    
    organ_counts = defaultdict(int)
    excluded_keys = {'Conclusion', 'Impression', 'IMPRESSION'}  # These are not organs
    
    print(f"Analyzing {data_type} data...")
    
    for case_id, case_data in data.items():
        if not isinstance(case_data, dict):
            continue
            
        # Count each organ that appears in this case
        for key in case_data.keys():
            if key not in excluded_keys and case_data[key]:  # Only count non-empty findings
                # Skip if the value indicates "No mention" or similar
                if isinstance(case_data[key], str) and case_data[key].lower().strip() in ['no mention', '', 'none']:
                    continue
                organ_counts[key] += 1
    
    print(f"Found {len(organ_counts)} organs in {data_type} data")
    return dict(organ_counts)

def create_inspect_plots(organ_counts):
    """Create plots for organ distributions in inspect data."""
    
    # Sort organs by count for better visualization
    organs_sorted = dict(sorted(organ_counts.items(), key=lambda x: x[1], reverse=True))
    organs = list(organs_sorted.keys())
    counts = list(organs_sorted.values())
    
    # Create DataFrame for easier plotting
    df_data = []
    for organ, count in organs_sorted.items():
        df_data.append({'Organ': organ, 'Count': count})
    
    df = pd.DataFrame(df_data)
    
    # Create figure with all plots for display
    fig_display, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig_display.suptitle('Organ Distribution Analysis in INSPECT Decomposed Medical Reports', fontsize=16, fontweight='bold')
    
    # Plot 1: Horizontal bar chart (main plot)
    colors = plt.cm.Set3(np.linspace(0, 1, len(organs)))
    bars = ax1.barh(range(len(organs)), counts, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(organs)))
    ax1.set_yticklabels(organs)
    ax1.set_xlabel('Number of Cases', fontweight='bold')
    ax1.set_title('INSPECT Data: Organ Distribution (Horizontal)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts):
        if v > 0:
            ax1.text(v + max(counts)*0.01, i, f'{v}', va='center', fontsize=9)
    
    # Plot 2: Vertical bar chart
    ax2.bar(range(len(organs)), counts, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(organs)))
    ax2.set_xticklabels(organs, rotation=45, ha='right')
    ax2.set_ylabel('Number of Cases', fontweight='bold')
    ax2.set_title('INSPECT Data: Organ Distribution (Vertical)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts):
        if v > 0:
            ax2.text(i, v + max(counts)*0.01, f'{v}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Pie chart for top organs
    top_n = min(10, len(organs))  # Show top 10 or fewer if less than 10 organs
    top_organs = organs[:top_n]
    top_counts = counts[:top_n]
    
    # Handle "others" category if there are more than 10 organs
    if len(organs) > top_n:
        others_count = sum(counts[top_n:])
        top_organs.append('Others')
        top_counts.append(others_count)
    
    ax3.pie(top_counts, labels=top_organs, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'INSPECT Data: Top {top_n} Organ Distribution', fontweight='bold')
    
    # Plot 4: Cumulative distribution
    cumulative_counts = np.cumsum(sorted(counts, reverse=True))
    total_cases = cumulative_counts[-1]
    cumulative_percentage = (cumulative_counts / total_cases) * 100
    
    ax4.plot(range(1, len(cumulative_counts) + 1), cumulative_percentage, 'o-', linewidth=2, markersize=6)
    ax4.set_xlabel('Number of Organs (Ranked by Frequency)', fontweight='bold')
    ax4.set_ylabel('Cumulative Percentage of Cases', fontweight='bold')
    ax4.set_title('INSPECT Data: Cumulative Organ Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    # Add percentage labels for first few points
    for i in range(min(5, len(cumulative_percentage))):
        ax4.annotate(f'{cumulative_percentage[i]:.1f}%', 
                    (i+1, cumulative_percentage[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Create separate figure with only the main plots for saving (plots 1 and 2)
    fig_save, (ax_save1, ax_save2) = plt.subplots(1, 2, figsize=(16, 8))
    fig_save.suptitle('Organ Distribution Analysis in INSPECT Decomposed Medical Reports', fontsize=16, fontweight='bold')
    
    # Recreate Plot 1: Horizontal bar chart
    ax_save1.barh(range(len(organs)), counts, color=colors, alpha=0.8)
    ax_save1.set_yticks(range(len(organs)))
    ax_save1.set_yticklabels(organs)
    ax_save1.set_xlabel('Number of Cases', fontweight='bold')
    ax_save1.set_title('INSPECT Data: Organ Distribution (Horizontal)', fontweight='bold')
    ax_save1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts):
        if v > 0:
            ax_save1.text(v + max(counts)*0.01, i, f'{v}', va='center', fontsize=9)
    
    # Recreate Plot 2: Vertical bar chart
    ax_save2.bar(range(len(organs)), counts, color=colors, alpha=0.8)
    ax_save2.set_xticks(range(len(organs)))
    ax_save2.set_xticklabels(organs, rotation=45, ha='right')
    ax_save2.set_ylabel('Number of Cases', fontweight='bold')
    ax_save2.set_title('INSPECT Data: Organ Distribution (Vertical)', fontweight='bold')
    ax_save2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(counts):
        if v > 0:
            ax_save2.text(i, v + max(counts)*0.01, f'{v}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    return fig_display, fig_save, df

def create_detailed_analysis(organ_counts):
    """Create detailed analysis and statistics."""
    print("\n" + "="*60)
    print("DETAILED INSPECT ORGAN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal unique organs found: {len(organ_counts)}")
    
    # Sort organs by frequency
    sorted_organs = sorted(organ_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Organ':<20} {'Cases':<10} {'Percentage':<12}")
    print("-" * 45)
    
    total_cases = sum(organ_counts.values())
    for organ, count in sorted_organs:
        percentage = (count / total_cases) * 100
        print(f"{organ:<20} {count:<10} {percentage:.1f}%")
    
    # Summary statistics
    counts = list(organ_counts.values())
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total organ mentions: {total_cases}")
    print(f"Average mentions per organ: {np.mean(counts):.1f}")
    print(f"Median mentions per organ: {np.median(counts):.1f}")
    print(f"Standard deviation: {np.std(counts):.1f}")
    print(f"Most frequent organ: {sorted_organs[0][0]} ({sorted_organs[0][1]} cases)")
    print(f"Least frequent organ: {sorted_organs[-1][0]} ({sorted_organs[-1][1]} cases)")
    
    # Distribution analysis
    print(f"\nTOP 5 MOST FREQUENT ORGANS:")
    for i, (organ, count) in enumerate(sorted_organs[:5], 1):
        percentage = (count / total_cases) * 100
        print(f"  {i}. {organ}: {count} cases ({percentage:.1f}%)")
    
    print(f"\nBOTTOM 5 LEAST FREQUENT ORGANS:")
    for i, (organ, count) in enumerate(sorted_organs[-5:], 1):
        percentage = (count / total_cases) * 100
        print(f"  {i}. {organ}: {count} cases ({percentage:.1f}%)")

def main():
    """Main analysis function."""
    print("Starting INSPECT organ distribution analysis...")
    
    # File path
    inspect_file = "output/inspect/inspect_impressions.json"
    
    # Check if file exists
    if not os.path.exists(inspect_file):
        print(f"Error: {inspect_file} not found!")
        return
    
    # Load data
    inspect_data = load_json_file(inspect_file)
    
    if inspect_data is None:
        print("Failed to load data file!")
        return
    
    # Extract organ counts
    organ_counts = extract_organ_counts(inspect_data, "inspect")
    
    if not organ_counts:
        print("No organ data found!")
        return
    
    # Create detailed analysis
    create_detailed_analysis(organ_counts)
    
    # Create plots
    print("\nCreating plots...")
    fig_display, fig_save, df = create_inspect_plots(organ_counts)
    
    # Save only the main plots (horizontal and vertical bar charts)
    output_file = "output/inspect/inspect_organ_distribution_analysis.png"
    fig_save.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_file}")
    
    # Save data to CSV
    csv_file = "output/inspect/inspect_organ_distribution_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")
    
    # Show all plots for display
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 