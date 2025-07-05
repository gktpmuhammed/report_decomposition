import json
import re

def jaccard_similarity(text1, text2):
    """Calculates Jaccard similarity between two texts."""
    # Pre-process text: lowercase, remove punctuation, split into words
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 1.0 # Both texts are empty
        
    return len(intersection) / len(union)

def compare_descriptions(file1, file2):
    """
    Compares two JSON files containing report descriptions.
    """
    try:
        with open(file1, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure both files exist.")
        return

    total_reports = 0
    total_anatomies_compared = 0
    total_similarity = 0
    dissimilar_count = 0
    dissimilar_examples = []
    similarity_threshold = 0.5 # Consider descriptions with similarity below this as dissimilar

    # Find common reports (volumes)
    common_volumes = set(data1.keys()) & set(data2.keys())
    
    print(f"Found {len(common_volumes)} common reports to compare.")

    for volume in common_volumes:
        report1 = data1[volume]
        report2 = data2[volume]
        
        # Find common anatomies within the report
        common_anatomies = set(report1.keys()) & set(report2.keys())
        
        report_has_comparison = False
        
        for anatomy in common_anatomies:
            desc1 = report1.get(anatomy, "")
            desc2 = report2.get(anatomy, "")

            # Skip default "no significant abnormalities" text
            if "shows no significant abnormalities" in desc1 or \
               "shows no significant abnormalities" in desc2:
                continue
            
            report_has_comparison = True
            total_anatomies_compared += 1
            
            similarity = jaccard_similarity(desc1, desc2)
            total_similarity += similarity
            
            if similarity < similarity_threshold:
                dissimilar_count += 1
                if len(dissimilar_examples) < 5: # Store a few examples
                    dissimilar_examples.append({
                        'volume': volume,
                        'anatomy': anatomy,
                        'similarity': f"{similarity:.2f}",
                        'description_1': desc1,
                        'description_2': desc2
                    })

        if report_has_comparison:
            total_reports += 1

    if total_anatomies_compared > 0:
        average_similarity = total_similarity / total_anatomies_compared
        print("\n--- Comparison Results ---")
        print(f"Compared {total_reports} reports and {total_anatomies_compared} non-default anatomy descriptions.")
        print(f"Average Jaccard Similarity: {average_similarity:.2f}")
        print(f"Found {dissimilar_count} descriptions with similarity below {similarity_threshold}.")
    else:
        print("\n--- Comparison Results ---")
        print("No non-default anatomy descriptions were found to compare.")

    if dissimilar_examples:
        print("\n--- Examples of Dissimilar Descriptions ---")
        for ex in dissimilar_examples:
            print(f"\nVolume: {ex['volume']}, Anatomy: {ex['anatomy']}, Similarity: {ex['similarity']}")
            print(f"  File 1 (desc_info.json): {ex['description_1']}")
            print(f"  File 2 (desc_info_manual.json): {ex['description_2']}")
            
if __name__ == "__main__":
    compare_descriptions('./data/desc_info.json', './data/desc_info_manual.json') 