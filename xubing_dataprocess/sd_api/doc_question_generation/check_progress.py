#!/usr/bin/env python3
"""
Check progress of question generation.
"""

import json
from pathlib import Path
from collections import defaultdict


INPUT_DIR = Path("/data/extracted_apis")
OUTPUT_DIR = Path("/data/generated_questions")


def count_entries(jsonl_file: Path) -> int:
    """Count number of entries in a JSONL file."""
    count = 0
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        print(f"Error reading {jsonl_file}: {e}")
    return count


def main():
    """Check and display progress."""
    print("Checking progress...\n")
    
    # Get all input files
    input_files = list(INPUT_DIR.glob("*.jsonl"))
    total_files = len(input_files)
    
    # Get all output files
    output_files = set(f.name for f in OUTPUT_DIR.glob("*.jsonl")) if OUTPUT_DIR.exists() else set()
    completed_files = len(output_files)
    
    print(f"Total input files: {total_files}")
    print(f"Completed files: {completed_files}")
    print(f"Remaining files: {total_files - completed_files}")
    print(f"Progress: {completed_files/total_files*100:.2f}%\n")
    
    # Count total entries
    total_input_entries = 0
    total_output_entries = 0
    success_count = 0
    failed_count = 0
    
    print("Counting entries...")
    for i, input_file in enumerate(input_files, 1):
        entries = count_entries(input_file)
        total_input_entries += entries
        
        output_file = OUTPUT_DIR / input_file.name
        if output_file.exists():
            output_entries = count_entries(output_file)
            total_output_entries += output_entries
            
            # Count successes and failures
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line)
                            if result.get('success', False):
                                success_count += 1
                            else:
                                failed_count += 1
                        except:
                            pass
        
        if i % 50 == 0:
            print(f"  Processed {i}/{total_files} files...")
    
    print(f"\nTotal input entries: {total_input_entries:,}")
    print(f"Total output entries: {total_output_entries:,}")
    print(f"Successful generations: {success_count:,}")
    print(f"Failed generations: {failed_count:,}")
    
    if total_output_entries > 0:
        print(f"Success rate: {success_count/total_output_entries*100:.2f}%")
    
    # Show next files to process
    remaining = sorted([f.name for f in input_files if f.name not in output_files])
    if remaining:
        print(f"\nNext files to process (showing first 10):")
        for name in remaining[:10]:
            print(f"  - {name}")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")
    else:
        print("\n✓ All files processed!")


if __name__ == "__main__":
    main()

