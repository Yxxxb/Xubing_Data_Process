#!/usr/bin/env python3
"""
Test script to process a single file with limited entries.
Useful for testing before running the full batch.
"""

import asyncio
import json
from pathlib import Path
from generate_questions import QuestionGenerator

# Configuration
INPUT_DIR = Path("/data/extracted_apis")
OUTPUT_DIR = Path("/data/test_output")
PROMPT_FILE = Path("/home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation/prompt_scene.txt")
API_KEY = "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"

# Limit entries for testing
MAX_ENTRIES = 3  # Only process first 3 entries


async def test_single_file():
    """Test processing a single file with limited entries."""
    # Get first input file
    input_files = sorted(INPUT_DIR.glob("*.jsonl"))
    if not input_files:
        print("❌ No input files found!")
        return
    
    input_file = input_files[0]
    print(f"Testing with file: {input_file.name}")
    print(f"Will process first {MAX_ENTRIES} entries only\n")
    
    # Read prompt template
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Create generator
    generator = QuestionGenerator(API_KEY, prompt_template)
    
    # Read entries (limit to MAX_ENTRIES)
    entries = []
    print("Reading entries...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_ENTRIES:
                break
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    print(f"  {i+1}. {entry.get('library')}.{entry.get('object')} ({entry.get('kind')})")
                except json.JSONDecodeError as e:
                    print(f"  ❌ Invalid JSON on line {i+1}: {e}")
    
    print(f"\nProcessing {len(entries)} entries...\n")
    
    # Process entries
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] Processing {entry.get('object')}...")
        result = await generator.process_entry(entry)
        results.append(result)
        
        # Debug: show result structure
        print(f"  Result keys: {list(result.keys())}")
        
        if result.get('success', False):
            print(f"  ✓ Success")
            # Show first 200 chars of generated question
            if 'question' in result:
                question = result['question']
                if question:
                    preview = question[:200] + "..." if len(question) > 200 else question
                    print(f"  Preview: {preview}\n")
                else:
                    print(f"  ⚠ Warning: Question is empty\n")
            else:
                print(f"  ⚠ Warning: 'question' key not found in result\n")
                print(f"  Result: {result}\n")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}\n")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"test_{input_file.name}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    print("="*60)
    print("Test Complete!")
    print(f"Processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    print(f"\nResults saved to: {output_file}")
    print("="*60)
    
    # Show full result for first entry
    if results:
        print("\n" + "="*60)
        print("Full result for first entry:")
        print("="*60)
        print(json.dumps(results[0], indent=2, ensure_ascii=False))


def main():
    """Main entry point."""
    print("="*60)
    print("Single File Test")
    print("="*60)
    print(f"This will process only {MAX_ENTRIES} entries from the first file")
    print("to verify the setup is working correctly.\n")
    
    try:
        asyncio.run(test_single_file())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

