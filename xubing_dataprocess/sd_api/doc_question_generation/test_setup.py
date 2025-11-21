#!/usr/bin/env python3
"""
Test script to verify the setup is correct.
"""

import sys
from pathlib import Path

def test_dependencies():
    """Test if all dependencies are installed."""
    print("Testing dependencies...")
    
    missing = []
    
    try:
        import openai
        print("✓ openai installed")
    except ImportError:
        missing.append("openai")
        print("✗ openai not installed")
    
    try:
        import aiofiles
        print("✓ aiofiles installed")
    except ImportError:
        missing.append("aiofiles")
        print("✗ aiofiles not installed")
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies installed ✓")
    return True


def test_directories():
    """Test if directories exist and are accessible."""
    print("\nTesting directories...")
    
    input_dir = Path("/data/extracted_apis")
    output_dir = Path("/data/generated_questions")
    prompt_file = Path("/home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation/prompt_scene.txt")
    
    # Check input directory
    if not input_dir.exists():
        print(f"✗ Input directory not found: {input_dir}")
        return False
    print(f"✓ Input directory exists: {input_dir}")
    
    # Count input files
    input_files = list(input_dir.glob("*.jsonl"))
    print(f"  Found {len(input_files)} JSONL files")
    
    if len(input_files) == 0:
        print("  ✗ No JSONL files found in input directory!")
        return False
    
    # Show sample files
    print(f"  Sample files:")
    for f in input_files[:3]:
        print(f"    - {f.name}")
    if len(input_files) > 3:
        print(f"    ... and {len(input_files) - 3} more")
    
    # Check prompt file
    if not prompt_file.exists():
        print(f"✗ Prompt file not found: {prompt_file}")
        return False
    print(f"✓ Prompt file exists: {prompt_file}")
    
    # Check output directory (create if needed)
    if not output_dir.exists():
        print(f"  Output directory will be created: {output_dir}")
    else:
        print(f"✓ Output directory exists: {output_dir}")
    
    print("\nAll directories OK ✓")
    return True


def test_api_key():
    """Test if API key is valid format."""
    print("\nTesting API key...")
    
    api_key = "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"
    
    if not api_key or not api_key.startswith("sk-"):
        print("✗ API key format looks invalid")
        return False
    
    print("✓ API key format looks valid")
    print("  (Note: This doesn't test if the key actually works)")
    return True


def test_sample_file():
    """Test reading a sample file."""
    print("\nTesting sample file reading...")
    
    input_dir = Path("/data/extracted_apis")
    input_files = list(input_dir.glob("*.jsonl"))
    
    if not input_files:
        print("✗ No files to test")
        return False
    
    sample_file = input_files[0]
    print(f"Reading: {sample_file.name}")
    
    try:
        import json
        count = 0
        with open(sample_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Only read first 3 lines
                    break
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    count += 1
                    
                    # Check required fields
                    required_fields = ['library', 'object', 'kind']
                    missing_fields = [field for field in required_fields if field not in entry]
                    
                    if missing_fields:
                        print(f"✗ Entry {i+1} missing fields: {missing_fields}")
                        return False
        
        print(f"✓ Successfully read {count} entries from sample file")
        print(f"  First entry: {entry.get('library')}.{entry.get('object')}")
        return True
        
    except Exception as e:
        print(f"✗ Error reading sample file: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Setup Verification")
    print("="*60)
    
    results = []
    
    results.append(("Dependencies", test_dependencies()))
    results.append(("Directories", test_directories()))
    results.append(("API Key", test_api_key()))
    results.append(("Sample File", test_sample_file()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*60)
    if all_passed:
        print("\n✓ All tests passed! You're ready to run the generator.")
        print("\nTo start processing:")
        print("  python3 generate_questions.py")
        print("\nOr use the shell script:")
        print("  chmod +x run.sh")
        print("  ./run.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

