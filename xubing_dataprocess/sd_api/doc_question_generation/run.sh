#!/bin/bash
# Script to run the question generation process

echo "Starting question generation..."
echo "This will process all API documentation files in /data/extracted_apis"
echo "Results will be saved to /data/generated_questions"
echo ""

# Run the Python script
python3 generate_questions.py

echo ""
echo "Generation complete. Check /data/generated_questions for results."

