import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Force json value into string')
parser.add_argument('--json_file', type=Path, help='json file to convert')
parser.add_argument('--output_file', type=Path, help='output file path')
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_data = {k: str(v) for k, v in data.items()}
    with open(args.output_file, 'w+', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2)
