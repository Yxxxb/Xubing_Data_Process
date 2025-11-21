import argparse
import json

parser = argparse.ArgumentParser(description='Merge json files')
parser.add_argument('--json_files', nargs='+', help='json files to merge')
parser.add_argument('--output_file', help='output file')
args = parser.parse_args()

if __name__ == '__main__':
    data = []
    for json_file in args.json_files:
        with open(json_file, 'r') as f:
            data.extend(json.load(f))

    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=2)
