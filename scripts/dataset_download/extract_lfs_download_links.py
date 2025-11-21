import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from datakit.utils.files import find_all_files, remove_path_prefix

parser = argparse.ArgumentParser()
parser.add_argument(
    '--repo_path',
    type=Path,
    required=True,
    help="Path to the cloned repo, \
          make sure you've cloned with GIT_LFS_SKIP_SMUDGE=1",
)
parser.add_argument(
    '--repo_url',
    type=str,
    required=True,
    help='Git url of the repo, e.g. https://huggingface.co/datasets/cais/wmdp',
)
parser.add_argument('--output_path', type=Path, default=None)
args = parser.parse_args()

LFS_FILES_SUFFIX = (
    '.7z',
    '.arrow',
    '.gz',
    '.tar',
    '.tar.gz',
    '.tgz',
    '.zip',
    '.parquet',
)

repo_path = args.repo_path
lfs_files = find_all_files(repo_path, LFS_FILES_SUFFIX)
lfs_files_suffix = [
    remove_path_prefix(lfs_file, repo_path) for lfs_file in lfs_files
]
print(f'Extracting LFS download links from {len(lfs_files_suffix)} files')

repo_url = args.repo_url.strip('/')
if 'resolve/main' not in repo_url:
    repo_url = f'{repo_url}/resolve/main'
download_urls = [
    f'{repo_url}/{lfs_file_suffix}' for lfs_file_suffix in lfs_files_suffix
]

download_data = []
for lfs_file, download_url in tqdm(zip(lfs_files, download_urls)):
    with open(lfs_file, 'r', encoding='utf-8') as f:
        try:
            data = f.read()
        except UnicodeDecodeError:
            print(f'Error reading {lfs_file}'
                  'You should download without LFS files,'
                  ' make sure you set GIT_LFS_SKIP_SMUDGE=1')
        if 'version https://git-lfs.github.com/spec/v1' in data:
            size = int(data.split('size')[1].strip())
            download_data.append([size, download_url])

output_path = args.output_path
if output_path is None:
    fname = repo_path.name
    output_path = os.path.join(repo_path.parent, f'{fname}_lfs_links.json')

print(f'Saving LFS download links to {output_path}')
with open(output_path, 'w+', encoding='utf-8') as f:
    json.dump(download_data, f, indent=2)
