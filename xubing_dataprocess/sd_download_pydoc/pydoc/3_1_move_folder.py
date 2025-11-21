# 将/pfs/training-data/xubingye/data/pydoc内前缀一致的移动到前缀的新文件夹内，例如xgboost.*都移动到新创建的xgboost文件夹内
import os
from tqdm import tqdm

pydoc_dir = '/pfs/training-data/xubingye/data/pydoc'

for file in tqdm(os.listdir(pydoc_dir)):
    if file.endswith('.html'):
        prefix = file.split('.')[0]
        new_folder = os.path.join(pydoc_dir, prefix)
        os.makedirs(new_folder, exist_ok=True)
        os.system(f'mv {os.path.join(pydoc_dir, file)} {new_folder}')
