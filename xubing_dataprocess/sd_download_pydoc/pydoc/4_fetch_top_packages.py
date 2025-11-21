"""
爬取 PyPI Stats 网页，获取 top N 的包名列表
"""
import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path

def fetch_top_packages(top_n=1000):
    """
    从 https://n0x5.github.io/PyPI_Stats/all.html 获取 top N 的包名
    """
    url = "https://n0x5.github.io/PyPI_Stats/all.html"
    
    print(f"Fetching package list from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    packages = []
    
    # 方法1: 查找表格
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')[1:]  # 跳过表头
        for i, row in enumerate(rows):
            if i >= top_n:
                break
            cols = row.find_all('td')
            if cols:
                # 通常第一列或第二列是包名
                package_name = None
                for col in cols:
                    text = col.get_text(strip=True)
                    # 检查是否是合法的包名
                    if text and re.match(r'^[a-zA-Z0-9_\-\.]+$', text) and not text.isdigit():
                        # 查找链接
                        link = col.find('a')
                        if link:
                            href = link.get('href', '')
                            if 'pypi.org' in href or 'project' in href:
                                package_name = text
                                break
                        # 如果没有链接，但看起来像包名
                        elif not package_name and len(text) > 2:
                            package_name = text
                
                if package_name:
                    packages.append(package_name)
                    if len(packages) % 100 == 0:
                        print(f"  Found {len(packages)} packages...")
    
    # 方法2: 查找所有 pypi.org 链接
    if not packages:
        print("  Trying alternative method: searching for PyPI links...")
        links = soup.find_all('a', href=re.compile(r'pypi\.org/project/'))
        for link in links[:top_n]:
            href = link.get('href', '')
            match = re.search(r'pypi\.org/project/([^/]+)', href)
            if match:
                package_name = match.group(1)
                if package_name not in packages:
                    packages.append(package_name)
    
    # 方法3: 查找列表项
    if not packages:
        print("  Trying alternative method: searching for list items...")
        items = soup.find_all(['li', 'div'], class_=re.compile(r'package|item'))
        for item in items[:top_n]:
            text = item.get_text(strip=True)
            # 尝试提取包名
            match = re.search(r'\b([a-zA-Z0-9_\-]+)\b', text)
            if match:
                package_name = match.group(1)
                if len(package_name) > 2 and package_name not in packages:
                    packages.append(package_name)
    
    print(f"\n✓ Found {len(packages)} packages")
    return packages[:top_n]

def save_package_list(packages, filename="top_packages.json"):
    """保存包列表到文件"""
    output_dir = Path("/pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    
    with open(output_file, 'w') as f:
        json.dump(packages, f, indent=2)
    
    print(f"✓ Saved package list to {output_file}")
    return output_file

def load_package_list(filename="top_packages.json"):
    """从文件加载包列表"""
    input_file = Path("/pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc") / filename
    
    if not input_file.exists():
        return []
    
    with open(input_file, 'r') as f:
        packages = json.load(f)
    
    print(f"✓ Loaded {len(packages)} packages from {input_file}")
    return packages

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch top PyPI packages')
    parser.add_argument('-n', '--top-n', type=int, default=1000,
                        help='Number of top packages to fetch (default: 1000)')
    parser.add_argument('-o', '--output', type=str, default='top_packages.json',
                        help='Output filename (default: top_packages.json)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Fetching Top {args.top_n} PyPI Packages")
    print("=" * 70)
    
    packages = fetch_top_packages(args.top_n)
    
    if packages:
        output_file = save_package_list(packages, args.output)
        print(f"\n✓ Successfully fetched {len(packages)} packages")
        print(f"✓ Saved to: {output_file}")
        
        # 显示前20个包
        print(f"\nFirst 20 packages:")
        for i, pkg in enumerate(packages[:20], 1):
            print(f"  {i}. {pkg}")
        
        if len(packages) > 20:
            print(f"  ... and {len(packages) - 20} more")
    else:
        print("\n✗ Failed to fetch packages")
        print("You may need to manually inspect the webpage structure")

if __name__ == "__main__":
    main()

