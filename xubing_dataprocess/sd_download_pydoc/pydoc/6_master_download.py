"""
主控制脚本：一键下载 top N Python 包的完整文档
整合所有步骤：爬取包列表 -> 并行下载主文档 -> 递归下载子文档
"""
import subprocess
import sys
import os
import json
import time
from pathlib import Path

# 配置
SCRIPT_DIR = Path(__file__).parent
PYDOC_ROOT = "/pfs/training-data/xubingye/data/code_doc/pydoc"
PACKAGE_LIST_FILE = SCRIPT_DIR / "top_packages.json"

def run_script(script_name, args=[]):
    """运行子脚本"""
    script_path = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Script failed with code {e.returncode}")
        return False

def check_package_list_exists():
    """检查包列表文件是否存在"""
    return PACKAGE_LIST_FILE.exists()

def get_package_count():
    """获取包列表中的包数量"""
    if not PACKAGE_LIST_FILE.exists():
        return 0
    
    with open(PACKAGE_LIST_FILE, 'r') as f:
        packages = json.load(f)
    return len(packages)

def get_downloaded_count():
    """获取已下载的包数量"""
    pydoc_root = Path(PYDOC_ROOT)
    if not pydoc_root.exists():
        return 0
    
    count = 0
    for subdir in pydoc_root.iterdir():
        if subdir.is_dir():
            # 检查是否有主包的 HTML 文件
            main_html = subdir / f"{subdir.name}.html"
            if main_html.exists():
                count += 1
    
    return count

def get_total_size():
    """获取已下载文档的总大小"""
    pydoc_root = Path(PYDOC_ROOT)
    if not pydoc_root.exists():
        return 0
    
    total_size = 0
    for root, dirs, files in os.walk(pydoc_root):
        for file in files:
            if file.endswith('.html'):
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
    
    return total_size

def format_size(bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Master script to download Python package documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download top 1000 packages with 8 parallel jobs
  python 6_master_download.py -n 1000 -j 8
  
  # Download top 10000 packages with 16 parallel jobs
  python 6_master_download.py -n 10000 -j 16
  
  # Skip fetching package list (use existing)
  python 6_master_download.py --skip-fetch -j 8
  
  # Only fetch package list, don't download
  python 6_master_download.py -n 5000 --fetch-only
        """
    )
    
    parser.add_argument('-n', '--top-n', type=int, default=1000,
                        help='Number of top packages to download (default: 1000)')
    parser.add_argument('-j', '--jobs', type=int, default=8,
                        help='Number of parallel jobs (default: 8)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum recursion depth for submodules (default: 5)')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching package list (use existing)')
    parser.add_argument('--fetch-only', action='store_true',
                        help='Only fetch package list, do not download')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of packages to process in this run')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Master Python Package Documentation Downloader")
    print("=" * 70)
    print(f"  Target: Top {args.top_n} packages")
    print(f"  Parallel jobs: {args.jobs}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Output directory: {PYDOC_ROOT}")
    print("=" * 70)
    
    # 显示当前状态
    print("\nCurrent Status:")
    if check_package_list_exists():
        pkg_count = get_package_count()
        print(f"  Package list: {pkg_count} packages")
    else:
        print(f"  Package list: Not found")
    
    downloaded_count = get_downloaded_count()
    total_size = get_total_size()
    print(f"  Downloaded: {downloaded_count} packages")
    print(f"  Total size: {format_size(total_size)}")
    print()
    
    start_time = time.time()
    
    # 步骤1: 获取包列表
    if not args.skip_fetch:
        print("\n" + "=" * 70)
        print("STEP 1: Fetching Package List")
        print("=" * 70)
        
        if not run_script('4_fetch_top_packages.py', 
                         ['-n', str(args.top_n), '-o', 'top_packages.json']):
            print("[ERROR] Failed to fetch package list")
            return 1
        
        print(f"\n✓ Package list saved to: {PACKAGE_LIST_FILE}")
    else:
        print("\n[INFO] Skipping package list fetch (using existing)")
        if not check_package_list_exists():
            print(f"[ERROR] Package list not found: {PACKAGE_LIST_FILE}")
            print("Please run without --skip-fetch to fetch the package list")
            return 1
    
    # 如果只是获取列表，则到此结束
    if args.fetch_only:
        print("\n[INFO] Fetch-only mode, stopping here")
        return 0
    
    # 步骤2: 并行下载文档
    print("\n" + "=" * 70)
    print("STEP 2: Downloading Documentation (Parallel)")
    print("=" * 70)
    
    download_args = [
        '-j', str(args.jobs),
        '--max-depth', str(args.max_depth),
    ]
    
    if args.limit:
        download_args.extend(['-n', str(args.limit)])
    
    if not run_script('5_parallel_download.py', download_args):
        print("[ERROR] Failed to download documentation")
        return 1
    
    # 最终统计
    total_elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    final_downloaded = get_downloaded_count()
    final_size = get_total_size()
    
    print(f"  Total packages downloaded: {final_downloaded}")
    print(f"  Total documentation size: {format_size(final_size)}")
    print(f"  Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f} seconds)")
    
    if final_downloaded > downloaded_count:
        new_packages = final_downloaded - downloaded_count
        print(f"  New packages in this run: {new_packages}")
        print(f"  Average time per package: {total_elapsed/new_packages:.1f}s")
    
    print("\n" + "=" * 70)
    print("✓ All done!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

