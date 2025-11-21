"""
快速测试脚本 - 测试下载功能是否正常
下载 5 个常见的小型包，验证整个流程
"""
import subprocess
import sys
import json
from pathlib import Path

# 测试用的小型包
TEST_PACKAGES = [
    "requests",
    "click",
    "rich",
    "pydantic",
    "typer"
]

SCRIPT_DIR = Path(__file__).parent
TEST_LIST_FILE = SCRIPT_DIR / "test_packages.json"

def create_test_package_list():
    """创建测试用的包列表"""
    with open(TEST_LIST_FILE, 'w') as f:
        json.dump(TEST_PACKAGES, f, indent=2)
    print(f"✓ Created test package list: {TEST_LIST_FILE}")

def run_test_download():
    """运行测试下载"""
    # 临时修改环境变量，指向测试包列表
    import os
    env = os.environ.copy()
    
    script_path = SCRIPT_DIR / "5_parallel_download.py"
    
    # 修改脚本中的包列表文件路径
    with open(script_path, 'r') as f:
        content = f.read()
    
    # 临时创建一个测试版本的脚本
    test_script = SCRIPT_DIR / "5_parallel_download_test.py"
    test_content = content.replace(
        'PACKAGE_LIST_FILE = "/pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc/top_packages.json"',
        f'PACKAGE_LIST_FILE = "{TEST_LIST_FILE}"'
    )
    
    with open(test_script, 'w') as f:
        f.write(test_content)
    
    print("\n" + "="*70)
    print("Running test download...")
    print("="*70)
    
    cmd = [sys.executable, str(test_script), '-j', '2', '--max-depth', '2']
    result = subprocess.run(cmd)
    
    # 清理测试脚本
    test_script.unlink()
    
    return result.returncode == 0

def main():
    print("="*70)
    print("Quick Test - Downloading 5 Small Packages")
    print("="*70)
    print(f"\nTest packages: {', '.join(TEST_PACKAGES)}")
    print(f"This will test the basic functionality before downloading thousands of packages\n")
    
    # 创建测试包列表
    create_test_package_list()
    
    # 运行测试下载
    success = run_test_download()
    
    if success:
        print("\n" + "="*70)
        print("✓ Test completed successfully!")
        print("="*70)
        print("\nYou can now run the full download:")
        print("  python 6_master_download.py -n 1000 -j 8")
    else:
        print("\n" + "="*70)
        print("✗ Test failed")
        print("="*70)
        print("\nPlease check the error messages above")
    
    # 清理测试文件
    if TEST_LIST_FILE.exists():
        TEST_LIST_FILE.unlink()

if __name__ == "__main__":
    main()

