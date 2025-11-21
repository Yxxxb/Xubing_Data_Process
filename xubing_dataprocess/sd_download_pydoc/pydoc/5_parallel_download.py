"""
并行下载 Python 包文档（主文档 + 递归子模块）
使用多进程大幅提升下载速度
"""
import subprocess
import sys
import os
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import re
from collections import deque

# 配置
PYDOC_ROOT = "/pfs/training-data/xubingye/data/code_doc/pydoc"
PACKAGE_LIST_FILE = "/pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc/top_packages.json"

# 包名和模块名映射
PACKAGE_TO_MODULE = {
    "beautifulsoup4": "bs4",
    "pytorch": "torch",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "protobuf": "google.protobuf",
}

def run_cmd(cmd, timeout=60):
    """执行命令，带超时"""
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"

def get_module_name(package_name):
    """获取包的模块名（用于 import）"""
    return PACKAGE_TO_MODULE.get(package_name, package_name.replace('-', '_'))

def is_package_installed(package_name):
    """检查包是否已安装"""
    code, out, err = run_cmd(f"{sys.executable} -m pip show {package_name}", timeout=10)
    return code == 0

def install_package(package_name):
    """安装包"""
    print(f"  [+] Installing {package_name}...")
    code, out, err = run_cmd(f"{sys.executable} -m pip install --quiet {package_name}", timeout=300)
    return code == 0

def get_package_dir(package_name):
    """获取包的文档目录"""
    module_name = get_module_name(package_name)
    return Path(PYDOC_ROOT) / module_name

def get_html_path(module_name):
    """获取模块的HTML文件路径"""
    main_module = module_name.split('.')[0]
    package_dir = Path(PYDOC_ROOT) / main_module
    filename = f"{module_name}.html"
    return package_dir / filename

def check_main_doc_exists(package_name):
    """检查主文档是否已存在"""
    module_name = get_module_name(package_name)
    main_html = get_html_path(module_name)
    return main_html.exists()

def generate_main_doc(package_name):
    """生成主文档"""
    module_name = get_module_name(package_name)
    target_path = get_html_path(module_name)
    
    # 确保目标目录存在
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 切换到目标目录
    original_dir = os.getcwd()
    try:
        os.chdir(target_path.parent)
        
        # 生成文档
        code, out, err = run_cmd(f"{sys.executable} -m pydoc -w {module_name}", timeout=120)
        
        # 检查是否成功
        if target_path.exists():
            print(f"  [✓] Generated {module_name}.html")
            return True
        else:
            print(f"  [✗] Failed to generate {module_name}.html")
            return False
    except Exception as e:
        print(f"  [✗] Error generating {module_name}.html: {e}")
        return False
    finally:
        os.chdir(original_dir)

def extract_submodules_from_html(html_file, base_package):
    """从 HTML 文档中提取子模块名称"""
    try:
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return []
    
    submodules = set()
    
    # 方法1：查找所有以 base_package. 开头的完整模块引用
    pattern1 = rf'\b({re.escape(base_package)}\.\w+(?:\.\w+)*)\b'
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        if match.count('.') <= 10 and not match.endswith('.html'):
            submodules.add(match)
    
    # 方法2：查找 HTML 链接中的子模块
    pattern2 = rf'href="({re.escape(base_package)}\.\w+(?:\.\w+)*)\.html"'
    matches2 = re.findall(pattern2, content)
    submodules.update(matches2)
    
    # 方法3：查找 "PACKAGE CONTENTS" 部分
    package_contents_pattern = r'<strong>PACKAGE CONTENTS</strong>.*?(?=<strong>|$)'
    package_section = re.search(package_contents_pattern, content, re.DOTALL | re.IGNORECASE)
    if package_section:
        section_content = package_section.group(0)
        submodule_names = re.findall(r'>(\w+(?:\.\w+)*)</a>', section_content)
        for name in submodule_names:
            if name and not name.startswith('_'):
                if '.' not in name:
                    submodules.add(f"{base_package}.{name}")
                elif name.startswith(base_package):
                    submodules.add(name)
    
    # 过滤
    submodules.discard(base_package)
    filtered_submodules = set()
    for submod in submodules:
        if submod.startswith(f"{base_package}.") and len(submod) > len(base_package) + 2:
            filtered_submodules.add(submod)
    
    return sorted(filtered_submodules)

def generate_submodule_doc(module_name):
    """为指定的子模块生成文档"""
    target_path = get_html_path(module_name)
    
    # 检查是否已存在
    if target_path.exists():
        return True
    
    # 确保目标目录存在
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 切换到目标目录
    original_dir = os.getcwd()
    try:
        os.chdir(target_path.parent)
        
        # 生成文档
        code, out, err = run_cmd(f"{sys.executable} -m pydoc -w {module_name}", timeout=60)
        
        # 检查是否成功生成
        if target_path.exists():
            return True
        else:
            return False
    except Exception as e:
        return False
    finally:
        os.chdir(original_dir)

def process_package_recursive(package_name, max_depth=5):
    """
    递归处理一个包，生成其所有层级的子模块文档
    """
    module_name = get_module_name(package_name)
    package_dir = get_package_dir(package_name)
    main_html = get_html_path(module_name)
    
    if not main_html.exists():
        return 0, 0
    
    # 使用队列进行 BFS
    queue = deque([(module_name, 0)])
    processed = set()
    all_submodules = set()
    
    success_count = 0
    fail_count = 0
    
    while queue:
        current_module, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        if current_module in processed:
            continue
        processed.add(current_module)
        
        # 获取当前模块的 HTML 文件
        current_html = get_html_path(current_module)
        if not current_html.exists():
            continue
        
        # 提取子模块
        submodules = extract_submodules_from_html(current_html, current_module)
        
        # 处理每个子模块
        for submodule in submodules:
            if submodule not in all_submodules:
                all_submodules.add(submodule)
                
                # 生成文档
                result = generate_submodule_doc(submodule)
                if result:
                    success_count += 1
                    queue.append((submodule, depth + 1))
                else:
                    fail_count += 1
    
    return success_count, fail_count

def process_single_package(args):
    """
    处理单个包（工作进程的主函数）
    返回：(package_name, success, main_doc_generated, submodules_generated, submodules_failed, elapsed_time)
    """
    package_name, max_depth = args
    start_time = time.time()
    
    print(f"\n[→] Processing: {package_name}")
    
    # 检查主文档是否已存在
    if check_main_doc_exists(package_name):
        print(f"  [Skip] Main doc already exists for {package_name}")
        # 只处理子模块
        success_sub, fail_sub = process_package_recursive(package_name, max_depth)
        elapsed = time.time() - start_time
        return (package_name, True, False, success_sub, fail_sub, elapsed)
    
    # 检查是否已安装
    if not is_package_installed(package_name):
        # 尝试安装
        if not install_package(package_name):
            print(f"  [✗] Failed to install {package_name}")
            elapsed = time.time() - start_time
            return (package_name, False, False, 0, 0, elapsed)
    
    # 生成主文档
    if not generate_main_doc(package_name):
        elapsed = time.time() - start_time
        return (package_name, False, False, 0, 0, elapsed)
    
    # 递归生成子模块文档
    success_sub, fail_sub = process_package_recursive(package_name, max_depth)
    
    elapsed = time.time() - start_time
    print(f"  [✓] Completed {package_name} in {elapsed:.1f}s (submodules: +{success_sub}, failed: {fail_sub})")
    
    return (package_name, True, True, success_sub, fail_sub, elapsed)

def load_package_list():
    """加载包列表"""
    package_file = Path(PACKAGE_LIST_FILE)
    if not package_file.exists():
        print(f"[ERROR] Package list not found: {PACKAGE_LIST_FILE}")
        print("Please run 4_fetch_top_packages.py first")
        return []
    
    with open(package_file, 'r') as f:
        packages = json.load(f)
    
    return packages

def filter_processed_packages(packages):
    """过滤掉已处理的包"""
    remaining = []
    for pkg in packages:
        if not check_main_doc_exists(pkg):
            remaining.append(pkg)
    
    return remaining

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel PyDoc Download')
    parser.add_argument('-j', '--jobs', type=int, default=min(8, cpu_count()),
                        help=f'Number of parallel jobs (default: min(8, {cpu_count()}))')
    parser.add_argument('-n', '--limit', type=int, default=None,
                        help='Limit number of packages to process')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip packages with existing main docs (default: True)')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum recursion depth for submodules (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Parallel Python Package Documentation Generator")
    print("=" * 70)
    print(f"  Parallel jobs: {args.jobs}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Output dir: {PYDOC_ROOT}")
    print("=" * 70)
    
    # 加载包列表
    packages = load_package_list()
    if not packages:
        return
    
    print(f"\n[INFO] Loaded {len(packages)} packages from list")
    
    # 过滤已处理的包
    if args.skip_existing:
        original_count = len(packages)
        packages = filter_processed_packages(packages)
        skipped = original_count - len(packages)
        if skipped > 0:
            print(f"[INFO] Skipping {skipped} packages with existing docs")
    
    # 限制数量
    if args.limit:
        packages = packages[:args.limit]
    
    print(f"[INFO] Will process {len(packages)} packages\n")
    
    if not packages:
        print("[INFO] No packages to process")
        return
    
    # 并行处理
    start_time = time.time()
    
    # 准备参数：(package_name, max_depth)
    package_args = [(pkg, args.max_depth) for pkg in packages]
    
    with Pool(processes=args.jobs) as pool:
        results = pool.map(process_single_package, package_args)
    
    # 统计结果
    total_elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r[1])
    fail_count = len(results) - success_count
    total_main_docs = sum(1 for r in results if r[2])
    total_submodules = sum(r[3] for r in results)
    total_failed_submodules = sum(r[4] for r in results)
    
    print(f"  Total packages: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  New main docs: {total_main_docs}")
    print(f"  New submodule docs: {total_submodules}")
    print(f"  Failed submodules: {total_failed_submodules}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Average time per package: {total_elapsed/len(results):.1f}s")
    
    # 显示失败的包
    failed_packages = [r[0] for r in results if not r[1]]
    if failed_packages:
        print(f"\nFailed packages ({len(failed_packages)}):")
        for pkg in failed_packages[:20]:
            print(f"  - {pkg}")
        if len(failed_packages) > 20:
            print(f"  ... and {len(failed_packages) - 20} more")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()

