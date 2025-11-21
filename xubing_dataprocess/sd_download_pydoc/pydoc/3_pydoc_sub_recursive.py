import subprocess
import sys
import os
import re
from pathlib import Path
from collections import deque

# 配置
PYDOC_ROOT = "/pfs/training-data/xubingye/data/code_doc/pydoc"

def run_cmd(cmd):
    """执行命令"""
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr

def extract_submodules_from_html(html_file, base_package):
    """从 HTML 文档中提取子模块名称"""
    try:
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERROR] Failed to read {html_file}: {e}")
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

def get_package_dir(package_name):
    """获取包的文档目录"""
    # 提取主包名（第一个点之前的部分）
    main_package = package_name.split('.')[0]
    return Path(PYDOC_ROOT) / main_package

def get_html_path(module_name):
    """获取模块的HTML文件路径"""
    main_package = module_name.split('.')[0]
    package_dir = Path(PYDOC_ROOT) / main_package
    # 文件名：module.submodule.html
    filename = f"{module_name}.html"
    return package_dir / filename

def get_main_packages():
    """获取所有已下载的主包名称（扫描子目录）"""
    pydoc_root = Path(PYDOC_ROOT)
    if not pydoc_root.exists():
        print(f"[ERROR] Directory {PYDOC_ROOT} does not exist")
        return []
    
    packages = []
    
    # 扫描每个子目录
    for subdir in pydoc_root.iterdir():
        if subdir.is_dir():
            # 检查是否有主包的 HTML 文件
            main_html = subdir / f"{subdir.name}.html"
            if main_html.exists():
                packages.append(subdir.name)
    
    return sorted(packages)

def generate_submodule_doc(module_name):
    """为指定的子模块生成文档"""
    target_path = get_html_path(module_name)
    
    # 检查是否已存在
    if target_path.exists():
        return True
    
    # 确保目标目录存在
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 切换到目标目录（pydoc 会在当前目录生成文件）
    original_dir = os.getcwd()
    try:
        os.chdir(target_path.parent)
        
        # 生成文档
        code, out, err = run_cmd(f"{sys.executable} -m pydoc -w {module_name}")
        
        # 检查是否成功生成
        source_file = Path(f"{module_name}.html")
        if source_file.exists():
            # 文件已经在正确的位置，只需要检查
            print(f"  [✓] Generated {module_name}.html")
            return True
        else:
            # 生成失败
            if "no Python documentation found" not in (err + out):
                if err and "ImportError" not in err and "ModuleNotFoundError" not in err:
                    # 只打印真正的错误
                    pass
            return False
    finally:
        os.chdir(original_dir)

def process_package_recursive(package_name, max_depth=10):
    """
    递归处理一个包，生成其所有层级的子模块文档
    
    使用广度优先搜索（BFS）来避免无限递归
    """
    print(f"\n[📦] Processing package: {package_name} (recursive)")
    
    # 获取主包的 HTML 路径
    package_dir = get_package_dir(package_name)
    main_html = package_dir / f"{package_name}.html"
    
    if not main_html.exists():
        print(f"  [Skip] {main_html} does not exist")
        return
    
    # 使用队列进行 BFS
    queue = deque([(package_name, 0)])  # (模块名, 深度)
    processed = set()  # 已处理的模块
    all_submodules = set()  # 所有发现的子模块
    
    success_count = 0
    fail_count = 0
    
    print(f"  [🔍] Starting recursive extraction...")
    
    while queue:
        current_module, depth = queue.popleft()
        
        # 检查深度限制
        if depth >= max_depth:
            print(f"  [INFO] Reached max depth {max_depth} for {current_module}")
            continue
        
        # 避免重复处理
        if current_module in processed:
            continue
        processed.add(current_module)
        
        # 获取当前模块的 HTML 文件
        current_html = get_html_path(current_module)
        if not current_html.exists():
            continue
        
        # 提取子模块
        submodules = extract_submodules_from_html(current_html, current_module)
        
        if submodules:
            print(f"  [INFO] Found {len(submodules)} submodules in {current_module} (depth {depth})")
        
        # 处理每个子模块
        for submodule in submodules:
            if submodule not in all_submodules:
                all_submodules.add(submodule)
                
                # 生成文档
                result = generate_submodule_doc(submodule)
                if result:
                    success_count += 1
                    # 将这个子模块加入队列，继续查找它的子模块
                    queue.append((submodule, depth + 1))
                else:
                    fail_count += 1
    
    print(f"  [✓] Processed {len(processed)} modules")
    print(f"  [✓] Generated {success_count} new docs, {fail_count} failed or unavailable")
    print(f"  [✓] Total discovered submodules: {len(all_submodules)}")

def get_package_stats(package_name):
    """获取包的统计信息"""
    package_dir = get_package_dir(package_name)
    if not package_dir.exists():
        return 0, 0
    
    html_files = list(package_dir.glob("*.html"))
    total_size = sum(f.stat().st_size for f in html_files if f.is_file())
    
    return len(html_files), total_size

def main():
    print("=" * 70)
    print("Python Package Submodule Documentation Generator (Recursive)")
    print("=" * 70)
    
    # 确保根目录存在
    pydoc_root = Path(PYDOC_ROOT)
    if not pydoc_root.exists():
        print(f"[ERROR] Root directory does not exist: {PYDOC_ROOT}")
        return
    
    # 获取所有主包
    packages = get_main_packages()
    # 这里我想处理ray及其之后的包，之前的不处理
    packages = [package for package in packages if package >= 'ray']
    print(packages)
    print(f"\n[INFO] Found {len(packages)} main packages in {PYDOC_ROOT}")
    if packages:
        print(f"[INFO] Packages: {', '.join(packages[:10])}{'...' if len(packages) > 10 else ''}")
    else:
        print(f"[INFO] No packages found. Expected directory structure:")
        print(f"       {PYDOC_ROOT}/package_name/package_name.html")
        return
    
    # 统计初始状态
    total_before = 0
    for package in packages:
        count, _ = get_package_stats(package)
        total_before += count
    print(f"[INFO] Total HTML files before: {total_before}")
    
    print("\n" + "=" * 70)
    
    # 处理每个包（递归）
    for package in packages:
        try:
            process_package_recursive(package, max_depth=5)
        except Exception as e:
            print(f"[ERROR] Error processing {package}: {e}")
    
    # 统计最终状态
    print("\n" + "=" * 70)
    print("Final Statistics:")
    print("=" * 70)
    
    total_after = 0
    for package in packages:
        count, size = get_package_stats(package)
        total_after += count
        size_mb = size / (1024 * 1024)
        print(f"  {package:20s}: {count:4d} files, {size_mb:6.2f} MB")
    
    print(f"\n  Total: {total_after} files (added {total_after - total_before} new files)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    main()

