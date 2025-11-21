import subprocess
import sys
import os
import re
from pathlib import Path

# 配置
PYDOC_DIR = "/pfs/training-data/xubingye/data/pydoc"

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
    
    # pydoc 生成的 HTML 中，子模块通常以链接形式出现
    # 格式如：<a href="datasets.splits.html">datasets.splits</a>
    # 或者在 "Package Contents" 部分列出
    
    # 方法1：查找所有以 base_package. 开头的完整模块引用
    # 匹配格式：base_package.xxx.yyy（确保是完整的模块路径）
    pattern1 = rf'\b({re.escape(base_package)}\.\w+(?:\.\w+)*)\b'
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        # 过滤掉太长的或看起来不像模块的
        if match.count('.') <= 5 and not match.endswith('.html'):  # 限制嵌套深度
            submodules.add(match)
    
    # 方法2：查找 HTML 链接中的子模块（最可靠）
    # 格式：<a href="package.submodule.html">
    pattern2 = rf'href="({re.escape(base_package)}\.\w+(?:\.\w+)*)\.html"'
    matches2 = re.findall(pattern2, content)
    submodules.update(matches2)
    
    # 方法3：查找 "PACKAGE CONTENTS" 部分（最准确的来源）
    # pydoc 会在这些部分列出子模块
    package_contents_pattern = r'<strong>PACKAGE CONTENTS</strong>.*?(?=<strong>|$)'
    package_section = re.search(package_contents_pattern, content, re.DOTALL | re.IGNORECASE)
    if package_section:
        section_content = package_section.group(0)
        # 提取该部分的所有单词（可能是子模块名）
        submodule_names = re.findall(r'>(\w+(?:\.\w+)*)</a>', section_content)
        for name in submodule_names:
            if name and not name.startswith('_'):  # 排除私有模块
                # 如果是简单名称，加上包前缀
                if '.' not in name:
                    submodules.add(f"{base_package}.{name}")
                elif name.startswith(base_package):
                    submodules.add(name)
    
    # 过滤掉基础包本身和无效项
    submodules.discard(base_package)
    
    # 额外过滤：排除看起来不像真实模块的项
    filtered_submodules = set()
    for submod in submodules:
        # 必须是 base_package.xxx 格式
        if submod.startswith(f"{base_package}.") and len(submod) > len(base_package) + 2:
            filtered_submodules.add(submod)
    
    return sorted(filtered_submodules)

def get_main_packages():
    """获取所有已下载的主包名称"""
    pydoc_dir = Path(PYDOC_DIR)
    if not pydoc_dir.exists():
        print(f"[ERROR] Directory {PYDOC_DIR} does not exist")
        return []
    
    packages = []
    for html_file in pydoc_dir.glob("*.html"):
        # 获取不带 .html 后缀的文件名作为包名
        package_name = html_file.stem
        # 排除已经是子模块的文档（包含点号的）
        if '.' not in package_name:
            packages.append(package_name)
    
    return sorted(packages)

def generate_submodule_doc(module_name):
    """为指定的子模块生成文档"""
    target_path = f"{PYDOC_DIR}/{module_name}.html"
    
    # 检查是否已存在
    if os.path.exists(target_path):
        # print(f"  [Skip] {module_name}.html already exists")  # 静默跳过已存在的
        return True
    
    # 生成文档
    code, out, err = run_cmd(f"{sys.executable} -m pydoc -w {module_name}")
    
    # 检查是否成功生成（在当前目录下）
    source_file = f"{module_name}.html"
    if os.path.exists(source_file):
        # 移动到目标目录
        try:
            os.rename(source_file, target_path)
            print(f"  [✓] Generated {module_name}.html")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to move {module_name}.html: {e}")
            return False
    else:
        # 生成失败（可能模块不存在或无法导入）
        if "no Python documentation found" in err or "no Python documentation found" in out:
            # print(f"  [Skip] {module_name} - no documentation available")  # 静默跳过
            pass
        else:
            # 只有真正的错误才打印警告
            if err and "ImportError" not in err and "ModuleNotFoundError" not in err:
                print(f"  [WARNING] Failed to generate docs for {module_name}")
                if len(err) < 500:  # 只打印较短的错误信息
                    print(f"    Error: {err.strip()[:200]}")
        return False

def process_package(package_name):
    """处理一个包，生成其所有子模块的文档"""
    print(f"\n[📦] Processing package: {package_name}")
    
    html_file = f"{PYDOC_DIR}/{package_name}.html"
    if not os.path.exists(html_file):
        print(f"  [Skip] {html_file} does not exist")
        return
    
    # 提取子模块
    print(f"  [🔍] Extracting submodules from {package_name}.html...")
    submodules = extract_submodules_from_html(html_file, package_name)
    
    if not submodules:
        print(f"  [INFO] No submodules found for {package_name}")
        return
    
    print(f"  [INFO] Found {len(submodules)} potential submodules")
    
    # 为每个子模块生成文档
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for submodule in submodules:
        result = generate_submodule_doc(submodule)
        if result:
            # 检查是新生成的还是已存在的
            target_path = f"{PYDOC_DIR}/{submodule}.html"
            if os.path.exists(target_path):
                success_count += 1
        else:
            fail_count += 1
    
    print(f"  [✓] Successfully generated {success_count} new docs, {fail_count} failed or unavailable")

def main():
    print("=" * 60)
    print("Python Package Submodule Documentation Generator")
    print("=" * 60)
    
    # 确保目标目录存在
    os.makedirs(PYDOC_DIR, exist_ok=True)
    
    # 获取所有主包
    packages = get_main_packages()
    print(f"\n[INFO] Found {len(packages)} main packages in {PYDOC_DIR}")
    print(f"[INFO] Packages: {', '.join(packages[:10])}{'...' if len(packages) > 10 else ''}")
    
    # 处理每个包
    total_generated = 0
    for package in packages:
        try:
            process_package(package)
        except Exception as e:
            print(f"[ERROR] Error processing {package}: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()

