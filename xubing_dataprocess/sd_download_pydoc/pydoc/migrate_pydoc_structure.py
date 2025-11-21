#!/usr/bin/env python3
"""
迁移 pydoc 文档从旧结构到新结构

旧结构：所有文件在同一目录
  /pydoc/numpy.html
  /pydoc/numpy.core.html
  /pydoc/pandas.html
  /pydoc/pandas.DataFrame.html

新结构：每个库一个目录
  /pydoc/numpy/numpy.html
  /pydoc/numpy/numpy.core.html
  /pydoc/pandas/pandas.html
  /pydoc/pandas/pandas.DataFrame.html
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

OLD_PYDOC_DIR = "/pfs/training-data/xubingye/data/pydoc"
NEW_PYDOC_DIR = "/pfs/training-data/xubingye/data/code_doc/pydoc"

def analyze_structure():
    """分析当前目录结构"""
    old_dir = Path(OLD_PYDOC_DIR)
    
    if not old_dir.exists():
        print(f"[ERROR] Directory does not exist: {OLD_PYDOC_DIR}")
        return None
    
    # 统计文件
    all_files = list(old_dir.glob("*.html"))
    
    if not all_files:
        print(f"[INFO] No HTML files found in {OLD_PYDOC_DIR}")
        return None
    
    # 检查是否是旧结构（文件在根目录）或新结构（文件在子目录）
    files_in_root = [f for f in all_files if f.parent == old_dir]
    subdirs = [d for d in old_dir.iterdir() if d.is_dir()]
    
    print("\n" + "=" * 70)
    print("Current Structure Analysis")
    print("=" * 70)
    print(f"Directory: {OLD_PYDOC_DIR}")
    print(f"HTML files in root: {len(files_in_root)}")
    print(f"Subdirectories: {len(subdirs)}")
    
    if subdirs:
        print("\nSubdirectories found:")
        for subdir in subdirs[:10]:
            file_count = len(list(subdir.glob("*.html")))
            print(f"  {subdir.name}: {file_count} files")
        if len(subdirs) > 10:
            print(f"  ... and {len(subdirs) - 10} more")
    
    return {
        'files_in_root': files_in_root,
        'subdirs': subdirs,
        'is_old_structure': len(files_in_root) > 0 and len(subdirs) == 0
    }

def group_files_by_package(files):
    """将文件按主包分组"""
    groups = defaultdict(list)
    
    for file_path in files:
        filename = file_path.stem  # 不带 .html 的文件名
        
        # 提取主包名（第一个点之前）
        if '.' in filename:
            main_package = filename.split('.')[0]
        else:
            main_package = filename
        
        groups[main_package].append(file_path)
    
    return groups

def migrate_files(dry_run=True):
    """执行迁移"""
    analysis = analyze_structure()
    
    if not analysis:
        return
    
    if not analysis['is_old_structure']:
        print("\n[INFO] Directory appears to already be in new structure")
        print("[INFO] Migration not needed")
        return
    
    files_in_root = analysis['files_in_root']
    groups = group_files_by_package(files_in_root)
    
    print("\n" + "=" * 70)
    print("Migration Plan")
    print("=" * 70)
    print(f"Total packages: {len(groups)}")
    print(f"Total files: {len(files_in_root)}")
    print(f"Destination: {NEW_PYDOC_DIR}")
    print()
    
    # 显示每个包的统计
    for package, files in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
        print(f"  {package:20s}: {len(files):4d} files")
    
    if len(groups) > 15:
        print(f"  ... and {len(groups) - 15} more packages")
    
    print()
    
    if dry_run:
        print("[DRY RUN] No files will be moved. Review the plan above.")
        print("[DRY RUN] Run with --execute to perform the migration.")
        return
    
    # 确认
    response = input("\n❓ Proceed with migration? (yes/NO): ")
    if response.lower() != 'yes':
        print("❌ Migration cancelled")
        return
    
    print("\n" + "=" * 70)
    print("Starting Migration")
    print("=" * 70)
    
    new_root = Path(NEW_PYDOC_DIR)
    new_root.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for package, files in groups.items():
        # 创建包目录
        package_dir = new_root / package
        package_dir.mkdir(exist_ok=True)
        
        print(f"\n[📦] Migrating {package} ({len(files)} files)...")
        
        for file_path in files:
            try:
                dest_path = package_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                success_count += 1
                
                if success_count % 50 == 0:
                    print(f"  [INFO] Migrated {success_count} files...")
                
            except Exception as e:
                print(f"  [ERROR] Failed to migrate {file_path.name}: {e}")
                error_count += 1
    
    print("\n" + "=" * 70)
    print("Migration Complete")
    print("=" * 70)
    print(f"✅ Successfully migrated: {success_count} files")
    if error_count > 0:
        print(f"❌ Errors: {error_count} files")
    print(f"\n📁 New location: {NEW_PYDOC_DIR}")
    print()
    print("💡 Next steps:")
    print("  1. Verify the new structure:")
    print(f"     ls -lh {NEW_PYDOC_DIR}/")
    print("  2. If everything looks good, you can remove the old files:")
    print(f"     rm {OLD_PYDOC_DIR}/*.html")
    print("  3. Run the recursive doc generator:")
    print("     python 3_pydoc_sub_recursive.py")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate pydoc structure")
    parser.add_argument('--execute', action='store_true',
                       help='Actually perform the migration (default: dry run)')
    parser.add_argument('--old-dir', default=OLD_PYDOC_DIR,
                       help=f'Old pydoc directory (default: {OLD_PYDOC_DIR})')
    parser.add_argument('--new-dir', default=NEW_PYDOC_DIR,
                       help=f'New pydoc directory (default: {NEW_PYDOC_DIR})')
    
    args = parser.parse_args()
    
    # 更新全局变量
    global OLD_PYDOC_DIR, NEW_PYDOC_DIR
    OLD_PYDOC_DIR = args.old_dir
    NEW_PYDOC_DIR = args.new_dir
    
    print("=" * 70)
    print("Pydoc Structure Migration Tool")
    print("=" * 70)
    
    migrate_files(dry_run=not args.execute)

if __name__ == "__main__":
    main()

