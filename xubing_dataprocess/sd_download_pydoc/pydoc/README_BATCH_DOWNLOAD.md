# Python Package Documentation Batch Download

这是一套优化的批量下载 Python 包文档的工具，能够高效地下载 top 1000/10000 Python 包的完整文档（包括主文档和所有子模块）。

## 🚀 主要优化

相比之前的方法，新版本有以下重大改进：

1. **并行处理**：使用多进程同时处理多个包，速度提升 5-10 倍
2. **智能跳过**：自动跳过已下载的文档，支持断点续传
3. **批量安装**：不再反复卸载/重新安装包
4. **递归深度可配置**：可以控制子模块的递归深度（默认 5 层）
5. **实时进度**：显示详细的下载进度和统计信息

**性能对比**：
- 旧方法：一晚上（~8小时）下载 20 个包
- 新方法：预计 1-2 小时下载 100+ 个包（取决于 CPU 核心数）

## 📁 文件说明

- `4_fetch_top_packages.py` - 从网页爬取 top N 包名列表
- `5_parallel_download.py` - 并行下载文档（多进程）
- `6_master_download.py` - 主控制脚本（推荐使用）

## 🔧 使用方法

### 方法1: 使用主控制脚本（推荐）

一键下载 top 1000 包的完整文档：

```bash
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc

# 下载 top 1000 包，使用 8 个并行进程
python 6_master_download.py -n 1000 -j 8

# 下载 top 10000 包，使用 16 个并行进程（如果 CPU 足够）
python 6_master_download.py -n 10000 -j 16

# 下载 top 5000 包，但本次运行只处理 100 个（分批次下载）
python 6_master_download.py -n 5000 -j 8 --limit 100
```

### 方法2: 分步执行

如果需要更精细的控制，可以分步执行：

#### 步骤1: 爬取包名列表

```bash
# 获取 top 1000 包名
python 4_fetch_top_packages.py -n 1000 -o top_packages.json

# 获取 top 10000 包名
python 4_fetch_top_packages.py -n 10000 -o top10000_packages.json
```

#### 步骤2: 并行下载文档

```bash
# 使用 8 个并行进程下载
python 5_parallel_download.py -j 8

# 只下载前 50 个包（测试用）
python 5_parallel_download.py -j 8 -n 50

# 设置子模块递归深度为 3 层
python 5_parallel_download.py -j 8 --max-depth 3
```

## ⚙️ 参数说明

### 6_master_download.py（主控制脚本）

```
-n, --top-n N        要下载的包数量（默认: 1000）
-j, --jobs N         并行进程数（默认: 8）
--max-depth N        子模块递归深度（默认: 5）
--skip-fetch         跳过包列表获取，使用现有列表
--fetch-only         只获取包列表，不下载文档
--limit N            本次运行只处理 N 个包
```

### 5_parallel_download.py（并行下载）

```
-j, --jobs N         并行进程数（默认: min(8, CPU核心数)）
-n, --limit N        限制处理的包数量
--max-depth N        子模块递归深度（默认: 5）
--skip-existing      跳过已存在的包（默认: True）
```

### 4_fetch_top_packages.py（爬取包名）

```
-n, --top-n N        获取 top N 的包（默认: 1000）
-o, --output FILE    输出文件名（默认: top_packages.json）
```

## 📊 预期性能

根据您的硬件配置，预期性能如下：

| CPU 核心数 | 并行进程 | 预计速度 | Top 1000 所需时间 |
|----------|---------|---------|-----------------|
| 4 核     | 4       | ~50 包/小时 | ~20 小时 |
| 8 核     | 8       | ~100 包/小时 | ~10 小时 |
| 16 核    | 16      | ~200 包/小时 | ~5 小时 |
| 32 核    | 32      | ~400 包/小时 | ~2.5 小时 |

**注意**：实际速度取决于：
- 网络速度
- 包的大小和复杂度
- 是否需要安装包
- 磁盘 I/O 速度

## 💡 使用建议

1. **分批下载**：对于 top 10000，建议分批下载，避免一次性处理太多包
   ```bash
   # 第一批：前 1000 个
   python 6_master_download.py -n 10000 -j 16 --limit 1000
   
   # 第二批：继续下载（自动跳过已下载的）
   python 6_master_download.py -n 10000 -j 16 --skip-fetch
   ```

2. **调整并行数**：根据 CPU 核心数调整 `-j` 参数
   ```bash
   # 查看 CPU 核心数
   nproc
   
   # 建议使用核心数的 50-100%
   python 6_master_download.py -j 16  # 对于 16 核 CPU
   ```

3. **断点续传**：脚本会自动跳过已下载的包，可以随时中断和恢复
   ```bash
   # 随时按 Ctrl+C 中断
   # 再次运行时会自动继续
   python 6_master_download.py -n 1000 -j 8 --skip-fetch
   ```

4. **监控进度**：使用 `tmux` 或 `screen` 在后台运行，避免 SSH 断开
   ```bash
   # 使用 tmux
   tmux new -s pydoc
   python 6_master_download.py -n 10000 -j 16
   # 按 Ctrl+B 然后 D 来 detach
   
   # 重新连接
   tmux attach -t pydoc
   ```

5. **查看结果**：
   ```bash
   # 查看已下载的包数量
   ls -d /pfs/training-data/xubingye/data/code_doc/pydoc/*/ | wc -l
   
   # 查看总大小
   du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/
   
   # 查看某个包的文档数量
   ls /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/ | wc -l
   ```

## 🐛 故障排除

### 问题1: 某些包下载失败

**原因**：包可能无法安装、没有文档、或模块名与包名不匹配

**解决**：这是正常现象，脚本会记录失败的包并继续处理其他包

### 问题2: 下载速度太慢

**解决**：
1. 增加并行进程数 `-j 16` 或更高
2. 检查网络连接
3. 减少递归深度 `--max-depth 3`

### 问题3: 内存不足

**解决**：
1. 减少并行进程数 `-j 4`
2. 使用 `--limit` 参数分批处理

### 问题4: 磁盘空间不足

**解决**：
- Top 1000 包预计需要 5-10 GB
- Top 10000 包预计需要 50-100 GB
- 可以定期清理不需要的文档

## 📝 输出目录结构

```
/pfs/training-data/xubingye/data/code_doc/pydoc/
├── numpy/
│   ├── numpy.html              # 主文档
│   ├── numpy.core.html         # 子模块
│   ├── numpy.core.multiarray.html
│   └── ...
├── pandas/
│   ├── pandas.html
│   ├── pandas.core.html
│   └── ...
└── ...
```

## 🎯 示例：完整工作流

```bash
# 1. 进入工作目录
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc

# 2. 启动 tmux 会话（可选但推荐）
tmux new -s pydoc_download

# 3. 下载 top 10000 包（可能需要几小时）
python 6_master_download.py -n 10000 -j 16

# 4. 如果需要中断，按 Ctrl+C
# 5. 继续下载（自动跳过已下载的）
python 6_master_download.py -n 10000 -j 16 --skip-fetch

# 6. 查看结果
ls -lh /pfs/training-data/xubingye/data/code_doc/pydoc/
```

## 📈 进度追踪

脚本会输出详细的进度信息：

```
[→] Processing: numpy
  [+] Installing numpy...
  [✓] Generated numpy.html
  [INFO] Found 150 submodules in numpy (depth 0)
  [✓] Generated numpy.core.html
  [✓] Generated numpy.lib.html
  ...
  [✓] Completed numpy in 45.2s (submodules: +150, failed: 5)

Summary:
  Total packages: 1000
  Successful: 950
  Failed: 50
  New main docs: 800
  New submodule docs: 15000
  Total time: 3600s
  Average time per package: 3.6s
```

## 🔄 与旧版本的兼容性

新脚本与之前的文件格式完全兼容：
- 使用相同的输出目录结构
- 可以无缝接续之前下载的文档
- 不会重复下载已存在的文件

如果您之前使用 `1_pydoc_top100.py` 和 `3_pydoc_sub_recursive.py` 下载了一些包，新脚本会自动跳过这些包。

