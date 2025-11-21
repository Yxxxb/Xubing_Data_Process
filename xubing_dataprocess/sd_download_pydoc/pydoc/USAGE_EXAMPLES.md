# 使用示例

## 快速开始

### 1. 最简单的方式（推荐）

使用交互式启动脚本：

```bash
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc
./QUICKSTART.sh
```

按照提示选择即可。

### 2. 命令行方式

#### 下载 top 1000 包（推荐用于首次使用）

```bash
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc

# 使用 8 个并行进程
python 6_master_download.py -n 1000 -j 8
```

#### 下载 top 10000 包

```bash
# 使用 16 个并行进程（需要较强的 CPU）
python 6_master_download.py -n 10000 -j 16
```

## 进阶使用

### 分批下载（推荐用于大量包）

如果需要下载大量包，建议分批进行：

```bash
# 第一次：下载 top 10000，但只处理前 500 个
python 6_master_download.py -n 10000 -j 8 --limit 500

# 后续运行：继续下载（自动跳过已下载的）
python 6_master_download.py -n 10000 -j 8 --skip-fetch
```

### 调整递归深度

如果只需要主文档和第一层子模块：

```bash
# 递归深度设为 1
python 6_master_download.py -n 1000 -j 8 --max-depth 1
```

### 只获取包列表，不下载

```bash
# 先获取 top 10000 的包名列表
python 6_master_download.py -n 10000 --fetch-only

# 稍后再下载
python 6_master_download.py -n 10000 -j 8 --skip-fetch
```

### 使用已有的包列表

如果已经有包列表文件：

```bash
# 使用现有的 top_packages.json
python 6_master_download.py --skip-fetch -j 8
```

## 后台运行（推荐用于长时间任务）

使用 tmux 或 screen 在后台运行，避免 SSH 断开：

### 使用 tmux

```bash
# 创建新会话
tmux new -s pydoc

# 在 tmux 中运行下载
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc
python 6_master_download.py -n 10000 -j 16

# 按 Ctrl+B 然后按 D 来 detach（退出但保持运行）

# 稍后重新连接
tmux attach -t pydoc

# 查看所有会话
tmux ls

# 结束会话
tmux kill-session -t pydoc
```

### 使用 screen

```bash
# 创建新会话
screen -S pydoc

# 在 screen 中运行下载
cd /pfs/training-data/xubingye/code/MMDataKit/xubing_dataprocess/download_data/pydoc
python 6_master_download.py -n 10000 -j 16

# 按 Ctrl+A 然后按 D 来 detach

# 重新连接
screen -r pydoc
```

### 使用 nohup（最简单但无法交互）

```bash
nohup python 6_master_download.py -n 10000 -j 16 > download.log 2>&1 &

# 查看日志
tail -f download.log

# 查看进程
ps aux | grep 6_master_download.py
```

## 监控和管理

### 查看下载进度

```bash
# 查看已下载的包数量
ls -d /pfs/training-data/xubingye/data/code_doc/pydoc/*/ | wc -l

# 查看总大小
du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/

# 查看每个包的大小（排序）
du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/*/ | sort -h

# 查看最大的 10 个包
du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/*/ | sort -hr | head -10
```

### 查看特定包的文档

```bash
# 列出 numpy 的所有文档
ls /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/

# 统计 numpy 的文档数量
ls /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/ | wc -l

# 查看 numpy 的文档大小
du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/
```

### 中断和恢复

```bash
# 运行时按 Ctrl+C 中断

# 再次运行会自动跳过已下载的包
python 6_master_download.py -n 10000 -j 16 --skip-fetch
```

## 自定义场景

### 场景1: 只下载特定的包

编辑或创建包列表文件：

```bash
# 创建自定义包列表
cat > my_packages.json << EOF
[
  "numpy",
  "pandas",
  "matplotlib",
  "scikit-learn",
  "tensorflow"
]
EOF

# 修改脚本使用自定义列表（或手动编辑 5_parallel_download.py）
python 5_parallel_download.py -j 4
```

### 场景2: 批量处理，每次 100 个包

```bash
# 创建一个循环脚本
for i in {0..9}; do
  echo "Processing batch $((i+1))/10..."
  python 6_master_download.py -n 1000 -j 8 --skip-fetch --limit 100
  sleep 10
done
```

### 场景3: 更新已下载的包

如果想重新下载某个包的文档（比如包更新了）：

```bash
# 删除旧文档
rm -rf /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/

# 重新下载（脚本会检测到不存在并下载）
python 6_master_download.py -n 1000 -j 8 --skip-fetch
```

## 性能优化建议

### 根据 CPU 核心数调整并行度

```bash
# 查看 CPU 核心数
nproc

# 一般建议使用 50-100% 的核心数
# 4 核 CPU: -j 4
# 8 核 CPU: -j 8
# 16 核 CPU: -j 12-16
# 32 核 CPU: -j 24-32
```

### 根据内存调整并行度

```bash
# 查看可用内存
free -h

# 如果内存不足，减少并行度
# < 8GB RAM: -j 2-4
# 8-16GB RAM: -j 4-8
# 16-32GB RAM: -j 8-16
# > 32GB RAM: -j 16+
```

### 磁盘空间估算

```bash
# Top 100: ~500MB - 1GB
# Top 1000: ~5GB - 10GB
# Top 10000: ~50GB - 100GB

# 检查可用空间
df -h /pfs/training-data/xubingye/data/
```

## 故障排除

### 问题: 包安装失败

```bash
# 单独安装失败的包
pip install package_name

# 或跳过失败的包（脚本会自动处理）
```

### 问题: 进程被杀死（内存不足）

```bash
# 减少并行度
python 6_master_download.py -n 1000 -j 2

# 或分批处理
python 6_master_download.py -n 1000 -j 8 --limit 50
```

### 问题: 下载太慢

```bash
# 增加并行度
python 6_master_download.py -n 1000 -j 16

# 减少递归深度（跳过深层子模块）
python 6_master_download.py -n 1000 -j 8 --max-depth 2
```

### 问题: 某些包的模块名不对

编辑 `5_parallel_download.py` 中的 `PACKAGE_TO_MODULE` 字典：

```python
PACKAGE_TO_MODULE = {
    "beautifulsoup4": "bs4",
    "pytorch": "torch",
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    # 添加更多映射
    "your-package": "your_module",
}
```

## 实际案例

### 案例1: 下载常用数据科学包

```bash
# 创建数据科学包列表
cat > datascience_packages.json << EOF
[
  "numpy", "pandas", "matplotlib", "seaborn", "plotly",
  "scikit-learn", "scipy", "statsmodels",
  "jupyter", "notebook", "ipython",
  "tensorflow", "pytorch", "keras",
  "xgboost", "lightgbm", "catboost"
]
EOF

# 下载（需手动修改脚本或使用 5_parallel_download.py）
```

### 案例2: 批量下载 Web 开发包

```bash
cat > web_packages.json << EOF
[
  "django", "flask", "fastapi", "tornado",
  "requests", "httpx", "aiohttp",
  "sqlalchemy", "psycopg2", "pymongo",
  "celery", "redis", "boto3"
]
EOF
```

### 案例3: 下载 top 5000 包（分 5 批）

```bash
#!/bin/bash
# 创建批处理脚本

for batch in {1..5}; do
  echo "================================================================"
  echo "Processing batch $batch/5..."
  echo "================================================================"
  
  python 6_master_download.py -n 5000 -j 12 --skip-fetch
  
  echo ""
  echo "Batch $batch completed. Waiting 30 seconds..."
  sleep 30
done

echo "All batches completed!"
```

## 查看结果

### 浏览文档

```bash
# 在浏览器中打开（需要配置 Web 服务器）
# 或使用 Python 的简单 HTTP 服务器
cd /pfs/training-data/xubingye/data/code_doc/pydoc/
python -m http.server 8000

# 然后在浏览器中访问 http://localhost:8000/numpy/numpy.html
```

### 搜索特定内容

```bash
# 在所有文档中搜索某个函数
grep -r "def array" /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/

# 统计某个包的文档中的类数量
grep -c "class " /pfs/training-data/xubingye/data/code_doc/pydoc/numpy/*.html
```

