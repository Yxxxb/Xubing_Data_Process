# 文件索引

## 📋 文档文件

| 文件 | 用途 | 何时阅读 |
|------|------|----------|
| **QUICKSTART.md** | 快速启动指南 | ⭐ 首次使用必读 |
| **PROJECT_SUMMARY.md** | 项目总结和架构说明 | 了解项目全貌 |
| **README.md** | 完整的项目文档 | 需要详细信息时 |
| **INDEX.md** | 本文件，文件索引 | 查找文件用途 |

## 🔧 执行文件

| 文件 | 用途 | 运行命令 |
|------|------|----------|
| **generate_questions.py** | 主程序，批量生成问题 | `python3 generate_questions.py` |
| **test_setup.py** | 验证环境设置 | `python3 test_setup.py` |
| **test_single_file.py** | 测试单个文件（3个词条） | `python3 test_single_file.py` |
| **check_progress.py** | 检查处理进度 | `python3 check_progress.py` |
| **run.sh** | Shell启动脚本 | `./run.sh` |

## 📦 配置文件

| 文件 | 用途 |
|------|------|
| **requirements.txt** | Python依赖包列表 |
| **prompt_scene.txt** | GPT Prompt模板 |

## 📝 生成文件（运行后生成）

| 文件/目录 | 用途 |
|-----------|------|
| **generation.log** | 运行日志 |
| **/data/generated_questions/** | 输出目录 |
| **/data/test_output/** | 测试输出目录 |

## 🚀 使用流程

### 首次使用
```
1. 阅读 QUICKSTART.md
2. 运行 test_setup.py
3. 运行 test_single_file.py (测试)
4. 运行 generate_questions.py (正式处理)
```

### 监控运行
```
1. tail -f generation.log (查看日志)
2. check_progress.py (查看进度)
```

### 遇到问题
```
1. 查看 generation.log
2. 阅读 README.md 的故障排查部分
3. 运行 test_setup.py 重新验证
```

## 📊 数据统计

- **输入**: 905个JSONL文件 @ `/data/extracted_apis/`
- **输出**: 905个JSONL文件 @ `/data/generated_questions/`
- **预估词条**: ~90,500个
- **预估时间**: ~12.6小时（10并发）

## 🎯 核心特性

✓ 异步并发处理  
✓ 智能速率控制  
✓ 自动重试机制  
✓ 断点续传支持  
✓ 详细日志记录  
✓ 进度监控工具  

## 📞 快速帮助

- **如何开始？** → 阅读 QUICKSTART.md
- **详细说明？** → 阅读 README.md  
- **项目概览？** → 阅读 PROJECT_SUMMARY.md
- **测试是否正常？** → 运行 `python3 test_setup.py`
- **先小规模测试？** → 运行 `python3 test_single_file.py`
- **查看进度？** → 运行 `python3 check_progress.py`
- **查看日志？** → `tail -f generation.log`

---

**当前状态**: ✅ 项目已完成，所有测试通过，可以开始运行

