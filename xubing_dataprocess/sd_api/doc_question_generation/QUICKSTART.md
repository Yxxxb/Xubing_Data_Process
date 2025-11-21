# 快速启动指南

## 1. 安装依赖

```bash
cd /home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation
pip install -r requirements.txt
```

## 2. 验证设置

```bash
python3 test_setup.py
```

## 3. 测试单个文件（推荐第一次运行）

```bash
python3 test_single_file.py
```

这将只处理第一个文件的前3个词条，验证API和配置是否正常工作。

## 4. 运行完整处理

```bash
# 前台运行
python3 generate_questions.py

# 后台运行（推荐）
nohup python3 generate_questions.py > output.log 2>&1 &
```

## 5. 监控进度

```bash
# 查看日志
tail -f generation.log

# 查看进度统计
python3 check_progress.py
```

## 6. 查看结果

生成的问题保存在：`/data/generated_questions/`

```bash
# 查看第一个结果
head -1 /data/generated_questions/*.jsonl | python3 -m json.tool
```

详细文档请查看 README.md 和 PROJECT_SUMMARY.md
