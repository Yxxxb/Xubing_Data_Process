# API Documentation Question Generation

自动从Python官方文档的API词条生成编程问题。

## 概述

该项目使用GPT-5模型根据Python API文档自动生成编程评估问题。系统支持高效的异步并发处理，同时控制API请求速率以避免超限。

## 功能特性

- **异步并发处理**: 使用asyncio和aiohttp实现高效的异步处理
- **速率限制**: 内置令牌桶算法控制API请求速率
- **重试机制**: 自动重试失败的请求，提高成功率
- **断点续传**: 支持从中断处继续处理
- **详细日志**: 记录处理进度和错误信息
- **统计信息**: 实时显示处理统计

## 目录结构

```
doc_question_generation/
├── generate_questions.py  # 主程序
├── requirements.txt       # Python依赖
├── run.sh                # 启动脚本
├── check_progress.py     # 进度检查工具
├── README.md             # 说明文档
├── generation.log        # 运行日志（运行后生成）
└── prompt_scene.txt      # Prompt模板
```

## 数据结构

### 输入数据 (/data/extracted_apis/*.jsonl)

每行包含一个API词条的JSON对象：

```json
{
  "library": "absl",
  "object": "absl.app.Error",
  "kind": "class",
  "signature": "(builtins.Exception)",
  "summary": "class Error(builtins.Exception)",
  "module": "absl.app",
  "[Code Snippet]": "...",
  "[Library Api Requirements]": "...",
  "[Library Api Doc]": "..."
}
```

### 输出数据 (/data/generated_questions/*.jsonl)

每行包含生成的问题：

```json
{
  "library": "absl",
  "object": "absl.app.Error",
  "kind": "class",
  "module": "absl.app",
  "signature": "(builtins.Exception)",
  "timestamp": "2025-10-24T10:30:00",
  "success": true,
  "question": "生成的完整问题内容..."
}
```

## 配置参数

在 `generate_questions.py` 中可以调整以下参数：

- `MAX_CONCURRENT_REQUESTS`: 最大并发请求数（默认10）
- `REQUESTS_PER_MINUTE`: 每分钟请求限制（默认50）
- `RETRY_ATTEMPTS`: 重试次数（默认3）
- `RETRY_DELAY`: 重试延迟秒数（默认2）

## 安装

```bash
cd /home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 方法1：使用启动脚本

```bash
chmod +x run.sh
./run.sh
```

### 方法2：直接运行Python脚本

```bash
python3 generate_questions.py
```

### 检查进度

运行过程中可以在另一个终端检查进度：

```bash
python3 check_progress.py
```

或者查看日志文件：

```bash
tail -f generation.log
```

## 性能考虑

1. **并发控制**: 默认最多10个并发请求，可根据API限制调整
2. **速率限制**: 使用令牌桶算法平滑控制请求速率
3. **断点续传**: 程序会跳过已处理的文件，可以随时中断和恢复
4. **错误处理**: 单个词条失败不影响其他词条的处理

## 预估处理时间

假设：
- 1000个文档文件
- 每个文档平均100个词条
- 每个词条处理时间约5秒（包括API调用和等待）

总词条数: 100,000
并发数: 10

预估时间: 100,000 * 5s / 10 = 50,000s ≈ 14小时

实际时间可能因API响应速度和网络状况有所不同。

## 故障排查

### 问题: API请求失败率高

- 检查API key是否有效
- 降低 `MAX_CONCURRENT_REQUESTS`
- 降低 `REQUESTS_PER_MINUTE`
- 增加 `RETRY_DELAY`

### 问题: 处理速度慢

- 增加 `MAX_CONCURRENT_REQUESTS`（注意不要超过API限制）
- 检查网络连接

### 问题: 内存占用高

- 当前实现一次性读取整个文件到内存
- 对于超大文件，可以修改代码使用流式处理

## 输出示例

```
2025-10-24 10:30:00 - INFO - Found 1000 JSONL files to process
2025-10-24 10:30:00 - INFO - Resuming: 1000 files remaining
2025-10-24 10:30:01 - INFO - Processing file: absl-py.jsonl
2025-10-24 10:30:01 - INFO - Found 45 entries in absl-py.jsonl
2025-10-24 10:30:15 - INFO - Progress: 10 processed, 10 success, 0 failed
2025-10-24 10:30:30 - INFO - Progress: 20 processed, 20 success, 0 failed
...
2025-10-24 10:35:00 - INFO - Completed absl-py.jsonl: 45 questions generated
============================================================
Processing complete!
Total entries processed: 45
Successful: 45
Failed: 0
Success rate: 100.00%
============================================================
```

## 注意事项

1. 确保有足够的API配额
2. 建议在稳定的网络环境下运行
3. 处理大量数据时建议使用screen或tmux等工具在后台运行
4. 定期检查日志文件以监控处理状态
5. 输出目录会自动创建，无需手动创建

## 技术栈

- Python 3.7+
- asyncio: 异步编程
- aiohttp: 异步HTTP客户端
- aiofiles: 异步文件操作
- OpenAI API: GPT-5模型

