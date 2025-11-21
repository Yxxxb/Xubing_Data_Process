# 项目总结

## 项目概述

已成功创建一个高效的API文档问题生成系统，用于从Python官方文档自动生成编程评估问题。

## 数据规模

- **输入文件数**: 905个JSONL文件
- **预估总词条数**: ~90,500 (按每文件100词条估算)
- **输入目录**: `/data/extracted_apis/`
- **输出目录**: `/data/generated_questions/`

## 核心功能

### 1. 高效异步处理
- 使用asyncio实现并发处理
- 默认10个并发请求
- 支持大规模数据处理

### 2. 智能速率控制
- 令牌桶算法控制API请求速率
- 默认50请求/分钟
- 自动处理rate limit错误

### 3. 可靠性保障
- 自动重试机制（3次）
- 断点续传支持
- 详细错误日志

### 4. 监控和统计
- 实时进度日志
- 独立的进度检查工具
- 成功率统计

## 项目文件

```
doc_question_generation/
├── generate_questions.py   # 主程序 - 核心生成逻辑
├── check_progress.py       # 进度检查工具
├── test_setup.py          # 设置验证工具
├── run.sh                 # 启动脚本
├── requirements.txt       # Python依赖
├── README.md             # 完整文档
├── QUICKSTART.md         # 快速启动指南
├── PROJECT_SUMMARY.md    # 本文件
├── prompt_scene.txt      # GPT Prompt模板
└── generation.log        # 运行日志（运行后生成）
```

## 技术架构

### 异步架构
```
QuestionGenerator
    ├── RateLimiter (令牌桶速率控制)
    ├── Semaphore (并发控制)
    └── AsyncOpenAI (API客户端)
```

### 处理流程
```
读取JSONL文件
    ↓
并发处理每个词条
    ├── 构建Prompt
    ├── 调用GPT-4 API
    ├── 重试失败请求
    └── 记录结果
    ↓
写入输出文件
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_CONCURRENT_REQUESTS | 10 | 最大并发请求数 |
| REQUESTS_PER_MINUTE | 50 | 每分钟请求限制 |
| RETRY_ATTEMPTS | 3 | 失败重试次数 |
| RETRY_DELAY | 2s | 重试间隔 |

## 性能预估

### 处理时间计算

**基于实际数据:**
- 文件数: 905
- 预估总词条: ~90,500
- 每词条处理时间: ~5秒（包括API调用）
- 并发数: 10

**预估总时间:**
```
90,500 词条 × 5秒 / 10并发 = 45,250秒 ≈ 12.6小时
```

**优化后（并发20，速率100/分钟）:**
```
90,500 词条 × 5秒 / 20并发 = 22,625秒 ≈ 6.3小时
```

注意：实际时间可能因网络状况和API响应速度而变化。

## 快速开始

### 1. 验证设置
```bash
cd /home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation
python3 test_setup.py
```

### 2. 启动处理
```bash
# 方法1: 直接运行
python3 generate_questions.py

# 方法2: 使用shell脚本
./run.sh

# 方法3: 后台运行（推荐）
nohup python3 generate_questions.py > output.log 2>&1 &
```

### 3. 监控进度
```bash
# 实时日志
tail -f generation.log

# 进度统计
python3 check_progress.py
```

## 输出格式

每个输入文件对应一个输出文件，包含生成的问题：

```json
{
  "library": "absl",
  "object": "absl.app.Error",
  "kind": "class",
  "module": "absl.app",
  "signature": "(builtins.Exception)",
  "timestamp": "2025-10-24T10:30:00",
  "success": true,
  "question": "完整的生成问题内容..."
}
```

## 特性优势

### 1. 高效性
- 异步并发处理，充分利用等待时间
- 批量处理，减少overhead
- 智能速率控制，最大化吞吐量

### 2. 可靠性
- 自动重试失败请求
- 断点续传，支持中断恢复
- 单个失败不影响整体处理

### 3. 可维护性
- 清晰的代码结构
- 详细的日志记录
- 完善的错误处理

### 4. 可扩展性
- 易于调整并发参数
- 支持自定义Prompt模板
- 可以轻松添加新的处理逻辑

## 注意事项

1. **API配额**: 确保OpenAI API有足够配额（预估需要90,500次调用）
2. **网络稳定**: 建议在稳定网络环境下运行
3. **磁盘空间**: 输出文件可能需要数GB空间
4. **监控运行**: 使用screen/tmux在后台长时间运行
5. **成本考虑**: 每次API调用有成本，请预估总费用

## API成本估算

假设使用GPT-4 API:
- 输入tokens: ~2000/请求 (包含prompt模板和文档)
- 输出tokens: ~4000/请求 (生成的问题)
- GPT-4价格: $0.03/1K输入tokens, $0.06/1K输出tokens

**总成本估算:**
```
90,500请求 × (2K输入 × $0.03/K + 4K输出 × $0.06/K)
= 90,500 × ($0.06 + $0.24)
= 90,500 × $0.30
= $27,150
```

**注意**: 这是粗略估算，实际成本可能因token数量变化而不同。建议先用小批量测试。

## 故障处理

### 常见问题

1. **Rate Limit错误** → 降低并发数和请求频率
2. **Timeout错误** → 检查网络连接，增加重试次数
3. **API Key无效** → 检查密钥是否正确和有效
4. **内存不足** → 减少并发数

### 调试步骤

1. 检查 `generation.log` 查看错误详情
2. 运行 `test_setup.py` 验证配置
3. 使用单文件测试调试问题
4. 检查API配额和限制

## 成功标准

运行完成后，你应该看到:
- 905个输出文件在 `/data/generated_questions/`
- 约90,500条生成的问题记录
- 成功率 > 95%
- 详细的处理日志

## 后续步骤

处理完成后，可以:
1. 分析生成的问题质量
2. 提取和分类问题
3. 构建问题数据库
4. 进行人工审核和优化

## 联系和支持

- 查看日志: `tail -f generation.log`
- 检查进度: `python3 check_progress.py`
- 阅读文档: `cat README.md`
- 快速指南: `cat QUICKSTART.md`

---

**项目状态**: ✓ 已完成并通过测试

**准备就绪**: 可以开始运行生成任务

