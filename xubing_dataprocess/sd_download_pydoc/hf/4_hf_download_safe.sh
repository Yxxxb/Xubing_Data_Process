#!/bin/bash

# 安全版本：避免429错误
# 特点：
# - 只用1个线程（最保守）
# - 延迟1.5秒（避免触发限流）
# - 自动断点续传
# - 只下载最新版本文档（跳过 v4.57.0 等旧版本）
# - 无限制页数（会下载所有transformers文档）

python hf_docs_crawler_fixed.py \
  --out /pfs/training-data/xubingye/data/code_doc/hf \
  --max-pages 0 \
  --concurrency 1 \
  --delay 0.5 \
  --langs en \
  --includes datasets diffusers tokenizers accelerate peft trl optimum evaluate hub

echo ""
echo "✅ 已完成下载（只下载最新版本，跳过了旧版本如 v4.57.0 等）"
echo ""
echo "💡 提示："
echo "  - 如果中途中断（Ctrl+C），再次运行此脚本会自动从断点继续"
echo "  - 如果还遇到429错误，可以增加 --delay 到 3.0 或更高"
echo "  - 如果想下载所有版本，添加 --all-versions 参数"
