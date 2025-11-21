#!/bin/bash
# 快速检查当前速度和预估完成时间

echo "======================================"
echo "速度统计 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"
echo ""

# 总量统计
TOTAL_INPUT=391610
CURRENT=$(cat /data/generated_questions/*.jsonl 2>/dev/null | wc -l)
REMAINING=$((TOTAL_INPUT - CURRENT))
PERCENT=$(awk "BEGIN {printf \"%.1f\", ($CURRENT/$TOTAL_INPUT)*100}")

echo "📊 总体进度:"
echo "  总输入: 391,610 条"
echo "  已完成: $CURRENT 条 ($PERCENT%)"
echo "  剩余: $REMAINING 条"
echo ""

# 文件统计
FILES_DONE=$(ls -1 /data/generated_questions/*.jsonl 2>/dev/null | wc -l)
echo "📁 文件进度: $FILES_DONE / 905"
echo ""

# 查找最近一次重启时间
echo "🕐 最近启动时间:"
grep "Found 905 JSONL files" generation.log | tail -1 | awk '{print $1, $2}'
START_TIME=$(grep "Found 905 JSONL files" generation.log | tail -1 | awk '{print $1, $2}')
echo ""

# 计算从启动到现在的速度
echo "⚡ 当前速度分析:"
echo "  查看最近10分钟的Progress记录..."
tail -500 generation.log | grep "Progress:" | tail -10
echo ""

# 429错误统计
ERROR_429=$(grep -c "429\|rate_limit" generation.log 2>/dev/null)
TIMEOUT=$(grep -c "timed out\|Timeout" generation.log 2>/dev/null)
echo "⚠️  错误统计:"
echo "  429错误: $ERROR_429 次"
echo "  超时错误: $TIMEOUT 次"
echo ""

# 最近完成的文件
echo "✅ 最近完成的文件:"
tail -100 generation.log | grep "Completed.*questions generated" | tail -5
echo ""

echo "======================================"
echo "💡 速度分析提示:"
echo "  - 观察Progress的数字增长"
echo "  - 如果每10分钟增长 > 1000，速度很好"
echo "  - 如果每10分钟增长 < 500，考虑优化"
echo "======================================"

