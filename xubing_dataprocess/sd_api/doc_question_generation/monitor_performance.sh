#!/bin/bash
# 性能监控脚本 - 实时显示速度和错误

echo "======================================"
echo "性能监控 (每10秒更新)"
echo "======================================"
echo ""

while true; do
    clear
    echo "======================================"
    echo "实时性能监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"
    echo ""
    
    # 统计完成的文件
    COMPLETED=$(ls -1 /data/generated_questions/*.jsonl 2>/dev/null | wc -l)
    TOTAL=905
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL)*100}")
    
    echo "📁 文件进度: $COMPLETED / $TOTAL ($PERCENT%)"
    echo ""
    
    # 统计生成的条目
    ENTRIES=$(cat /data/generated_questions/*.jsonl 2>/dev/null | wc -l)
    echo "📝 已生成: $ENTRIES 条"
    echo ""
    
    # 最近的进度
    echo "🔄 最近进度:"
    tail -5 generation.log | grep "Progress:" | tail -1
    echo ""
    
    # 429错误统计
    RATE_LIMIT_ERRORS=$(grep -c "429\|rate_limit" generation.log 2>/dev/null)
    echo "⚠️  429错误: $RATE_LIMIT_ERRORS 次"
    
    # 超时错误统计
    TIMEOUT_ERRORS=$(grep -c "timed out\|Timeout" generation.log 2>/dev/null)
    echo "⏱️  超时错误: $TIMEOUT_ERRORS 次"
    echo ""
    
    # 最近的文件
    echo "📄 最近处理的文件:"
    tail -50 generation.log | grep "Processing file:" | tail -3
    echo ""
    
    # 计算速度（最近10分钟）
    TEN_MIN_AGO=$(date -d '10 minutes ago' '+%Y-%m-%d %H:%M:%S')
    RECENT_ENTRIES=$(grep "$TEN_MIN_AGO" generation.log -A 100000 | grep "Progress:" | tail -1 | grep -oP '\d+ success' | grep -oP '\d+')
    if [ ! -z "$RECENT_ENTRIES" ]; then
        SPEED=$(awk "BEGIN {printf \"%.1f\", $RECENT_ENTRIES/10}")
        echo "⚡ 当前速度: ~$SPEED 条/分钟"
    fi
    echo ""
    
    echo "======================================"
    echo "按 Ctrl+C 退出监控"
    echo "======================================"
    
    sleep 10
done

