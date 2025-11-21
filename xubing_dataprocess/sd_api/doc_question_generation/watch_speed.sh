#!/bin/bash
# 实时监控处理速度

echo "开始监控速度（每30秒更新一次）..."
echo "按Ctrl+C停止"
echo ""

while true; do
    clear
    echo "========================================"
    echo "实时速度监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # 最近1分钟的请求数
    one_min_ago=$(date -d '1 minute ago' '+%Y-%m-%d %H:%M')
    count_1min=$(grep "HTTP.*200 OK" generation.log 2>/dev/null | awk -v start="$one_min_ago" '$0 > start' | wc -l)
    echo "📊 最近1分钟: $count_1min 个请求"
    
    # 最近5分钟的请求数
    five_min_ago=$(date -d '5 minutes ago' '+%Y-%m-%d %H:%M')
    count_5min=$(grep "HTTP.*200 OK" generation.log 2>/dev/null | awk -v start="$five_min_ago" '$0 > start' | wc -l)
    avg_5min=$(echo "scale=1; $count_5min / 5" | bc)
    echo "📊 最近5分钟: $count_5min 个请求 (平均 $avg_5min/分钟)"
    
    # 最近10分钟的请求数
    ten_min_ago=$(date -d '10 minutes ago' '+%Y-%m-%d %H:%M')
    count_10min=$(grep "HTTP.*200 OK" generation.log 2>/dev/null | awk -v start="$ten_min_ago" '$0 > start' | wc -l)
    avg_10min=$(echo "scale=1; $count_10min / 10" | bc)
    echo "📊 最近10分钟: $count_10min 个请求 (平均 $avg_10min/分钟)"
    
    echo ""
    echo "🔗 活跃连接数: $(netstat -an 2>/dev/null | grep ESTABLISHED | grep -c ':443' || echo 'N/A')"
    
    echo ""
    echo "⚠️  最近错误:"
    tail -50 generation.log | grep -E "(429|502|ERROR)" | tail -3
    
    echo ""
    echo "========================================"
    echo "优化前速度: ~130-140/分钟"
    echo "期望速度: >200/分钟"
    echo "========================================"
    
    sleep 30
done
