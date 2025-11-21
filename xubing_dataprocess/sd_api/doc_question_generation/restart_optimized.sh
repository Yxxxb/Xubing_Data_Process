#!/bin/bash
# 重启优化版本的脚本

echo "========================================"
echo "重启并发文件处理版本"
echo "========================================"
echo ""

# 找到并停止旧进程
OLD_PID=$(ps aux | grep "python3 generate_questions.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$OLD_PID" ]; then
    echo "找到旧进程 PID: $OLD_PID"
    echo "停止旧进程..."
    kill $OLD_PID
    sleep 2
    
    # 确认已停止
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "强制停止..."
        kill -9 $OLD_PID
        sleep 1
    fi
    echo "✅ 旧进程已停止"
else
    echo "⚠️  未找到运行中的进程"
fi

echo ""
echo "启动优化版本（5个文件并发处理）..."
cd /home/xubing/code/MMDataKit/xubing_dataprocess/api_shanda/doc_question_generation

# 备份旧日志
if [ -f generation.log ]; then
    cp generation.log generation.log.backup_$(date +%Y%m%d_%H%M%S)
fi

# 启动新进程
nohup python3 generate_questions.py > generation.log 2>&1 &
NEW_PID=$!

sleep 2

# 确认启动成功
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ 新进程已启动！PID: $NEW_PID"
    echo ""
    echo "优化内容："
    echo "  - 同时处理 5 个文件（之前是串行处理）"
    echo "  - 预期速度提升：3-5倍"
    echo "  - 预期从 145/分钟 → 450-725/分钟"
    echo ""
    echo "监控命令："
    echo "  tail -f generation.log"
    echo "  ./watch_speed.sh"
    echo ""
else
    echo "❌ 启动失败！请检查日志"
    tail -20 generation.log
    exit 1
fi

echo "========================================"
echo "等待10秒后查看启动日志..."
echo "========================================"
sleep 10
tail -30 generation.log

