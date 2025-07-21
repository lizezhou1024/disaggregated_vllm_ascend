#!/bin/bash

MODEL_PATH="/models/Qwen2-7B-Instruct"
API_URL="http://localhost:9000/v1/completions"
MAX_TOKENS=10
TEMPERATURE=0
CONCURRENCY=4
REPEAT=3

total_ttft=0
total_tpot=0
count=0
declare -a ttft_array
declare -a tpot_array

# 函数：发送请求并测量性能
run_test() {
    local prompt="$1"
    local iteration="$2"
    
    start_time=$(date +%s.%N)
    response=$(curl -X POST -s "$API_URL" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_PATH"'",
            "prompt": "'"$prompt"'",
            "max_tokens": '"$MAX_TOKENS"',
            "temperature": '"$TEMPERATURE"'
        }')
    end_time=$(date +%s.%N)

    total_time=$(echo "$end_time - $start_time" | bc)
    ttft=$(echo "$response" | jq -r '.usage.first_token_time_ms' | awk '{print $1 / 1000}')
    generated_tokens=$(echo "$response" | jq -r '.usage.completion_tokens')

    # 计算 TPOT（避免除零）
    if [ "$generated_tokens" -gt 1 ]; then
        tpot=$(echo "scale=4; ($total_time - $ttft) / ($generated_tokens - 1)" | bc)
    else
        tpot=0  # 如果生成的 token <= 1，TPOT 无意义，设为 0
    fi

    # 记录数据（跳过无效数据）
    if [ -n "$ttft" ] && [ -n "$tpot" ]; then
        echo "[Test $iteration] Prompt: \"$prompt\""
        echo "  - TTFT: ${ttft}s | TPOT: ${tpot}s/token | Total: ${total_time}s | Tokens: ${generated_tokens}"
        
        total_ttft=$(echo "$total_ttft + $ttft" | bc)
        total_tpot=$(echo "$total_tpot + $tpot" | bc)
        ttft_array+=("$ttft")
        tpot_array+=("$tpot")
        ((count++))
    else
        echo "⚠️  Invalid data for prompt: \"$prompt\" (skipped)"
    fi
}

# 主测试流程
echo "🚀 Starting batch test (Concurrency: $CONCURRENCY, Repeat: $REPEAT)"
while read -r prompt; do
    for ((i=1; i<=REPEAT; i++)); do
        run_test "$prompt" "$i" &
        # 控制并发数
        if (( $(jobs -r -p | wc -l) >= CONCURRENCY )); then
            wait -n
        fi
    done
done < prompts.txt
wait

# 检查是否有有效数据
if [ "$count" -eq 0 ]; then
    echo "❌ No valid data collected. Check server or prompts."
    exit 1
fi

# 计算统计数据
avg_ttft=$(echo "scale=4; $total_ttft / $count" | bc)
avg_tpot=$(echo "scale=4; $total_tpot / $count" | bc)

# 计算百分位数（仅当数组非空时）
if [ "${#ttft_array[@]}" -gt 0 ]; then
    p90_ttft=$(printf "%s\n" "${ttft_array[@]}" | sort -n | datamash perc 90 2>/dev/null || echo "N/A")
    p95_ttft=$(printf "%s\n" "${ttft_array[@]}" | sort -n | datamash perc 95 2>/dev/null || echo "N/A")
else
    p90_ttft="N/A"
    p95_ttft="N/A"
fi

if [ "${#tpot_array[@]}" -gt 0 ]; then
    p90_tpot=$(printf "%s\n" "${tpot_array[@]}" | sort -n | datamash perc 90 2>/dev/null || echo "N/A")
    p95_tpot=$(printf "%s\n" "${tpot_array[@]}" | sort -n | datamash perc 95 2>/dev/null || echo "N/A")
else
    p90_tpot="N/A"
    p95_tpot="N/A"
fi

# 输出最终报告
echo -e "\n📊 Final Performance Report"
echo "================================="
echo "Total Valid Requests: $count"
echo "Average TTFT: ${avg_ttft}s"
echo "Average TPOT: ${avg_tpot}s/token"
echo "P90 TTFT: ${p90_ttft}"
echo "P95 TTFT: ${p95_ttft}"
echo "P90 TPOT: ${p90_tpot}"
echo "P95 TPOT: ${p95_tpot}"
echo "================================="