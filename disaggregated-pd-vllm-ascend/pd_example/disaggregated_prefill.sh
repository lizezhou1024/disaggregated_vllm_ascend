#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer --gpu-memory-utilization 0.8 \    --disable-async-output-proc \ 
# vllm serve /models/DeepSeek-R1-Distill-Qwen-7B \
# ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /models/Qwen2-7B-Instruct \
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8100 \
   --tensor-parallel-size 4 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_buffer_size":5e9,"peer_world_size":2,"peer_kv_parallel_size":2}' &

# decoding instance, which is the KV consumer --gpu-memory-utilization 0.8 \ --max-model-len 100
# vllm serve /models/DeepSeek-R1-Distill-Qwen-7B \
# ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /models/Qwen2-7B-Instruct \
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8200 \
   --tensor-parallel-size 2 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9,"peer_world_size":4,"peer_kv_parallel_size":1}' &

# ASCEND_RT_VISIBLE_DEVICES=6 vllm serve /models/Qwen2-7B-Instruct \
ASCEND_RT_VISIBLE_DEVICES=6 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8201 \
   --tensor-parallel-size 2 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9,"peer_world_size":4,"peer_kv_parallel_size":1}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens 
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between prefill and decode instances
NUM_DECODE=2 python3 /vllm-workspace/vllm/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
# sleep 5
echo "online server started !!!!"

# # serve two example requests
# prompt1="San Francisco is a"
# prompt2="peking university is the most"
# # Start the vLLM server with the specified model
# output1=$(curl -X POST -s http://localhost:9000/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "/models/Qwen2-7B-Instruct",
# "prompt": "'"$prompt1"'",
# "max_tokens": 10,
# "temperature": 0
# }')

# output2=$(curl -X POST -s http://localhost:9000/v1/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "/models/Qwen2-7B-Instruct",
# "prompt": "'"$prompt2"'",
# "max_tokens": 10,
# "temperature": 0
# }')

# text1=$(echo "$output1" | jq -r '.choices[0].text')
# text2=$(echo "$output2" | jq -r '.choices[0].text')

# # Cleanup commands
# # Cleanup commands
# # pgrep python | xargs kill -9
# # pkill -f python
# echo ""
# sleep 1

# # Print the outputs of the curl requests
# echo ""
# echo "input of first request: $prompt1"
# echo "result of first request: $prompt1$text1"
# echo "Output of first request: $output1"

# echo "Input of first request: $prompt2"
# echo "result of second request: $prompt2$text2"
# echo "Output of second request: $output2"
# echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
# echo ""
