

# Enable strict error handling: exit on error (-e) and print commands (-x)
set -xe

# Display warning about experimental feature
echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# Set up signal handler to catch Ctrl+C (SIGINT) for graceful shutdown
trap 'cleanup' INT

# Cleanup function to terminate all Python processes when script is interrupted
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Kill all Python processes to clean up vLLM instances
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

# Export the host IP address for vLLM distributed communication
# This gets the first IP address from the hostname command output
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# Check if Quart (async web framework) is installed, install if missing
# Quart is required for the disaggregated prefill proxy server
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# Function to wait for vLLM server to become ready
# Takes a port number as parameter and polls the health endpoint
wait_for_server() {
  local port=$1
  # Wait up to 1200 seconds (20 minutes) for server to start
  # Continuously check if the completions endpoint is accessible
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# Configuration note: You can adjust --kv-ip and --kv-port for distributed inference
# across multiple machines if needed

# ====================================================================
# Start the PREFILL INSTANCE (Producer)
# ====================================================================
# Uses NPU devices 0-3 (4 cards) for the prefill stage
# This instance handles initial token processing and KV cache generation

ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8100 \
   --tensor-parallel-size 4 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_buffer_size":5e9,"peer_world_size":2,"peer_kv_parallel_size":2}' &

# ====================================================================
# Start DECODE INSTANCES (Consumers)
# ====================================================================
# Two decode instances using NPU devices 4-7 (4 cards total, 2 per instance)
# These instances handle token generation using KV cache from prefill stage

# First decode instance: Uses NPU devices 4-5 (2 cards), serves on port 8200
# kv_rank=0 indicates this is the first decode worker
ASCEND_RT_VISIBLE_DEVICES=4 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8200 \
   --tensor-parallel-size 2 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9,"peer_world_size":4,"peer_kv_parallel_size":1}' &

# Second decode instance: Uses NPU devices 6-7 (2 cards), serves on port 8201
# kv_rank=1 indicates this is the second decode worker
ASCEND_RT_VISIBLE_DEVICES=6 vllm serve /models/DeepSeek-R1-Distill-Qwen-32B \
   --port 8201 \
   --tensor-parallel-size 2 \
   --max-model-len 10240 \
   --kv-transfer-config \
   '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9,"peer_world_size":4,"peer_kv_parallel_size":1}' &

# ====================================================================
# Wait for all instances to be ready
# ====================================================================
# Wait for prefill instance (port 8100) and first decode instance (port 8200) to start
# Note: We only check these two as they represent the primary endpoints
wait_for_server 8100
wait_for_server 8200

# ====================================================================
# Start the Proxy Server
# ====================================================================
# Launch a proxy server that provides a unified interface on port 8000
# 
# Proxy workflow:
# 1. Receives client requests on port 8000
# 2. Sends request to prefill instance (port 8100) with max_tokens=1
#    This generates the KV cache for the input tokens
# 3. After prefill completes, forwards the request to decode instances
#    (ports 8200/8201) which use the KV cache to generate output tokens
# 4. Returns the final response to the client
#
# NUM_DECODE=2: Specifies there are 2 decode instances to load balance between
# 
# NOTE: This API usage is experimental and subject to change.
# Future versions will introduce "vllm connect" command for easier
# connection management between prefill and decode instances.
NUM_DECODE=2 python3 /vllm-workspace/vllm/benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &

# Optional sleep to allow proxy server to initialize
# sleep 5

echo "🚀 Disaggregated prefill-decode online server started successfully!"
echo "📡 Service available at: http://localhost:8000"
echo "🔧 Prefill instance: http://localhost:8100"
echo "🔧 Decode instances: http://localhost:8200, http://localhost:8201"
echo "💡 Use Ctrl+C to gracefully shutdown all instances"


