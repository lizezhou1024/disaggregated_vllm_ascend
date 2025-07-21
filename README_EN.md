# Disaggregated Prefill-Decode vLLM Ascend

## Project Overview

This project is a Disaggregated Prefill-Decode vLLM implementation running on Huawei Ascend NPUs. The project enables efficient large language model inference on Ascend NPU devices with the following features:

### Key Features
- **NPU Support**: Optimized specifically for Huawei Ascend NPUs, fully utilizing NPU computational capabilities
- **Disaggregated Architecture**: Adopts a disaggregated prefill-decode architecture, separating model inference into Prefill and Decode stages
- **Tensor Parallelism (TP)**: Supports tensor parallelism technology to improve model inference efficiency
- **Prefill-Decode Separation (PD)**: Supports separate deployment and optimization of prefill and decode stages

### Default Configuration
- **Total NPUs**: 8 NPU cards (tensor-parallel-size=8)
- **Prefill NPUs**: 4 NPU cards dedicated to the prefill stage
- **Decode NPUs**: 4 NPU cards dedicated to the decode stage

This architectural design enables better hardware resource utilization, improved inference throughput and response speed, making it particularly suitable for large-scale language model online inference services.

# Build Environment
You can use the docker image if your environment is Ascend 910B, which comes with pre-configured environment dependencies: vllm-ascend: v0.7.3-dev
```
docker pull lianzizhou1024/disaggregated_vllm_ascend
```

## Build from source

### Install Conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p 
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
conda init bash  
conda --version

# create conda
conda create -n vllm python=3.9
conda activate vllm
```

### Install CANN 8.0.RC3

```
pip3 install attrs numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run"
sh Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run --full

source /data/home/2301111947/Ascend/ascend-toolkit/set_env.sh

wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 8.0.RC3/Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run"
sh Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run --install
```

### Install pre-release torch-npu
```
mkdir pta
cd pta
wget https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.5.1/20250308.3/pytorch_v2.5.1_py39.tar.gz
tar -xvf pytorch_v2.5.1_py39.tar.gz
pip3 install ./torch_npu-2.5.1.dev20250308-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_x86_64.whl
```

### Install vLLM

```
# Install vLLM

cd disaggregated-pd-vllm-ascend
VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

# Install vLLM Ascend

cd ascend-vllm-plugin
pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
```

### Download Model

```
pip install -U huggingface_hub  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir ./models/DeepSeek-R1-Distill-Qwen-32B
```

## Build Docker Environment

### Pull Image

```
docker pull quay.io/ascend/vllm-ascend:v0.7.3-dev
```

### Run Docker Image

```
sudo docker run -d \
    --name vllm \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --volume=/usr/local/dcmi:/usr/local/dcmi \
    --volume=/usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    --volume=/usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    --volume=/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    --volume=/etc/ascend_install.info:/etc/ascend_install.info \
    --volume=/home/[youruser]/models:/models \
    --volume=/home/[youruser]/disaggregated-pd-vllm-ascend/vllm:/usr/local/python3.10/lib/python3.10/site-packages/vllm \
    --volume=/home/[youruser]/ascend-vllm-plugin/vllm_ascend:/usr/local/python3.10/lib/python3.10/site-packages/vllm-ascend \
    quay.io/ascend/vllm-ascend:v0.7.3-dev
```
```
vllm serve --tensor-parallel-size 2 /models/DeepSeek-R1-Distill-Qwen-32B
```

### Test vLLM

```
time curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.7
  }'
```

### Test Disaggregated Feature

```
cd disaggregated-pd-vllm-ascend
# Start Online Server
# Default 4 npu for prefill and 4 npu for decode
bash pd_example/disaggregated_prefill.sh
# Test Server
bash pd_example/test_pd_online.sh
# Kill Server
bash pd_example/kill_server.sh
```

## Performance Comparison

The following table shows the performance comparison between Disaggregated Prefill-Decode architecture (PD Version) and traditional vLLM Tensor Parallelism (TP=8):

**Test Model**: DeepSeek-R1-Distill-Qwen-32B

| Version | Benchmark Duration (s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) | Request Throughput (req/s) | Output Token Throughput (tok/s) | Total Token Throughput (tok/s) |
|---------|----------------------|----------------|------------------|---------------|----------------|------------------|---------------|---------------|-----------------|--------------|---------------------------|--------------------------------|-------------------------------|
| PD Version | 101.51 | 293.34 | 294.72 | 454.66 | 137.62 | 131.97 | 258.57 | 115.40 | 99.60 | 331.31 | 1.97 | 421.38 | 847.74 |
| vLLM TP=8 | 123.21 | 292.91 | 288.79 | 446.85 | 203.34 | 189.29 | 326.60 | 162.39 | 134.57 | 644.48 | 1.62 | 348.55 | 699.81 |
| Percentage Difference | -17.61% | +0.15% | +2.05% | +1.75% | -32.32% | -30.28% | -20.83% | -28.94% | -25.99% | -48.59% | +21.60% | +20.89% | +21.14% |

### Performance Metrics Explanation
- **TTFT (Time To First Token)**: Time from request start to generating the first token
- **TPOT (Time Per Output Token)**: Time to generate each output token
- **ITL (Inter-Token Latency)**: Latency between tokens

### Key Advantages
- **Throughput Improvement**: 21.60% increase in request throughput, 21.14% increase in total token throughput
- **Latency Reduction**: 32.32% reduction in time per output token, 28.94% reduction in inter-token latency
- **Efficiency Gain**: 17.61% reduction in benchmark completion time

## References

The disaggregated prefill-decode architecture design of this project is inspired by the following research:

- **DistServe**: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://github.com/LLMServe/DistServe)
