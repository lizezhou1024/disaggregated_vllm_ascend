
# Disaggregated Prefill-Decode vLLM Ascend 

## Language / 语言选择

- [English Documentation](./README_EN.md)

---

# 分离式预填充解码vLLM昇腾版

## 项目简介

本项目是一个在华为昇腾NPU上运行的分离式预填充-解码vLLM实现。该项目支持在昇腾NPU设备上高效运行大语言模型推理，具有以下特性：

### 主要特性
- **NPU支持**: 专为华为昇腾NPU优化，充分利用NPU的计算能力
- **分离式架构**: 采用分离式预填充-解码架构，将模型推理过程分为预填充（Prefill）和解码（Decode）两个阶段
- **张量并行**: 支持张量并行技术，提高模型推理效率
- **预填充解码分离**: 支持预填充和解码阶段的分离部署和优化

### 默认配置
- **总卡数**: 8张NPU卡（tensor-parallel-size=8）
- **预填充卡数**: 4张NPU卡专门用于预填充阶段
- **解码卡数**: 4张NPU卡专门用于解码阶段

这种架构设计能够更好地利用硬件资源，提高推理吞吐量和响应速度，特别适合大规模语言模型的在线推理服务。

# 构建环境
如果您的环境是昇腾910B可以使用已经配置好的docker环境，环境的依赖项：vllm-ascend: v0.7.3-dev
```
docker pull lianzizhou1024/disaggregated_vllm_ascend
```

## 从源码构建

### 安装Conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p 
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
conda init bash  
conda --version

# 创建conda环境
conda create -n vllm python=3.9
conda activate vllm
```

### 安装CANN 8.0.RC3

```
pip3 install attrs numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py wheel typing_extensions
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run"
sh Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run --full

source /data/home/2301111947/Ascend/ascend-toolkit/set_env.sh

wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 8.0.RC3/Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run"
sh Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run --install
```

### 安装预发布版torch-npu
```
mkdir pta
cd pta
wget https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.5.1/20250308.3/pytorch_v2.5.1_py39.tar.gz
tar -xvf pytorch_v2.5.1_py39.tar.gz
pip3 install ./torch_npu-2.5.1.dev20250308-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_x86_64.whl
```

### 安装vLLM

```
# 安装vLLM

cd disaggregated-pd-vllm-ascend
VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

# 安装vLLM Ascend

cd ascend-vllm-plugin
pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
```

### 下载模型

```
pip install -U huggingface_hub  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir ./models/DeepSeek-R1-Distill-Qwen-32B
```

## 构建Docker环境

### 拉取镜像

```
docker pull quay.io/ascend/vllm-ascend:v0.7.3-dev
```

### 运行Docker镜像

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

### 测试vLLM

```
time curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.7
  }'
```

### 测试分离式功能

```
cd disaggregated-pd-vllm-ascend
# 启动在线服务器
# 默认4张NPU用于预填充，4张NPU用于解码
bash pd_example/disaggregated_prefill.sh
# 测试服务器
bash pd_example/test_pd_online.sh
# 停止服务器
bash pd_example/kill_server.sh
```

## 性能对比

下表展示了分离式预填充解码架构（PD分离版）与传统vLLM张量并行（TP=8）的性能对比：

**测试模型**: DeepSeek-R1-Distill-Qwen-32B

| 版本 | 基准测试时长 (s) | 平均TTFT (ms) | 中位数TTFT (ms) | P99 TTFT (ms) | 平均TPOT (ms) | 中位数TPOT (ms) | P99 TPOT (ms) | 平均ITL (ms) | 中位数ITL (ms) | P99 ITL (ms) | 请求吞吐量 (req/s) | 输出令牌吞吐量 (tok/s) | 总令牌吞吐量 (tok/s) |
|------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|-----------------|------------------|-----------------| 
| PD 分离版 | 101.51 | 293.34 | 294.72 | 454.66 | 137.62 | 131.97 | 258.57 | 115.40 | 99.60 | 331.31 | 1.97 | 421.38 | 847.74 |
| vLLM TP=8 | 123.21 | 292.91 | 288.79 | 446.85 | 203.34 | 189.29 | 326.60 | 162.39 | 134.57 | 644.48 | 1.62 | 348.55 | 699.81 |
| 百分比差异 | -17.61% | +0.15% | +2.05% | +1.75% | -32.32% | -30.28% | -20.83% | -28.94% | -25.99% | -48.59% | +21.60% | +20.89% | +21.14% |

### 性能指标说明
- **TTFT (Time To First Token)**: 从请求开始到生成第一个令牌的时间
- **TPOT (Time Per Output Token)**: 每个输出令牌的生成时间
- **ITL (Inter-Token Latency)**: 令牌间延迟

### 关键优势
- **吞吐量提升**: 请求吞吐量提升21.60%，总令牌吞吐量提升21.14%
- **延迟降低**: 每个令牌生成时间减少32.32%，令牌间延迟减少28.94%
- **效率提升**: 基准测试完成时间减少17.61%

