
# Disaggregated pd vllm ascend
 

# Build Environment
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

### Test vllm

```
time curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.7
  }'
```

### Test Disaggreagted Feature

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