# LLM serving with prefill-decode disaggregation on Ascend
Adatped from [vllm-project/vllm](https://github.com/vllm-project/vllm).

## Environment
- Server: 115.27.161.17 (910B * 8)
- Docker container: vllm
- Dependency:
  - vllm-ascend: v0.7.3-dev


## Run vLLM
`vllm serve --tensor-parallel-size 2 /models/Qwen2-7B-Instruct`

## 测试
```
time curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2-7B-Instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.7
  }'
```
