# AF3 NPU 推理服务

OpenAI 兼容的 AlphaFold 3 昇腾 NPU 推理服务（FastAPI）。基于 xfold PyTorch
实现 + torch_npu，绕过 vLLM（AF3 为固定输入→固定输出模型，PagedAttention /
KV Cache / Continuous Batching 不适用）。

背景与完整部署步骤见 `../AF3_Ascend_NPU_field_deployment_handbook.md`。

## 关键设计

- **两阶段进程布局**：预特征化 pkl 反序列化会拉起 `alphafold3.cpp`，与
  `torch_npu` 同进程加载后 pickle 会 segfault。启动时先（未 import
  torch_npu）把内置 test pkl 转成 npz；用户自定义 pkl 用干净子进程加载。
- **NPU 约束**：float64 全部转 float32（`aclnnMatmul` 不支持 DT_DOUBLE）；
  fastnn 三个实现强制 `torch`（无 triton）；**默认 fp32 精度**（昇腾上比
  bf16 更快、置信度更好，现场实测）。
- **推理串行化**：单卡模型全局锁保护，并发请求排队。

## 配置（环境变量，均有默认值）

| 变量 | 默认 | 说明 |
|------|------|------|
| `AF3_SRC` | `/home/alphafold3-3.0.4/src` | alphafold3 源码（数据管线 + cpp.so） |
| `XFOLD_SRC` | `/home/xfold/xfold-main` | xfold 源码 |
| `AF3_MODEL_DIR` | `/home` | 权重目录（af3.bin / af3.bin.zst + fourier npy） |
| `AF3_DEVICE` | `npu:0` | 设备，容器内单卡即 npu:0 |
| `AF3_PRECISION` | `fp32` | `fp32` 或 `bf16` |
| `AF3_NUM_RECYCLES` | `10` | 循环次数（冒烟可设 1） |
| `AF3_NUM_SAMPLES` | `5` | 扩散采样数 |
| `AF3_DIFFUSION_STEPS` | `200` | 扩散步数（冒烟可设 10–20） |
| `AF3_HOST` / `AF3_PORT` | `0.0.0.0` / `9800` | 监听地址 |
| `LIBCIFPP_DATA_DIR` | 手册默认 rsrc 目录 | libcifpp 资源目录 |

依赖：`fastapi`、`uvicorn`、`pydantic`（容器内 `pip3 install fastapi
uvicorn`），以及 xfold / alphafold3 全部运行依赖。

## 启动

```bash
source /home/af3_env.sh   # CANN / LD_LIBRARY_PATH / LIBCIFPP_DATA_DIR

# 生产（fp32，官方完整配置 10 recycle / 5 sample / 200 step）
nohup python3 service/af3_service.py > /home/af3_service.log 2>&1 &

# 快速冒烟
AF3_NUM_RECYCLES=1 AF3_NUM_SAMPLES=1 AF3_DIFFUSION_STEPS=10 \
  python3 service/af3_service.py
```

## API

| 端点 | 说明 |
|------|------|
| `GET /health` | 服务与模型状态（设备、精度、参数量） |
| `GET /v1/models` | OpenAI 模型列表 |
| `POST /v1/chat/completions` | OpenAI 兼容；`messages[-1].content` 传 **预特征化 pkl 路径** |
| `POST /predict` | `{"pkl_path": "..."}` |
| `POST /predict/test` | 用内置 test pkl 冒烟 |

```bash
curl -s http://localhost:9800/health
curl -s -X POST http://localhost:9800/predict/test

python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://<服务器IP>:9800/v1', api_key='none')
resp = client.chat.completions.create(
    model='alphafold3-3.0.4',
    messages=[{'role': 'user', 'content': '/path/to/featurised_example.pkl'}])
print(resp.choices[0].message.content)"
```

返回为张量摘要（shape/dtype/norm）与关键指标（`contact_probs_mean`
对 JAX 参考均值差应约 0.0011）。**注意：本服务只接受预特征化 pkl 路径，
不是原始氨基酸序列；完整 MSA 管线请走 xfold `run_alphafold.py`。**
