# AlphaFold 3 昇腾 NPU 现场部署操作手册

> 本文档用于在客户现场的 Atlas 800I A2 服务器上从零部署 AlphaFold 3 推理服务。
> 按照以下步骤顺序操作即可完成部署。
>
> 踩过的坑都写在对应步骤和「故障排查」里。

---

## 一、适配方案概述

### 1.1 背景

AlphaFold 3（v3.0.4）是 DeepMind 开发的蛋白质结构预测模型，原始实现基于 JAX/Haiku 框架，运行在 NVIDIA GPU 上。模型权重 `af3.bin` 以 Haiku 自定义二进制格式存储。客户要求在华为昇腾 Atlas 800I A2 服务器（Ascend 910B4 NPU）上运行该模型，并提供 OpenAI 兼容的推理 API。

### 1.2 核心挑战

| 挑战 | 说明 |
|------|------|
| 框架不兼容 | JAX 在昇腾 NPU 上无原生支持；模型原始代码依赖 JAX/Haiku/tokamax |
| 权重格式不兼容 | `af3.bin` 是 Haiku 二进制格式，无法直接被 PyTorch 加载 |
| vLLM 不适用 | vLLM 的 PagedAttention / KV Cache / Continuous Batching 专为自回归 LLM 设计；AF3 是固定输入→固定输出的结构预测模型，架构完全不匹配 |
| C++ 扩展依赖 | alphafold3 数据管线依赖 `alphafold3.cpp`（mmCIF 解析、FASTA 迭代器、DSSP 等），编译需多个 GitHub C++ 依赖 |
| NPU 算子差异 | 昇腾 NPU 不支持 Triton 内核、不支持 float64；bfloat16 算子行为与 GPU 有差异 |

### 1.3 适配方案

**总体思路**：使用 xfold（AlphaFold 3 的 PyTorch 开源复现）作为模型实现，复用 vllm-ascend 容器中的 torch_npu + CANN 运行环境，绕过 vLLM 引擎，搭建独立的 FastAPI 推理服务。

```
┌─────────────────────────────────────────────────────────────┐
│ 适配方案架构                                                │
│                                                             │
│  原始方案 (GPU)                 迁移后方案 (NPU)             │
│  ┌─────────────┐               ┌──────────────────────┐     │
│  │ JAX/Haiku   │               │ xfold (PyTorch)      │     │
│  │ af3.bin     │   权重转换    │ import_jax_weights_  │     │
│  │ tokamax     │ ───────────→  │ af3.bin → PyTorch    │     │
│  │ GPU         │               │ torch_npu (NPU)      │     │
│  │ run_af3.py  │               │ FastAPI Server       │     │
│  └─────────────┘               └──────────────────────┘     │
│                                                             │
│  复用: vllm-ascend 容器 (torch_npu + CANN 环境)              │
│  绕过: vLLM 推理引擎 (不适用 AF3 架构)                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 各组件职责

| 组件 | 来源 | 作用 |
|------|------|------|
| **xfold** | GitHub 开源 (`xfold-main.zip`) | AF3 的完整 PyTorch 实现（Pairformer、Diffusion Transformer、AtomCrossAttention、ConfidenceHead、DistogramHead）。提供 `import_jax_weights_()` 直接解析 `af3.bin` 并映射到 PyTorch 参数 |
| **alphafold3-3.0.4** | GitHub `google-deepmind/alphafold3` | 数据管线（MSA / 模板 / 特征化）和 C++ 扩展。C++ 扩展编译为 `cpp.so` |
| **af3.bin** | DeepMind 官方权重 | 约 1.07 GB，约 405 个参数 key，约 3.68 亿参数。Haiku 自定义二进制 |
| **torch_npu** | vllm-ascend 容器内置 | PyTorch 昇腾后端（`aclnnMatmul`、`npu_layer_norm` 等） |
| **FastAPI 服务** | 自行编写 `af3_service.py` | OpenAI 兼容 API：`/v1/chat/completions`、`/predict` 等 |

### 1.5 权重转换原理

xfold 的 `params.py` 实现 Haiku 二进制 → PyTorch state_dict：

```
af3.bin (Haiku binary, 1.07 GB)
│
├─ get_alphafold3_params()     ← 解析二进制 record
│    格式: header(5 个 int32) + payload(scope+name+dtype+shape+bytes)
│    返回: {scope/name: tensor}
│
├─ get_translation_dict(model) ← 遍历 PyTorch 模型结构
│    LinearParams / LayerNormParams / LinearHMAParams /
│    TriMulParams / DiffusionTransformerParams / stacked()
│
└─ assign(translation, params) ← 逐参数 copy_()
```

转换后参数量与 Haiku 一致（约 3.68 亿）。`contact_probs` 与 JAX 参考均值差约 **0.0011**，用于确认权重映射正确。

### 1.6 NPU 算子适配

| 原始实现 (GPU) | NPU 适配 | 说明 |
|----------------|----------|------|
| tokamax Triton Flash Attention | `dot_product_attention_torch` | xfold fastnn 切到 `torch` |
| tokamax Triton GLU | 原生 PyTorch GLU | `F.linear` + `swish` |
| Triton LayerNorm | `torch.nn.LayerNorm` | 标准实现 |
| `device_type="cuda"` | `device_type="npu"` | autocast 设备类型 |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` | 同步 |
| float64 tensor | float32 | `aclnnMatmul` 不支持 `DT_DOUBLE` |

> 生产上 NPU 完整推理更推荐 **float32**（现场测过比 bf16 更快）。附录服务脚本仍用 bf16 autocast，与内网冒烟一致；正式压测可改为 fp32。

### 1.7 pkl 加载 Segfault 绕过

预特征化 `.pkl` 反序列化会拉起 `alphafold3.cpp`。`torch_npu` 与 `alphafold3.cpp` 同时加载后，pickle 某些 C++ 对象会 **segfault**。

**两阶段加载：**

1. **无 torch_npu**：加载 pkl → 抽出 numpy → 存 `.npz`
2. **有 torch_npu**：从 `.npz` 转 tensor 上 NPU

服务启动时先把内置 test_data 写成 npz（在 import torch_npu 之前）；用户自定义 pkl 用独立 Python 子进程加载。

### 1.8 精度验证结论

| 对比维度 | RMSD / 指标 | 说明 |
|----------|-------------|------|
| NPU PyTorch vs CPU PyTorch（同 seed） | **1.82 Å** | NPU 计算精度可用 |
| CPU PyTorch vs JAX 参考（同 seed） | 16.61 Å | 框架 RNG 差异（非 NPU 问题） |
| NPU PyTorch vs JAX 参考 | 14.82 Å | 两者叠加 |
| contact_probs（NPU vs JAX） | 均值差 0.0011 | 权重转换正确 |

约 15Å 的坐标差中约 88% 来自 PyTorch 与 JAX 的 RNG，与 NPU 硬件无关。

---

## 二、前置条件检查

### 2.1 确认服务器环境

```bash
# 确认 NPU 驱动正常
npu-smi info

# 确认有空闲 NPU（找一张没有进程占用的卡）
# 关注 "Process id" 列，空的即为空闲卡
# 记下空闲卡编号（如 NPU 5），后续用 /dev/davinci5

# 确认 Docker 镜像存在
docker images | grep vllm-ascend

# 容器内 Python 预期: 3.11.x
```

### 2.2 准备所需文件

提前放到服务器 `/home`：

| 文件 | 来源 | 大小 | 用途 |
|------|------|------|------|
| `af3.bin` | DeepMind 官方权重 | 1.07 GB | 模型权重 |
| alphafold3-3.0.4 源码 | GitHub google-deepmind/alphafold3 | ~5 MB | 数据管线 + C++ 扩展 |
| xfold 源码 | GitHub（`xfold-main.zip`） | ~2 MB | PyTorch 模型 |
| `components.cif.gz` | files.wwpdb.org/pub/pdb/data/monomers/ | 118 MB | CCD 字典 |
| 8 个 C++ 依赖 zip | 见下表 | ~7 MB | 编译 cpp 扩展 |
| fourier 参数 | alphafold3-decoded | ~5 KB | Fourier 嵌入 |

C++ 依赖包：

| 文件 | GitHub 仓库 |
|------|-------------|
| `abseil-cpp-d7aaad83b488fd62bd51c81ecf16cd938532cc0a.zip` | abseil/abseil-cpp |
| `pybind11-2e0815278cb899b20870a67ca8205996ef47e70f.zip` | pybind/pybind11 |
| `pybind11_abseil-bddf30141f9fec8e577f515313caec45f559d319.zip` | pybind/pybind11_abseil |
| `libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180.zip` | pdb-redo/libcifpp |
| `dssp-57560472b4260dc41f457706bc45fc6ef0bc0f10.zip` | PDB-REDO/dssp |
| `regex-boost-1.92.0.zip` | boostorg/regex |
| `eigen-3.4.0.zip` | libeigen/eigen |
| `libmcfp-2.0.4.zip` | mhekkel/libmcfp |

另需：`alphafold3-decoded`（用于导出 `fourier_weight.npy` / `fourier_bias.npy`），或直接把这两个 npy 放到 `/home`。

---

## 三、创建 Docker 容器

### 3.1 创建容器（使用空闲 NPU）

将 `davinci5` 换成实际空闲卡号：

```bash
# 用 docker create 避免触发联网拉取
docker create \
  --name af3-xfold \
  --device /dev/davinci5 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --ipc shareable \
  --network host \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home:/home/ \
  quay.io/ascend/vllm-ascend:v0.18.0rc1 \
  sleep 999999

docker start af3-xfold
docker ps --filter name=af3-xfold
docker exec af3-xfold npu-smi info
```

### 3.2 验证 Python + torch_npu

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import torch; print(\"torch:\", torch.__version__)"
python3 -c "import torch_npu; print(\"torch_npu:\", torch_npu.__version__)"
python3 -c "import torch; print(\"NPU available:\", torch.npu.is_available())"
python3 -c "import torch; print(\"NPU count:\", torch.npu.device_count())"
'
```

预期：

```
torch: 2.9.0+cpu
torch_npu: 2.9.0.post1
NPU available: True
NPU count: 1
```

---

## 四、安装 Python 依赖

### 4.1 配置 pip 镜像

```bash
docker exec af3-xfold bash -c '
mkdir -p /root/.pip /root/.config/pip
cat > /root/.pip/pip.conf << EOF
[global]
index-url = http://75.254.11.81:11569/pypi/simple
trusted-host = 75.254.11.81
timeout = 120
EOF
rm -f /root/.config/pip/pip.conf
'
```

> 现场若有其他 PyPI 镜像，替换 `index-url`。容器必须能访问 pip 源。

### 4.2 安装依赖包

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH

pip3 install absl-py
pip3 install scikit_build_core
pip3 install zstandard
pip3 install rdkit
pip3 install etils
pip3 install tqdm
pip3 install "tokamax==0.0.12"
pip3 install "dm-haiku==0.0.16"
pip3 install jax
pip3 install einops

python3 -c "import absl; print(\"absl OK\")"
python3 -c "import zstandard; print(\"zstandard OK\")"
python3 -c "import rdkit; print(\"rdkit OK\")"
python3 -c "import etils; print(\"etils OK\")"
python3 -c "import tokamax; print(\"tokamax OK\")"
python3 -c "import haiku as hk; print(\"haiku OK\")"
python3 -c "import jax; print(\"jax OK\")"
'
```

另需服务侧：`fastapi`、`uvicorn`、`pydantic`（若镜像未预装）。

---

## 五、部署代码

### 5.1 解压源码

```bash
# 宿主机操作（/home 已挂进容器）
cd /home
unzip -qo xfold-main.zip -d xfold
ls /home/xfold/xfold-main/xfold/alphafold3.py
ls /home/alphafold3-3.0.4/src/alphafold3/params.py
ls /home/af3.bin
```

### 5.2 修补 xfold 的 params.py

xfold 默认找 `af3.bin.zst`，现场用未压缩 `af3.bin`：

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
python3 << "PYEOF"
filepath = "/home/xfold/xfold-main/xfold/params.py"
with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "import zstandard",
    "try:\n    import zstandard\nexcept ImportError:\n    zstandard = None"
)

content = content.replace(
    'params = get_alphafold3_params(model_path / "af3.bin.zst")',
    'params = get_alphafold3_params(model_path / "af3.bin")'
)

content = content.replace(
    "if is_compressed:\n            yield zstandard.ZstdDecompressor().stream_reader(f)",
    "if is_compressed and zstandard is not None:\n            yield zstandard.ZstdDecompressor().stream_reader(f)\n        elif is_compressed:\n            raise ImportError(\"zstandard is required for compressed files\")"
)

with open(filepath, "w") as f:
    f.write(content)
print("params.py patched OK")
PYEOF
'
```

> 若上游缩进与上述 `replace` 对不齐，改为按符号手工改，不要死磕整段字符串。

### 5.3 转换 Fourier 参数

`import_jax_weights_()` 还要读 `/home/fourier_weight.npy` 和 `/home/fourier_bias.npy`。

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
python3 << "PYEOF"
import torch, numpy as np, os

src = "/home/alphafold3-decoded/alphafold3-decoded-main/data/params"
dst = "/home"

if os.path.exists(src):
    w = torch.load(os.path.join(src, "diff_fourier_weight.pt"), weights_only=False)
    b = torch.load(os.path.join(src, "diff_fourier_bias.pt"), weights_only=False)
    np.save(os.path.join(dst, "fourier_weight.npy"), w.numpy())
    np.save(os.path.join(dst, "fourier_bias.npy"), b.numpy())
    print(f"Fourier params saved: weight={w.shape}, bias={b.shape}")
else:
    print("WARNING: alphafold3-decoded not found, skip fourier params")
    print("Put fourier_weight.npy and fourier_bias.npy under /home manually")
PYEOF
'
```

---

## 六、编译 alphafold3.cpp C++ 扩展

这是最容易踩坑的步骤。

### 6.1 解压 C++ 依赖

```bash
docker exec af3-xfold bash -c '
mkdir -p /home/cpp_deps
cd /home/cpp_deps
for f in abseil-cpp-*.zip pybind11-*.zip pybind11_abseil-*.zip libcifpp-*.zip dssp-*.zip regex-boost-*.zip eigen-*.zip libmcfp-*.zip; do
  unzip -qo /home/$f 2>/dev/null
done
ls -d /home/cpp_deps/*/
'
```

解压后目录名必须能对上 CMake `SOURCE_DIR`（带 commit hash 的那种）。

### 6.2 修改 CMakeLists.txt（GitHub → 本地源码）

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
python3 << "PYEOF"
import os

path = "/home/alphafold3-3.0.4/pyproject.toml"
with open(path, "r") as f:
    content = f.read()
content = content.replace("requires-python = \">=3.12\"", "requires-python = \">=3.11\"")
with open(path, "w") as f:
    f.write(content)
print("pyproject.toml patched: Python >= 3.11")

path = "/home/alphafold3-3.0.4/CMakeLists.txt"
with open(path, "r") as f:
    content = f.read()

patches = [
    ("GIT_REPOSITORY https://github.com/abseil/abseil-cpp\n  GIT_TAG d7aaad83b488fd62bd51c81ecf16cd938532cc0a # 20240116.2",
     "SOURCE_DIR /home/cpp_deps/abseil-cpp-d7aaad83b488fd62bd51c81ecf16cd938532cc0a"),
    ("GIT_REPOSITORY https://github.com/pybind/pybind11\n  GIT_TAG 2e0815278cb899b20870a67ca8205996ef47e70f # v2.12.0",
     "SOURCE_DIR /home/cpp_deps/pybind11-2e0815278cb899b20870a67ca8205996ef47e70f"),
    ("GIT_REPOSITORY https://github.com/pybind/pybind11_abseil\n  GIT_TAG bddf30141f9fec8e577f515313caec45f559d319 # HEAD @ 2024-08-07",
     "SOURCE_DIR /home/cpp_deps/pybind11_abseil-bddf30141f9fec8e577f515313caec45f559d319"),
    ("GIT_REPOSITORY https://github.com/pdb-redo/libcifpp\n  GIT_TAG ac98531a2fc8daf21131faa0c3d73766efa46180 # v7.0.3",
     "SOURCE_DIR /home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180"),
    ("GIT_REPOSITORY https://github.com/PDB-REDO/dssp\n  GIT_TAG 57560472b4260dc41f457706bc45fc6ef0bc0f10 # v4.4.7",
     "SOURCE_DIR /home/cpp_deps/dssp-57560472b4260dc41f457706bc45fc6ef0bc0f10"),
]
for old, new in patches:
    if old in content:
        content = content.replace(old, new)
        print(f"Patched: {new[:80]}...")
    else:
        print(f"WARNING: not found: {old[:80]}...")

if "find_package(Boost 1.74 REQUIRED" not in content:
    content = content.replace(
        "target_link_libraries(\n  cpp\n  PRIVATE absl::check",
        "find_package(Boost 1.74 REQUIRED COMPONENTS regex)\n\ntarget_link_libraries(\n  cpp\n  PRIVATE absl::check"
    )
    content = content.replace(
        "cifpp::cifpp)",
        "cifpp::cifpp\n          Boost::regex)"
    )
    print("Patched: added Boost::regex link")

with open(path, "w") as f:
    f.write(content)

path = "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/CMakeLists.txt"
with open(path, "r") as f:
    content = f.read()
content = content.replace("find_package(Boost 1.80", "find_package(Boost 1.74")
content = content.replace(
    "GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git\n\t\tGIT_TAG 3.4.0",
    "SOURCE_DIR /home/cpp_deps/eigen-3.4.0"
)
with open(path, "w") as f:
    f.write(content)
print("libcifpp CMakeLists.txt patched")

path = "/home/cpp_deps/dssp-57560472b4260dc41f457706bc45fc6ef0bc0f10/CMakeLists.txt"
with open(path, "r") as f:
    content = f.read()
content = content.replace(
    "GIT_REPOSITORY https://github.com/mhekkel/libmcfp\n\t\tGIT_TAG v1.3.1",
    "SOURCE_DIR /home/cpp_deps/libmcfp-2.0.4"
)
content = content.replace(
    "GIT_REPOSITORY https://github.com/pdb-redo/libcifpp.git\n\t\tGIT_TAG v7.0.3",
    "SOURCE_DIR /home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180"
)
idx = content.find("add_executable(mkdssp")
if idx >= 0:
    content = content[:idx]
    content += "\n# mkdssp executable removed (only need dssp library)\n"
with open(path, "w") as f:
    f.write(content)
print("dssp CMakeLists.txt patched")

rsrc = "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc"
os.makedirs(rsrc, exist_ok=True)
with open(os.path.join(rsrc, "components.cif"), "w") as f:
    f.write("")
print("Empty components.cif created (will be replaced later)")
print("\n=== All CMakeLists.txt patches complete ===")
PYEOF
'
```

容器内需已安装 `libboost-regex-dev`（或等价包），否则 `find_package(Boost ... regex)` 会失败。

### 6.3 编译

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

BUILD_DIR=/home/alphafold3-3.0.4/build_cpp
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path(\"include\"))")
NUMPY_INCLUDE_DIR=$(python3 -c "import numpy; print(numpy.get_include())")

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DSKBUILD_PROJECT_NAME=alphafold3 \
  -DSKBUILD_PROJECT_VERSION=3.0.4 \
  -DPython3_EXECUTABLE=$(which python3) \
  -DPython3_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
  -DPython3_NumPy_INCLUDE_DIR=$NUMPY_INCLUDE_DIR \
  -DCIFPP_DOWNLOAD_CCD=OFF \
  -DBUILD_TESTING=OFF \
  -G Ninja \
  /home/alphafold3-3.0.4

cmake --build . --target cpp -j$(nproc)

find $BUILD_DIR -name "cpp*.so" -type f
cp $BUILD_DIR/cpp.cpython-311-aarch64-linux-gnu.so /home/alphafold3-3.0.4/src/alphafold3/cpp.so
echo "cpp.so copied!"

python3 -c "
import sys
sys.path.insert(0, \"/home/alphafold3-3.0.4/src\")
import alphafold3.cpp
print(\"alphafold3.cpp OK!\")
print(\"Attributes:\", [x for x in dir(alphafold3.cpp) if not x.startswith(\"_\")])
"
'
```

预期：`alphafold3.cpp OK!` 以及约 13 个子模块。编译大约 2–5 分钟。

---

## 七、生成 CCD 数据

### 7.1 解压 components.cif

```bash
cd /home
gunzip -kf components.cif.gz
ls -la components.cif   # 预期约 514 MB

cp components.cif /home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc/components.cif
```

### 7.2 生成 CCD pickle

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export LIBCIFPP_DATA_DIR=/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc

cd /home/alphafold3-3.0.4/src

python3 -c "
import sys
sys.path.insert(0, \".\")
from alphafold3.constants.converters.ccd_pickle_gen import main
main([\"ccd_pickle_gen\", \"/home/components.cif\", \"alphafold3/constants/converters/chemical_components.pickle\"])
print(\"CCD pickle generated\")
"

ln -sf chemical_components.pickle alphafold3/constants/converters/ccd.pickle

python3 -c "
import sys
sys.path.insert(0, \".\")
from alphafold3.constants.converters.chemical_component_sets_gen import main
main([\"chemical_component_sets_gen\", \"alphafold3/constants/converters/chemical_component_sets.pickle\"])
print(\"chemical_component_sets generated\")
"

ls -la alphafold3/constants/converters/*.pickle
'
```

CCD pickle 约 1–2 分钟，产物约 546 MB。

### 7.3 验证 pkl 加载

**不要在已 import torch_npu 的同一进程里测 pickle。** 下面这条是干净进程：

```bash
docker exec af3-xfold bash -c '
export PATH=/usr/local/python3.11.14/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export LIBCIFPP_DATA_DIR=/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc

python3 -c "
import sys, pickle
sys.path.insert(0, \"/home/alphafold3-3.0.4/src\")
with open(\"/home/alphafold3-3.0.4/src/alphafold3/test_data/featurised_example.pkl\", \"rb\") as f:
    batch = pickle.load(f)
print(f\"PKL loaded OK! type={type(batch)}, len={len(batch) if isinstance(batch, list) else \"N/A\"}\")
if isinstance(batch, list):
    b = batch[0]
    print(f\" keys ({len(b)}): {list(b.keys())[:10]}\")
"
'
```

预期：`PKL loaded OK! type=<class 'list'>, len=1`。

---

## 八、验证模型加载 + NPU 推理

### 8.1 创建启动脚本

```bash
cat > /home/af3_env.sh << 'EOF'
#!/bin/bash
export PATH=/usr/local/python3.11.14/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export LIBCIFPP_DATA_DIR=/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc
EOF
chmod +x /home/af3_env.sh
```

### 8.2 运行验证脚本

```bash
docker exec af3-xfold bash -c '
source /home/af3_env.sh

python3 << "PYEOF"
import sys, os, pickle, time, numpy as np

sys.path.insert(0, "/home/alphafold3-3.0.4/src")
print("1. Loading pkl...", flush=True)
with open("/home/alphafold3-3.0.4/src/alphafold3/test_data/featurised_example.pkl", "rb") as f:
    batch = pickle.load(f)
batch = batch[0]
arrays = {}
for k, v in batch.items():
    if isinstance(v, np.ndarray):
        arrays[k] = v
    elif isinstance(v, (int, float, bool)):
        arrays[k] = np.array(v)
np.savez("/tmp/test_batch.npz", **arrays)
print(f"   Saved {len(arrays)} arrays", flush=True)

sys.path.insert(0, "/home/xfold/xfold-main")
import torch, torch_npu
from xfold.fastnn import config as fastnn_config
fastnn_config.layer_norm_implementation = "torch"
fastnn_config.dot_product_attention_implementation = "torch"
fastnn_config.gated_linear_unit_implementation = "torch"
from xfold.alphafold3 import AlphaFold3
from xfold.params import import_jax_weights_
import pathlib

DEVICE = torch.device("npu:0")
print("2. Building model...", flush=True)
model = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=10)
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
print("3. Loading weights...", flush=True)
import_jax_weights_(model, pathlib.Path("/home"))
model.eval()
model = model.to(device=DEVICE)
print(f"   Model on {DEVICE}", flush=True)

data = np.load("/tmp/test_batch.npz", allow_pickle=True)
batch = {}
for k in data.files:
    arr = data[k]
    if arr.dtype == np.object_:
        continue
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.int64:
        arr = arr.astype(np.int32)
    t = torch.from_numpy(arr)
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    if t.dtype == torch.float64:
        t = t.to(torch.float32)
    batch[k] = t.to(device=DEVICE)

print("4. Running forward pass...", flush=True)
t0 = time.time()
with torch.inference_mode():
    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        result = model(batch)
torch.npu.synchronize()
print(f"   Done in {time.time()-t0:.2f}s", flush=True)
print(f"   Result keys: {list(result.keys())}", flush=True)
print("\n=== VERIFICATION PASSED ===", flush=True)
PYEOF
'
```

预期类似：

```
1. Loading pkl...
   Saved 65 arrays
2. Building model...
   Parameters: 368,384,666
3. Loading weights...
   Model on npu:0
4. Running forward pass...
   Done in 1.75s
   Result keys: ['diffusion_samples', 'distogram', 'predicted_lddt', ...]

=== VERIFICATION PASSED ===
```

注意：容器内只有一张挂载进去的卡时，对进程来说就是 `npu:0`，即使宿主机编号是 davinci5。

---

## 九、启动推理服务

### 9.1 部署服务脚本

将附录中的 `af3_service.py` 放到 `/home/af3_service.py`：

```bash
ls -la /home/af3_service.py
```

### 9.2 启动服务

```bash
docker exec af3-xfold bash -c '
source /home/af3_env.sh
pkill -f af3_service 2>/dev/null || true
sleep 1
nohup python3 /home/af3_service.py > /home/af3_service.log 2>&1 &
echo "PID: $!"

for i in $(seq 1 60); do
  sleep 5
  if curl -s http://localhost:9800/health 2>/dev/null | grep -q "ok"; then
    echo "Service ready after $((i*5))s"
    curl -s http://localhost:9800/health
    break
  fi
  echo "  waiting... (${i}x5s)"
done
'
```

### 9.3 验证服务

```bash
docker exec af3-xfold curl -s http://localhost:9800/health
docker exec af3-xfold curl -s http://localhost:9800/v1/models
docker exec af3-xfold curl -s -X POST http://localhost:9800/predict/test
```

### 9.4 外部访问

容器是 `--network host`，用服务器 IP：

```bash
curl http://<服务器IP>:9800/health

python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://<服务器IP>:9800/v1', api_key='none')
resp = client.chat.completions.create(
    model='alphafold3-3.0.4',
    messages=[{'role': 'user', 'content': '/home/alphafold3-3.0.4/src/alphafold3/test_data/featurised_example.pkl'}]
)
print(resp.choices[0].message.content)
"
```

当前服务只吃 **预特征化 pkl 路径**，不是原始氨基酸字符串。完整 MSA 管线需另接官方 `run_alphafold.py --run_inference=false`。

---

## 故障排查

### 容器创建失败 `No such device`

```bash
ls /dev/davinci*
ls /dev/davinci_manager
```

### torch_npu 导入失败 `libascend_hal.so not found`

```bash
docker exec af3-xfold ls /usr/local/Ascend/driver/lib64/
```

缺失则按第三节重新挂驱动建容器。

### pip install 失败 `Could not resolve host`

```bash
docker exec af3-xfold ping -c1 75.254.11.81
docker exec af3-xfold cat /root/.pip/pip.conf
docker exec af3-xfold cat /etc/resolv.conf
```

### C++ 编译失败仍出现 `GIT_REPOSITORY`

```bash
grep GIT_REPOSITORY /home/alphafold3-3.0.4/CMakeLists.txt
# 预期：无输出
ls /home/cpp_deps/*/
```

### C++ 产物 import 报 `undefined symbol: boost::re_detail`

```bash
grep "Boost::regex" /home/alphafold3-3.0.4/CMakeLists.txt
```

确认系统已装 Boost regex 开发包后重编。

### pkl 加载 segfault

根因：同一进程里 `torch_npu` + `alphafold3.cpp` + pickle。  
处理：先转 npz 再 import torch；用户 pkl 走子进程。检查 `preload_test_data()` 是否在 import torch 之前。

### NPU 报 `DT_DOUBLE not supported`

所有 float64 输入转 float32，检查 `npz_to_torch_batch()`。

### 端口占用

```bash
docker exec af3-xfold bash -c 'pkill -f af3_service; sleep 2'
```

### 看日志

```bash
docker exec af3-xfold tail -50 /home/af3_service.log
```

### xfold `run_alphafold.py` 不要当 NPU 入口

xfold 22bdeed 的 `run_alphafold.py` 会 import 3.0.4 已删除的 `base_model` / `Diffuser`。现场推理走 `af3_service.py` + `xfold.alphafold3`。官方 JAX `run_alphafold.py` 只适合 CPU 数据管线或 GPU 对拍。

---

## 附录：af3_service.py

将以下内容保存为 `/home/af3_service.py`。

```python
#!/usr/bin/env python3
"""AlphaFold3 NPU Inference Service with OpenAI-compatible API."""
import sys, os, json, pickle, time, pathlib, subprocess
import numpy as np

# Phase 1: Load test pkl to npz BEFORE importing torch_npu
sys.path.insert(0, "/home/alphafold3-3.0.4/src")
os.environ.setdefault(
    "LIBCIFPP_DATA_DIR",
    "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc",
)

TEST_PKL = "/home/alphafold3-3.0.4/src/alphafold3/test_data/featurised_example.pkl"
TEST_NPZ = "/tmp/test_batch.npz"


def preload_test_data():
    print("Pre-loading test pkl to npz...", flush=True)
    with open(TEST_PKL, "rb") as f:
        batch = pickle.load(f)
    if isinstance(batch, list):
        batch = batch[0]
    arrays = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (int, float, bool)):
            arrays[k] = np.array(v)
    np.savez(TEST_NPZ, **arrays)
    print(f"Pre-loaded {len(arrays)} arrays", flush=True)


preload_test_data()

# Phase 2: Import torch_npu
import torch
import torch_npu

sys.path.insert(0, "/home/xfold/xfold-main")
from xfold.fastnn import config as fastnn_config

fastnn_config.layer_norm_implementation = "torch"
fastnn_config.dot_product_attention_implementation = "torch"
fastnn_config.gated_linear_unit_implementation = "torch"

from xfold.alphafold3 import AlphaFold3
from xfold.params import import_jax_weights_

DEVICE = torch.device("npu:0")
MODEL = None


def load_model():
    global MODEL
    print("Loading AlphaFold3 model...", flush=True)
    model = AlphaFold3(num_recycles=1, num_samples=1, diffusion_steps=10)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    import_jax_weights_(model, pathlib.Path("/home"))
    model.eval()
    model = model.to(device=DEVICE)
    MODEL = model
    print(f"Model on {DEVICE}. Ready.", flush=True)


def npz_to_torch_batch(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    batch = {}
    for k in data.files:
        arr = data[k]
        if arr.dtype == np.object_:
            continue
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        t = torch.from_numpy(arr)
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        if t.dtype == torch.float64:
            t = t.to(torch.float32)
        batch[k] = t.to(device=DEVICE)
    return batch


def load_pkl_via_subprocess(pkl_path, npz_path="/tmp/custom_batch.npz"):
    env = dict(os.environ)
    env["LIBCIFPP_DATA_DIR"] = (
        "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc"
    )
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:"
        + env.get("LD_LIBRARY_PATH", "")
    )
    script = f"""
import sys, pickle, numpy as np
sys.path.insert(0, "/home/alphafold3-3.0.4/src")
with open({pkl_path!r}, "rb") as f:
    batch = pickle.load(f)
if isinstance(batch, list):
    batch = batch[0]
arrays = {{}}
for k, v in batch.items():
    if isinstance(v, np.ndarray):
        arrays[k] = v
    elif isinstance(v, (int, float, bool)):
        arrays[k] = np.array(v)
np.savez({npz_path!r}, **arrays)
"""
    result = subprocess.run(
        ["/usr/local/python3.11.14/bin/python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pkl load failed: {result.stderr[-500:]}")
    return npz_path


def run_inference(batch):
    with torch.inference_mode():
        with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
            result = MODEL(batch)
    torch.npu.synchronize()
    return result


def result_to_serializable(result):
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "norm": float(obj.float().norm().item()) if obj.numel() > 0 else 0.0,
            }
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, (int, float, str, bool)):
            return obj
        return str(obj)

    return convert(result)


def start_server():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import Optional
    import uvicorn

    app = FastAPI(title="AlphaFold3 NPU Service", version="1.0.0")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": "alphafold3-3.0.4",
            "device": str(DEVICE),
            "npu_available": torch.npu.is_available(),
            "parameters": sum(p.numel() for p in MODEL.parameters()),
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "alphafold3-3.0.4",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "xfold-npu",
                }
            ],
        }

    class ChatRequest(BaseModel):
        model: str = "alphafold3-3.0.4"
        messages: list
        temperature: Optional[float] = 1.0
        max_tokens: Optional[int] = 4096
        stream: Optional[bool] = False

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        user_msg = req.messages[-1]["content"] if req.messages else ""
        pkl_path = user_msg.strip()
        if not os.path.exists(pkl_path):
            return JSONResponse(
                {
                    "id": f"af3-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "alphafold3-3.0.4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"error": f"File not found: {pkl_path}"}
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        try:
            npz_path = load_pkl_via_subprocess(pkl_path)
            batch = npz_to_torch_batch(npz_path)
            t0 = time.time()
            result = run_inference(batch)
            elapsed = time.time() - t0
            return JSONResponse(
                {
                    "id": f"af3-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "alphafold3-3.0.4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "status": "success",
                                        "inference_time_sec": elapsed,
                                        "result_summary": result_to_serializable(result),
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        except Exception as e:
            return JSONResponse(
                {
                    "id": f"af3-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "alphafold3-3.0.4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps({"error": str(e)}),
                            },
                            "finish_reason": "error",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )

    class PredictRequest(BaseModel):
        pkl_path: str

    @app.post("/predict")
    async def predict(req: PredictRequest):
        if not os.path.exists(req.pkl_path):
            raise HTTPException(status_code=404, detail=f"File not found: {req.pkl_path}")
        try:
            npz_path = load_pkl_via_subprocess(req.pkl_path)
            batch = npz_to_torch_batch(npz_path)
            t0 = time.time()
            result = run_inference(batch)
            elapsed = time.time() - t0
            return {
                "status": "ok",
                "inference_time_sec": elapsed,
                "result_summary": result_to_serializable(result),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/test")
    async def predict_test():
        try:
            batch = npz_to_torch_batch(TEST_NPZ)
            t0 = time.time()
            result = run_inference(batch)
            elapsed = time.time() - t0
            return {
                "status": "ok",
                "inference_time_sec": elapsed,
                "result_summary": result_to_serializable(result),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    print("Starting FastAPI server on 0.0.0.0:9800...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=9800, log_level="info")


if __name__ == "__main__":
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/Ascend/driver/lib64:/usr/lib/aarch64-linux-gnu:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["LIBCIFPP_DATA_DIR"] = (
        "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc"
    )
    load_model()
    start_server()
```

---

## 部署检查清单

- [ ] 容器 `af3-xfold` 正常运行
- [ ] 容器内 `npu-smi info` 能看到目标 NPU
- [ ] `torch_npu` 导入成功，`torch.npu.is_available() == True`
- [ ] Python 依赖齐全（absl、zstandard、rdkit、etils、tokamax、haiku、jax、einops、fastapi、uvicorn）
- [ ] xfold `params.py` 已改（`af3.bin` + zstandard 可选）
- [ ] `/home/fourier_weight.npy` 与 `/home/fourier_bias.npy` 存在
- [ ] `alphafold3/cpp.so` 存在且 `import alphafold3.cpp` 成功
- [ ] `chemical_components.pickle`（约 546MB）与 `chemical_component_sets.pickle` 已生成
- [ ] 干净进程能加载 test pkl（约 65 keys）
- [ ] 权重加载成功（约 3.68 亿参数）
- [ ] 模型在 `npu:0` 上 forward 成功（快速约 1.75s / 完整约 80s）
- [ ] FastAPI `:9800`，`/health`、`/v1/models`、`/predict/test` 正常
