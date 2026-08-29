# AlphaFold3 昇腾 NPU 迁移 — 第二轮修复需求（63.41 现场实测）

> 对象：外网 PC / https://github.com/Stonesan233/alphafold3_ascend  
> 基线：xfold `22bdeed` / alphafold3 `c0f97ed`  
> 现场：Atlas 800I A2（A3-syn-31，NPU 4），容器 `vllm-ascend:v0.23.0-a3`，  
> Python 3.12.13，torch 2.10.0 + torch_npu 2.10.0.post4，g++ 11.4.0  
>
> 部署已跑通：模型加载、NPU fp32 推理、OpenAI 兼容服务（冒烟 1.06s，  
> `predicted_lddt_mean` 83.6，参数量 368,384,666）。  
> 下列 **7 项为现场手工修复**，需落进仓库。不要再让客户现场 sed。

---

## 修复清单

| # | 组件 | 问题 | 优先级 |
|---|------|------|--------|
| 1 | `deps/eigen` | 打包缺 `Core` / `Sparse` 模块，编译必失败 | P0 |
| 2 | `deps/libmcfp` | 2.0.4 需 C++23，现场 g++ 11 编不过 | P0 |
| 3 | `deps/libcifpp` | eigen include 少一层，`#include "Core"` 找不到 | P0 |
| 4 | `alphafold3/pyproject.toml` | `jax[cuda12]` 在 NPU 环境拖约 3GB 无用 CUDA 包 | P1 |
| 5 | `service/af3_service.py` | Pydantic 类写在函数内，FastAPI 启动 `NameError` | P0 |
| 6 | `alphafold3/CMakeLists.txt` | libcifpp 的 catch2 测试断网拉 GitHub | P1 |
| 7 | 文档 | `components.cif` 预放置与编译参数没写清 | P1 |

评审结论：7 项根因与修法都成立，可按本文落地。额外注意见文末「不要漏的点」。

---

## 修复 1：`deps/eigen` 补齐 Core / Sparse（P0）

**现场报错**

```text
/deps/eigen/Eigen/Eigenvalues:11:10: fatal error: Core: No such file or directory
11 | #include "Core"
```

**根因**

`deps/eigen/` 不完整。Eigen 3.4.0 的 `Eigen/` 需要：

1. 模块入口文件：`Core`、`Sparse`、`Cholesky`、`Eigenvalues`、`Geometry`、`LU`、`QR`、`SVD` 等  
2. `src/` 实现：`src/Core/`（约 76 个文件）、`src/Sparse/`、`src/Cholesky/` 等  

现场缺 `Eigen/Core`、`Eigen/Sparse` 以及整个 `Eigen/src/Core/`。  
高度怀疑打包脚本 / `.gitignore` 把名为 `Core`、`Sparse` 的文件或目录当构建产物滤掉了。

**修法**

用官方 `eigen-3.4.0.zip` 整包覆盖 `deps/eigen/`，或至少补：

```text
deps/eigen/Eigen/Core
deps/eigen/Eigen/Sparse
deps/eigen/Eigen/src/Core/     # 含 util/Macros.h 等
```

同时检查仓库 `.gitignore`、打包脚本，去掉对 `Core`/`Sparse` 的误过滤。

**验收**

```bash
test -f deps/eigen/Eigen/Core
test -d deps/eigen/Eigen/src/Core
ls deps/eigen/Eigen/src/Core/util/Macros.h
```

---

## 修复 2：`deps/libmcfp` 换成 v1.3.4（P0）

**现场报错**

```text
error: 'unreachable' is not a member of 'std'   # C++23
error: invalid type for parameter 3 of 'constexpr' function
```

**根因**

deps 里是 libmcfp **v2.0.4**（C++23：`std::unreachable`、constexpr string）。  
现场 g++ **11.4**，最高 C++20。  
dssp v4.4.7 上游钦定的是 libmcfp **v1.3.1**（1.3.x 为 C++17/20）。

**修法**

用 **v1.3.4**（1.3.x 最新 bugfix，与 1.3.1 API 兼容，现场已编过）整目录替换 `deps/libmcfp/`：

- https://github.com/mhekkel/libmcfp/archive/refs/tags/v1.3.4.tar.gz  
- 更新 README / `AF3_Ascend_deps_packaging_requirements.md` 里仍写着的 v2.0.4  

**已验证**：v1.3.4 + g++ 11.4 + `-std=gnu++20` 通过。

---

## 修复 3：libcifpp 的 eigen include 加一层（P0）

**现场报错**（补齐 Eigen 文件后仍报）

```text
-I/deps/eigen -I/deps/libcifpp/include ...
/deps/eigen/Eigen/Eigenvalues:11: fatal error: Core: No such file or directory
```

**根因**

`deps/libcifpp/CMakeLists.txt` 约 259 行：

```cmake
set(EIGEN_INCLUDE_DIR ${my-eigen3_SOURCE_DIR})
```

只加了 eigen 根目录。`Eigen/Eigenvalues` 里是 `#include "Core"`，按 `Eigen/` 目录解析，还需要 `${my-eigen3_SOURCE_DIR}/Eigen`。

**修法（现场已通）**

```cmake
# 原
set(EIGEN_INCLUDE_DIR ${my-eigen3_SOURCE_DIR})

# 改（CMake 列表，分号分隔）
set(EIGEN_INCLUDE_DIR "${my-eigen3_SOURCE_DIR};${my-eigen3_SOURCE_DIR}/Eigen")
```

同步改 `patches/libcifpp-boost-eigen.patch`，与 deps 内已改文件一致。

---

## 修复 4：去掉强制 `jax[cuda12]`（P1）

**现场现象**

每次 `pip install -e .` 拉约 3GB NVIDIA 包（cublas / cudnn / cusparse / nccl 等），NPU 容器无用。三次编译约 10GB 流量 + 20 分钟。

**根因**

官方 `pyproject.toml`：

```toml
"jax[cuda12]==0.10.2; sys_platform != 'darwin'",
```

非 Darwin 无条件拉 CUDA extra。

**修法（现场已通）**

NPU/CPU 只需能 `import jax` / `jax.numpy`：

```toml
dependencies = [
  # ...
  "jax==0.10.2; sys_platform != 'darwin'",
  "jax-mps==0.10.9; sys_platform == 'darwin' and platform_machine == 'arm64'",
  # ...
]

[project.optional-dependencies]
cuda = ["jax[cuda12]==0.10.2; sys_platform != 'darwin'"]
```

默认安装不再拉 nvidia 包；真有 GPU 时：`pip install -e ".[cuda]"`。

---

## 修复 5：`af3_service.py` Pydantic 类移到模块顶层（P0）

**现场报错**

```text
File "service/af3_service.py", line 434, in main
    app = build_app()
NameError: name 'ChatRequest' is not defined
```

**根因**

`ChatRequest` / `PredictRequest` 写在 `build_app()` **内部**，文件有 `from __future__ import annotations`。  
FastAPI 用 `inspect.signature(..., eval_str=True)` 在 **模块全局** 求值注解字符串，找不到函数局部类。

**修法（现场已通）**

1. 两个 Pydantic 类移到模块顶层（`def build_app():` 之前）  
2. `from pydantic import BaseModel` 放到文件顶部 import  
3. 字段默认值里的 `MODEL_ID` 必须在顶层已定义，或写成字面量 `"alphafold3-3.0.4"`

**验收**

```bash
python3 -m py_compile service/af3_service.py
python3 service/af3_service.py    # 能进 uvicorn（模型加载约 30s）
curl -s http://localhost:9800/health
```

---

## 修复 6：关掉 libcifpp catch2 / CCD 在线拉取（P1）

**现场报错**

```text
Performing download step (git clone) for 'catch2-populate'
fatal: unable to access 'https://github.com/catchorg/Catch2.git/'
CMake Error at deps/libcifpp/test/CMakeLists.txt:11 (FetchContent_MakeAvailable)
```

**根因**

`libcifpp/test/CMakeLists.txt` 无条件 FetchContent Catch2。仅 `-DBUILD_TESTING=OFF` 时，部分版本仍 include 该文件。  
dssp 里的 `set(CIFPP_DOWNLOAD_CCD OFF)` 在 **dssp 子作用域**，管不到顶层第一次 `MakeAvailable(cifpp)`。

**方案 A（推荐，写进顶层 CMake）**

`alphafold3/CMakeLists.txt` 在 `FetchContent_MakeAvailable(...)` **之前**：

```cmake
set(CIFPP_DOWNLOAD_CCD OFF CACHE BOOL
    "Do not download components.cif from wwpdb during configure" FORCE)
set(CIFPP_INSTALL_UPDATE_SCRIPT OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(DSSP_BUILD_MKDSSP OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(pybind11 abseil-cpp pybind11_abseil cifpp dssp)
```

`FORCE` 防止被子项目 `set(... ON CACHE)` 盖掉。  
若 scikit-build 另起 cache，README 仍保留：

```bash
pip install -e . --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"
```

**方案 B（deps 瘦身，可与 A 同时做）**

打包时删掉 `deps/libcifpp/test/`、`deps/libcifpp/examples/`、`deps/dssp/test/`。

---

## 修复 7：README 补全（P1）

### 7.1 `components.cif` 必须预放置

514MB `components.cif` 是 **数据包**（与 `af3.bin`、遗传库同级），不进 git、不进 `deps.tar.gz`。

编译前：

```bash
cp /path/to/data-package/components.cif deps/libcifpp/rsrc/components.cif
ls -lh deps/libcifpp/rsrc/components.cif   # 必须约 514MB，禁止 0 字节
```

libcifpp：文件存在且 **size ≠ 0** 才跳过 wwpdb。空占位会被删掉再下（现场踩过）。

顶层已 `CIFPP_DOWNLOAD_CCD=OFF` 时，编译可以没有这份文件；**生成 CCD pickle / 跑数据管线** 仍然需要真文件，并设置：

```bash
export LIBCIFPP_DATA_DIR=/path/to/deps/libcifpp/rsrc
```

### 7.2 完整编译命令

```bash
export ALPHAFOLD3_DEPS_DIR=/path/to/deps   # 不要写成 ALPHAFAFOLD3_DEPS_DIR
cp /path/to/data-package/components.cif "$ALPHAFOLD3_DEPS_DIR/libcifpp/rsrc/components.cif"

pip install -e alphafold3/ --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"
```

修复 6 方案 A 落地后，`--config-settings` 可省略，但建议 README 两行都写。

### 7.3 环境变量防呆

现场曾把 `ALPHAFOLD3_DEPS_DIR` 敲成 `ALPHAFAFOLD3_DEPS_DIR`（多一个 F），走了在线 FetchContent。  
README 在变量旁加提示；CMake 在 `DEPS_DIR` 为空时 `message(WARNING ...)`。

### 7.4 其它现场依赖（写进 README）

```bash
pip install fastapi uvicorn zstandard rdkit etils absl-py tqdm
pip install tokamax==0.0.12
# tokamax>=0.1.0 去掉了 DotProductAttentionImplementation，不要升级
```

---

## 现场已验证流程（回归用）

```bash
# 0. 容器 vllm-ascend:v0.23.0-a3，绑 NPU，挂 /mnt/weight/alphafold3
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev

pip install fastapi uvicorn zstandard rdkit etils absl-py tqdm
pip install tokamax==0.0.12

export ALPHAFOLD3_DEPS_DIR=$PWD/deps
cp /mnt/weight/alphafold3/components.cif deps/libcifpp/rsrc/

cd alphafold3
pip install -e . --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"

python3 -c "import alphafold3.cpp; print('OK')"

python3 -c "
from alphafold3.constants.converters.ccd_pickle_gen import main
main(['g', '/mnt/weight/alphafold3/components.cif',
      'src/alphafold3/constants/converters/chemical_components.pickle'])
"
cd src && ln -sf chemical_components.pickle \
  alphafold3/constants/converters/ccd.pickle
python3 -c "
from alphafold3.constants.converters.chemical_component_sets_gen import main
main(['g', 'alphafold3/constants/converters/chemical_component_sets.pickle'])
"

cd ../service
export AF3_SRC=../alphafold3/src
export XFOLD_SRC=../xfold
export AF3_MODEL_DIR=/mnt/weight/alphafold3
export LIBCIFPP_DATA_DIR=../deps/libcifpp/rsrc
export AF3_NUM_RECYCLES=1 AF3_NUM_SAMPLES=1 AF3_DIFFUSION_STEPS=10
nohup python3 af3_service.py > /tmp/af3.log 2>&1 &
sleep 45
curl -s http://localhost:9800/health
curl -s -X POST http://localhost:9800/predict/test
```

---

## 回归基线（A3-syn-31 实测）

| 指标 | 值 |
|------|-----|
| `cpp.so` import | OK，约 13 个子模块，无 boost 缺符号 |
| 冒烟（recycles=1, samples=1, steps=10） | 约 1.06s |
| `predicted_lddt_mean` | 约 83.60 |
| 参数量 | 368,384,666 |
| 设备 / 精度 | `npu:0`，fp32 |
| 权重 | `af3.bin.zst`（自动检测） |

---

## 现场环境

| 组件 | 版本 |
|------|------|
| 硬件 | Atlas 800I A2（A3-syn-31），NPU 4 |
| 容器 | `quay.io/ascend/vllm-ascend:v0.23.0-a3` |
| Python | **3.12.13**（与内网 189 的 3.11 不同，`cpp.so` 按 3.12 ABI 编） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| g++ | 11.4.0（C++20 上限 → libmcfp 必须 1.3.x） |

---

## 不要漏的点（评审补丁）

1. **不要把 514MB `components.cif` 推进 GitHub。** 只写拷贝步骤。  
2. **Eigen 不完整和 include 少一层是两个独立 P0**，只补文件或只改 CMake 都会再炸。  
3. **`CIFPP_DOWNLOAD_CCD` 与 `BUILD_TESTING` 都要在顶层 FORCE**，不要只改 dssp。  
4. **Python 3.11 / 3.12 的 `cpp.so` 不能混用。** README 写清按容器 Python 现场编。  
5. **`tokamax==0.0.12` 钉死。**  
6. 旧文档里的 libmcfp **v2.0.4 全部改成 v1.3.4**。  
7. 验收至少覆盖：断网 `pip install -e alphafold3/`、`import alphafold3.cpp`、`py_compile af3_service.py`、`/health` + `/predict/test`。

---

## 给外网的最短任务

1. 重打 `deps/eigen`（完整 3.4.0），查清 `Core`/`Sparse` 被滤掉的原因。  
2. `deps/libmcfp` → v1.3.4。  
3. libcifpp：`EIGEN_INCLUDE_DIR` 含根目录 + `Eigen/`。  
4. 顶层 CMake：`CIFPP_DOWNLOAD_CCD` / `BUILD_TESTING` / `DSSP_BUILD_MKDSSP` 全部 OFF FORCE。  
5. `jax[cuda12]` 改 optional extra。  
6. Pydantic 模型移出 `build_app()`。  
7. README：cif 预放置、完整 pip 参数、变量名、tokamax 钉版本。
