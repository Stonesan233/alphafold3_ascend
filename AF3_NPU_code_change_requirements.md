# AlphaFold3 昇腾 NPU 迁移 — 代码修改需求文档（修订版）

> 本文档描述需要在**外网 PC**上完成的 xfold / alphafold3 代码修改。
> 修改完成后打包为 zip（建议同时提供基于锁定 commit 的 patch），交付内网部署。
>
> **修订说明**：相对初稿，本版根据代码评审补齐了：基线 commit 锁定、按符号定位、`fourier_*.npy`、triton 才是主雷、NPU 默认 float32、CMake 依赖目录命名、dssp 补丁交付物、外网/内网验证分层、服务层范围。

---

## 〇、怎么用这份文档

1. **先锁版本，再改代码。** 不要对最新 `main` 盲改。
2. **以符号/函数名为准，行号仅供参考。** 上游会变。
3. **外网只做门禁验证**（安装、import、权重加载、CPU/GPU 冒烟、离线编译）。
4. **NPU 行为以内网回归为准**（float64、NPU 设备、精度与耗时）。
5. 改完必须能回答：相对锁定 commit，每个文件改了什么、为什么改。

---

## 一、项目背景与当前状态

### 1.1 已完成的工作（内网已验证）

- xfold 在昇腾 NPU 上跑通 AF3 推理（torch 2.9 + torch_npu 2.9.0.post1）
- `af3.bin`（Haiku 二进制）权重转换已验证正确
- 完整推理：bfloat16 约 97.4s / float32 约 79.8s  
  （150 tokens，5 samples × 200 diffusion steps）
- FastAPI OpenAI 兼容推理服务已运行（服务代码是否入库见 2.3 范围）
- 全量数据库（约 627GB）已就位

内网关键结论（修改时必须保持）：

- 权重转换正确：`contact_probs` 与 JAX 参考输出均值差约 **0.0011**
- NPU 上必须走 **fastnn 的 torch 实现**（triton 不可用）
- NPU 不支持 float64，输入中的 float64 必须转 float32
- 昇腾上 **float32 比 bfloat16 更快且置信度更好**，生产默认应用 float32

### 1.2 当前代码的问题

内网部署时对 xfold / alphafold3 做了若干临时 patch（服务器上 sed/python 改文件），未进版本库，不可维护。需要在外网 PC 上将这些修改正式化。

### 1.3 代码仓库与基线版本（必须先锁定）

| 仓库 | 上游 | 用途 |
|------|------|------|
| xfold | https://github.com/Shenggan/xfold | PyTorch 版 AF3 模型 + 权重导入 + 推理入口 |
| alphafold3 | https://github.com/google-deepmind/alphafold3 | **只用数据管线 + C++ 扩展**，不用 JAX 模型 |

**基线 commit（外网开工前由内网填写，禁止空着 clone latest）：**

```text
xfold      : <填写内网正在运行的 git SHA 或 tag>
alphafold3 : <填写内网正在运行的 git SHA 或 tag，例如 v3.0.4>
```

操作顺序：

```bash
git clone https://github.com/Shenggan/xfold.git
cd xfold && git checkout <锁定 SHA>

git clone https://github.com/google-deepmind/alphafold3.git
cd alphafold3 && git checkout <锁定 SHA>
```

若内网暂时给不出 SHA：至少用内网 `git log -1` / 源码包头注释对齐后再改。

---

## 二、范围

### 2.1 必须改（P0）

- xfold：权重双格式、设备抽象、triton 保护、float64→float32、NPU 默认 float32
- alphafold3：Python 3.11、离线 FetchContent、Boost::regex

### 2.2 应该改（P1）

- alphafold3 依赖 dssp：可关闭 mkdssp 构建
- xfold：推理精度做成 CLI（`--precision fp32|bf16`），NPU 默认 fp32，CUDA 可保持 bf16

### 2.3 默认不在本次范围（除非内网明确要求一并交付）

- FastAPI / OpenAI 兼容服务（`af3_service.py` 等）
- MSA/模板数据库下载脚本
- NPU Flash Attention / `npu_fusion_attention` 性能优化
- 与 JAX 扩散 RNG 对齐（跨框架坐标 RMSD 问题，已定性为非 NPU 缺陷）

若交付口径包含「可对外提供推理 API」，服务代码需另开一节入库，不与本次模型仓改造混为一谈。

---

## 三、xfold 修改（优先级高）

定位方式：在仓库内搜索下列符号，不要依赖行号。

### 修改 1：params.py 支持压缩/非压缩双格式权重

**文件**：`xfold/params.py`  
**搜索**：`import zstandard`、`def open_for_reading`、`def import_jax_weights_`、`af3.bin.zst`

**现状**：

- `get_alphafold3_params(checkpoint_path)` 已能按后缀判断 `.zst` / 非压缩
- `select_model_files()` 也能识别两种后缀
- **真正的硬编码在 `import_jax_weights_()`**：固定 `model_path / "af3.bin.zst"`
- `import zstandard` 在模块顶层，未装 zstandard 时整个模块可能无法 import

**目标**：自动检测权重文件；zstandard 仅在读 `.zst` 时必需。

```python
# import 部分
try:
    import zstandard
except ImportError:
    zstandard = None


def open_for_reading(model_files, is_compressed: bool):
    with contextlib.closing(_MultiFileIO(model_files)) as f:
        if is_compressed:
            if zstandard is None:
                raise ImportError(
                    "zstandard is required to load .zst weights. "
                    "Install it with: pip install zstandard"
                )
            yield zstandard.ZstdDecompressor().stream_reader(f)
        else:
            yield f


def _resolve_af3_weight_file(model_path: pathlib.Path) -> pathlib.Path:
    zst_path = model_path / "af3.bin.zst"
    bin_path = model_path / "af3.bin"
    if zst_path.exists():
        return zst_path
    if bin_path.exists():
        return bin_path
    raise FileNotFoundError(
        f"No model weights found in {model_path} "
        f"(looked for af3.bin.zst and af3.bin)"
    )


def import_jax_weights_(model, model_path: pathlib.Path):
    params = get_alphafold3_params(_resolve_af3_weight_file(model_path))
    # 其余 mapping / assign 逻辑保持不变
    ...
```

**不要漏掉同函数里已有的辅助文件**（上游就有，改路径时一并保留）：

```text
model_path / "fourier_weight.npy"
model_path / "fourier_bias.npy"
```

验证权重目录最少应包含：

```text
af3.bin  或  af3.bin.zst
fourier_weight.npy
fourier_bias.npy
```

两个 npy 缺失时的报错应保持清晰，不要只测 bin 文件。

### 修改 2：消除 CUDA 硬编码，支持 NPU / CUDA / CPU

**文件**：`run_alphafold.py`  
**搜索**：`torch.device('cuda')`、`device_type="cuda"`、`torch.cuda.synchronize`

**目标**：自动选设备，并抽象 synchronize。

```python
def get_device() -> torch.device:
    """Auto-detect best available device: NPU > CUDA > CPU."""
    try:
        import torch_npu  # noqa: F401
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
```

替换点：

| 原代码 | 改为 |
|--------|------|
| `device = torch.device('cuda')` | `device = get_device()` |
| `torch.amp.autocast(device_type="cuda", ...)` | 见修改 4（不要只换 device_type 仍写死 bf16） |
| `torch.cuda.synchronize()`（两处） | `device_sync(model_runner._device)` |

允许后续用环境变量覆盖卡号（可选，P1）：

```text
AF3_DEVICE=npu:5   # 内网曾用 NPU 5
```

未实现覆盖时，文档/README 注明默认 `npu:0`。

### 修改 3：fastnn — triton 保护是主改，config 自动检测是辅改

**文件**：

- `xfold/fastnn/config.py`
- `xfold/fastnn/attention.py`
- `xfold/fastnn/layer_norm.py`
- `xfold/fastnn/gated_linear_unit.py`
- 以及其他顶层 `import triton` 的 fastnn 文件

**搜索**：`import triton`、`import triton.language`、`layer_norm_implementation`

**现状（重要）**：

- 上游 `config.py` **默认已经是 `"torch"`**，不是默认 triton
- 真正会在无 CUDA 机器上直接失败的是各文件顶部 **无条件 `import triton`**

**P0（必须）**：所有 `import triton` 改为可选，并在走 triton kernel 的入口检查。

```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False
```

调用 triton kernel 前：

```python
if implementation == "triton" and not HAS_TRITON:
    implementation = "torch"
```

或直接 fallback 到现有 torch 实现函数。

**P1（建议）**：`config.py` 增加自动检测，无 CUDA 时保持 torch：

```python
def _detect_default_implementation() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "triton"
    except ImportError:
        pass
    return "torch"


layer_norm_implementation = _detect_default_implementation()
dot_product_attention_implementation = _detect_default_implementation()
gated_linear_unit_implementation = _detect_default_implementation()
```

NPU 环境没有 CUDA，检测结果必须是 `"torch"`。不要在 NPU 上尝试 triton。

### 修改 4：NPU 数据类型 + 默认精度

**文件**：`run_alphafold.py` → `ModelRunner.run_inference`  
**搜索**：`run_inference`、`featurised_example`、`autocast`、`deletion_mean`

#### 4.1 float64 → float32（P0）

昇腾 `aclnnMatmul` 不支持 `DT_DOUBLE`。在张量 `.to(device)` 之后增加：

```python
featurised_example = pytree.tree_map_only(
    torch.Tensor,
    lambda x: x.to(dtype=torch.float32) if x.dtype == torch.float64 else x,
    featurised_example,
)
```

确认文件中已使用 `torch.utils._pytree`（或项目里现有的 pytree 别名），不要引入未使用的新依赖。

#### 4.2 不要只把 CUDA autocast 换成 NPU bfloat16（P0）

错误改法（禁止作为 NPU 生产默认）：

```python
# BAD on NPU
with torch.amp.autocast(device_type=self._device.type, dtype=torch.bfloat16):
    ...
```

正确策略：

| 设备 | 默认精度 | autocast |
|------|----------|----------|
| NPU | **float32** | 不启用 bf16 autocast |
| CUDA | 可保持上游 bf16 autocast | `device_type="cuda"` |
| CPU | float32 | 不启用 |

建议实现（P1 做成 flag，P0 至少 NPU 走 fp32）：

```python
def _use_bf16_autocast(device: torch.device, precision: str) -> bool:
    if precision == "fp32":
        return False
    if precision == "bf16":
        return device.type in ("cuda", "npu")
    # auto
    return device.type == "cuda"


# run_inference 内
if _use_bf16_autocast(self._device, self._precision):
    with torch.amp.autocast(device_type=self._device.type, dtype=torch.bfloat16):
        result = self._model(...)
else:
    result = self._model(...)
```

CLI 建议：

```text
--precision {auto,fp32,bf16}
# auto: CUDA→bf16，NPU/CPU→fp32
```

### 修改 5：确认无 CUDA 可安装

**文件**：`setup.py` / `pyproject.toml`（若有）

xfold 本身无必须的 CUDA C++ 扩展。验收：`pip install -e .` 在无 CUDA 的外网 PC 上成功。

---

## 四、alphafold3 修改（优先级中）

官方仓只改数据管线 / C++ 扩展构建，不改 JAX 模型。

### 修改 6：放宽 Python 版本

**文件**：`pyproject.toml`  
**搜索**：`requires-python`

```toml
# 原
requires-python = ">=3.12"

# 改
requires-python = ">=3.11"
```

理由：vllm-ascend 容器为 Python 3.11.14；内网已在 3.11 上跑通数据管线与 cpp 扩展。

### 修改 7：CMake 支持离线依赖

**文件**：`CMakeLists.txt`  
**搜索**：`FetchContent_Declare`

上游声明名是 **`cifpp`**（仓库是 `pdb-redo/libcifpp`）。本地目录名与判断条件必须写死，避免 `cifpp` / `libcifpp` 对不上。

**约定本地目录结构**（`$ALPHAFOLD3_DEPS_DIR/`）：

```text
abseil-cpp/     # abseil/abseil-cpp @ d7aaad83b488fd62bd51c81ecf16cd938532cc0a
pybind11/       # pybind/pybind11 @ 2e0815278cb899b20870a67ca8205996ef47e70f (v2.12.0)
pybind11_abseil/# pybind/pybind11_abseil @ bddf30141f9fec8e577f515313caec45f559d319
libcifpp/       # pdb-redo/libcifpp @ ac98531a2fc8daf21131faa0c3d73766efa46180 (v7.0.3)
dssp/           # PDB-REDO/dssp @ 57560472b4260dc41f457706bc45fc6ef0bc0f10 (v4.4.7)
                # 若做了修改 9，此处必须是打过补丁的 dssp
```

示例（五个依赖同一模式）：

```cmake
set(DEPS_DIR "$ENV{ALPHAFOLD3_DEPS_DIR}" CACHE PATH
    "Directory containing pre-downloaded dependency sources")

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/abseil-cpp")
  FetchContent_Declare(abseil-cpp SOURCE_DIR "${DEPS_DIR}/abseil-cpp")
else()
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG d7aaad83b488fd62bd51c81ecf16cd938532cc0a
    EXCLUDE_FROM_ALL)
endif()

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/libcifpp")
  FetchContent_Declare(cifpp SOURCE_DIR "${DEPS_DIR}/libcifpp")
else()
  FetchContent_Declare(
    cifpp
    GIT_REPOSITORY https://github.com/pdb-redo/libcifpp
    GIT_TAG ac98531a2fc8daf21131faa0c3d73766efa46180)
endif()
```

有网且未设置 `ALPHAFOLD3_DEPS_DIR` 时，必须仍能走 GitHub FetchContent（不能改坏原路径）。

### 修改 8：链接 Boost::regex

**文件**：`CMakeLists.txt`  
**搜索**：`target_link_libraries`、`find_package(Python3`

**现状**：libcifpp 使用 `boost::regex`，官方 CMakeLists 未链接，内网 `cpp.so` import 报：

```text
undefined symbol: _ZN5boost16re_detail_10740014verify_optionsEjNS_15regex_constants12_match_flagsE
```

```cmake
find_package(Boost 1.74 REQUIRED COMPONENTS regex)

target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          # ... 保持原有其它库 ...
          cifpp::cifpp
          Boost::regex)
```

**外网编译机依赖**（文档必须写上，否则外网 `find_package` 会失败）：

```bash
# Debian/Ubuntu 示例
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev
```

Boost 版本以发行版可用为准；若找不到 1.74，可将 `1.74` 放宽为系统已装版本，但必须仍能链接 `Boost::regex`。

### 修改 9：dssp / mkdssp（P1，补丁打在依赖源码上）

**注意**：改的是 **dssp 第三方源码**，不是 alphafold3 仓库自己的逻辑。只交 AF3 的 diff **带不走** 这个修复。

内网问题：dssp 会编 `mkdssp` 并链接 `libmcfp::libmcfp`，新版 libmcfp 不提供该 target。

在 **`$ALPHAFOLD3_DEPS_DIR/dssp/CMakeLists.txt`**：

```cmake
option(DSSP_BUILD_MKDSSP "Build the mkdssp executable" ON)

if(DSSP_BUILD_MKDSSP)
  add_executable(mkdssp ${CMAKE_CURRENT_SOURCE_DIR}/src/mkdssp.cpp)
  target_link_libraries(mkdssp PRIVATE libmcfp::libmcfp dssp::dssp)
  # 保留原 install rules
endif()
```

alphafold3 配置时：

```bash
cmake -DDSSP_BUILD_MKDSSP=OFF ...
```

**交付物必须包含**：patched `dssp/` 目录，或针对官方 dssp tag 的 `.patch` 文件。

---

## 五、验证（分层，禁止混为一谈）

### 5.1 外网门禁（GPU PC 或仅 CPU 均可）

#### xfold

```bash
cd xfold
pip install -e .

# 无 CUDA 也必须能 import（triton 保护）
python -c "
from xfold.fastnn import config
print('fastnn impl:', config.dot_product_attention_implementation)
from xfold.alphafold3 import AlphaFold3
print('AlphaFold3 import OK')
"

# 权重探测
# 1) 仅 af3.bin.zst + 两个 npy → 能加载
# 2) 仅 af3.bin + 两个 npy → 能加载
# 3) 两者皆无 → FileNotFoundError
# 4) 仅 .zst 且未装 zstandard → 清晰 ImportError
# 5) 缺 fourier_weight.npy / fourier_bias.npy → 错误可理解

# 可选：CPU/GPU 一次极小 forward（官方 featurised pkl）
# 预期键：diffusion_samples / distogram / predicted_lddt / full_pae / full_pde
```

外网 **通过** 的含义：能装、能 import、权重路径逻辑正确。  
**不等于** NPU 精度验收通过。

#### alphafold3

```bash
# Python 3.11 下
cd alphafold3
pip install -e . --no-build-isolation

# 离线编译（有网机器模拟离线）
# 1. 五个依赖源码放到 deps/（dssp 用打过补丁的那份）
# 2. export ALPHAFOLD3_DEPS_DIR=/path/to/deps
# 3. cmake / build
# 4. python -c "import alphafold3.cpp; print('OK')"
#    不得再出现 boost undefined symbol

# 在线编译（不设 ALPHAFOLD3_DEPS_DIR）
# 必须仍能 FetchContent 成功，证明没改坏原路径
```

### 5.2 内网回归（NPU，外网不要声称已完成）

使用与此前相同的官方 test pkl / 参考输出：

| 项 | 期望 |
|----|------|
| 设备 | `npu`，fastnn 三个实现均为 `torch` |
| dtype | 无 float64 进 NPU matmul |
| 默认精度 | float32；完整配置耗时应接近此前 ~80s 量级（150 tokens，10 recycle / 5 sample / 200 step） |
| trunk | `contact_probs` vs JAX 参考均值差约 0.0011 量级 |
| import | `alphafold3.cpp` 无 boost 缺符号 |

不要求外网复现 79.8s 与 0.0011；这两项是内网回归门禁。

---

## 六、交付物

1. **锁定基线 SHA**（写在交付说明第一行）
2. **xfold 完整修改树**  
   - zip，或 `git format-patch` / `git diff <基线SHA>`
3. **alphafold3 修改**  
   - `pyproject.toml`、`CMakeLists.txt` 的 diff 即可
4. **dssp 补丁**（若做了修改 9）  
   - `dssp.patch` 或完整 patched `dssp/` 目录
5. **修改说明**（可按下表）

| 文件 | 改动 | 对应章节 | 优先级 |
|------|------|----------|--------|
| xfold/params.py | zstd 可选；自动选 af3.bin / af3.bin.zst | 修改 1 | P0 |
| xfold/run_alphafold.py | 设备抽象、fp64 转换、NPU 默认 fp32 | 修改 2、4 | P0 |
| xfold/fastnn/* | triton 可选 import + fallback | 修改 3 | P0 |
| alphafold3/pyproject.toml | requires-python >=3.11 | 修改 6 | P0 |
| alphafold3/CMakeLists.txt | 离线 DEPS_DIR + Boost::regex | 修改 7、8 | P0 |
| deps/dssp/CMakeLists.txt | DSSP_BUILD_MKDSSP | 修改 9 | P1 |

6. **外网验证记录**：命令、Python 版本、是否有 CUDA、通过/失败项  
   明确标注「未做 NPU 回归」

---

## 七、内网环境参考（回归用，非外网必达）

| 组件 | 版本 |
|------|------|
| 硬件 | Atlas 800I A2，Ascend 910B4 |
| OS | EulerOS 2.0 SP12 (aarch64) |
| Docker | quay.io/ascend/vllm-ascend:v0.18.0rc1 |
| Python | 3.11.14 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post1 |
| CANN | 8.5.1 |
| 额外 pip | absl-py, zstandard, rdkit, etils, tqdm, tokamax==0.0.12, dm-haiku==0.0.16, jax(CPU), einops |

---

## 八、给外网执行同学的最短清单

1. 向内网要 xfold / alphafold3 的 **精确 SHA**，checkout 后再改  
2. 按符号改，不按旧文档行号  
3. xfold 四件套：权重双格式、设备抽象、triton 保护、fp64→fp32 + NPU fp32  
4. 权重测试目录带上 `fourier_weight.npy` / `fourier_bias.npy`  
5. AF3：3.11、离线 FetchContent（`libcifpp/` 目录名）、Boost::regex、dssp 补丁单独交付  
6. 外网只签门禁；NPU 数字回内网用同一 pkl 回归  
