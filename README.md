# alphafold3-ascend

AlphaFold 3 (xfold / alphafold3) 昇腾 NPU 迁移适配。

在 Atlas 800I A2（Ascend 910B4，torch 2.9 + torch_npu 2.9.0.post1）上跑通
AF3 完整推理：float32 约 79.8s / bfloat16 约 97.4s（150 tokens，5 samples ×
200 diffusion steps），`contact_probs` 与 JAX 参考输出均值差约 0.0011。

## 仓库结构

```text
xfold/          PyTorch 版 AF3（基线 22bdeed），已含 NPU 适配修改
alphafold3/     官方仓（基线 c0f97ed），仅改数据管线 / C++ 扩展构建，
                不改 JAX 模型
service/        af3_service.py — OpenAI 兼容 FastAPI 推理服务（NPU）
deps/           离线依赖源码（7 个，已预打补丁，见下表）
patches/        依赖补丁（仅供参考 / 审计，部署不依赖）
deliverables/   相对基线 SHA 的 git diff + 交付说明
```

修改依据与逐条说明见 `AF3_NPU_code_change_requirements.md`。

## 主要修改

**xfold**

| 文件 | 改动 |
|------|------|
| `xfold/params.py` | zstandard 可选；自动探测 `af3.bin.zst` / `af3.bin` 双格式权重；fourier npy 缺失时报清晰错误 |
| `run_alphafold.py` | 设备抽象（NPU > CUDA > CPU，`AF3_DEVICE` 环境变量可覆盖卡号）；float64→float32 转换（NPU aclnnMatmul 不支持 DT_DOUBLE）；NPU 默认 float32；`--precision {auto,fp32,bf16}`、`--num_recycles`、`--diffusion_steps` |
| `run_alphafold.py` | 适配 alphafold3 3.0.4 API：`Diffuser`/`base_model`（3.0.2 时代，已被官方删除）替换为 `alphafold3.model.model` 的 `Model.get_inference_result` / `ModelResult` / `InferenceResult`；`cached_ccd` → `Ccd` |
| `xfold/fastnn/*` | triton 可选 import + torch fallback；无 CUDA 环境自动使用 torch 实现 |
| `pyproject.toml` | 补充 `jax` / `dm-haiku` / `tokamax`（run_alphafold.py 经 alphafold3 import 链需要，CPU 版即可） |

**alphafold3**

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | `requires-python >= 3.11` |
| `CMakeLists.txt` | `ALPHAFOLD3_DEPS_DIR` 离线 FetchContent（本地目录名 `libcifpp`）；链接 `Boost::regex`（>=1.74，修复 cpp.so undefined symbol）；`DSSP_BUILD_MKDSSP` 默认 OFF |
| `deps/libcifpp`（预打补丁） | Boost 1.80→1.74；eigen FetchContent 改本地 `SOURCE_DIR` |
| `deps/dssp`（预打补丁） | mkdssp 可关；libmcfp/libcifpp FetchContent 改本地 `SOURCE_DIR` |
| `patches/`（根目录） | libcifpp / dssp 补丁（`patch -p1` 格式，仅审计用） |

## 服务器部署

### xfold 推理（NPU）

```bash
cd xfold
pip install -e .

# run_alphafold.py 还依赖 alphafold3 Python 包（数据管线 + cpp 扩展），
# 见下节；jax / dm-haiku / tokamax 已在 pyproject 中声明（CPU 版即可）。
python run_alphafold.py --json_path <input.json> --output_dir <out> \
    --model_dir <weights_dir> --precision auto

AF3_DEVICE=npu:5 python run_alphafold.py ...   # 指定卡号，默认 npu:0

# 快速冒烟（降低循环与扩散步数）
python run_alphafold.py ... --num_recycles 1 --diffusion_steps 20
```

注意：走完整数据管线需要 CCD 数据文件 `chemical_components.pickle`
（约 546MB）与 `components.cif`，由 `build_data` / `ccd_pickle_gen.py`
生成，部署时勿遗漏。
```

权重目录需包含（`af3.bin` 或 `af3.bin.zst` 二选一）：

```text
af3.bin(.zst)  fourier_weight.npy  fourier_bias.npy
```

### alphafold3 数据管线 / C++ 扩展（离线编译，零手工 patch）

客户现场三步（依赖已预改好打包在 `deps/`，无需 git apply / sed）：

```bash
# 0. 系统包（Boost regex 用系统包，不打进 deps；已有 1.74 即可，不要求 1.80）
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev

# 1. 解压依赖（或直接用仓库内 deps/ 目录）
tar xzf deps.tar.gz        # 仓库内执行 tar czf deps.tar.gz deps 可重新生成

# 2. 编译安装（Python 3.11）
export ALPHAFOLD3_DEPS_DIR=$PWD/deps
cd alphafold3
pip install -e . --no-build-isolation

# 3. 验收
python -c "import alphafold3.cpp; print('OK')"
#   不得出现 boost undefined symbol
#   断网配置阶段不得访问 github.com / gitlab.com
```

`deps/`（7 个目录，均短目录名，已含全部补丁）：

| 目录 | 上游 / 版本 | 预打补丁 |
|------|-------------|----------|
| `abseil-cpp/` | abseil @ `d7aaad83` | — |
| `pybind11/` | pybind @ `2e081527` (v2.12.0) | — |
| `pybind11_abseil/` | pybind @ `bddf3014` | — |
| `libcifpp/` | pdb-redo @ `ac98531a` (v7.0.3) | Boost 1.80→1.74；eigen 改本地 `SOURCE_DIR` |
| `dssp/` | PDB-REDO @ `5756047` (v4.4.7) | mkdssp 可关（alphafold3 默认 OFF）；libmcfp/libcifpp 改本地 `SOURCE_DIR` |
| `eigen/` | libeigen 3.4.0（仅头文件使用，已裁剪） | — |
| `libmcfp/` | mhekkel @ v2.0.4（仅 dssp 配置阶段需要） | — |

不设 `ALPHAFOLD3_DEPS_DIR` 时仍走 GitHub FetchContent 在线构建，原路径不受影响。
补丁内容见 `patches/`（`patch -p1` 格式，仅供审计，非部署前置）。

### 推理服务（OpenAI 兼容 API）

内网验证路线已正式入库：`service/af3_service.py`（FastAPI，端口 9800，
`/v1/chat/completions`、`/predict`、`/predict/test`）。相对内网冒烟版的改进：
路径/参数全部环境变量化（`AF3_SRC`、`AF3_MODEL_DIR`、`AF3_PRECISION` 等）、
**默认 fp32**（与仓库 NPU 精度策略一致，正式压测无需改代码）、完整配置
10 recycle / 5 sample / 200 step（冒烟可 env 覆盖）、推理全局锁串行化、
子进程加载用 `sys.executable`。详见 `service/README.md`。

部署手册：`AF3_Ascend_NPU_field_deployment_handbook.md`（现场从零部署步骤
与故障排查）。

## 验证状态

- 已通过：全部改动 `py_compile` 语法检查；run_alphafold.py 引用的
  alphafold3 3.0.4 API 已逐一静态核实；dssp 补丁 `git apply` 往返校验
- 待内网 NPU 回归：`contact_probs` 均值差 ~0.0011、fp32 完整配置 ~80s 量级、
  `import alphafold3.cpp` 无 boost 缺符号、run_alphafold.py 端到端与
  af3_service.py（官方 test pkl / input）

## 内网环境参考

| 组件 | 版本 |
|------|------|
| 硬件 | Atlas 800I A2，Ascend 910B4 |
| OS / 容器 | EulerOS 2.0 SP12 (aarch64)，quay.io/ascend/vllm-ascend:v0.18.0rc1 |
| Python / torch | 3.11.14，torch 2.9.0+cpu + torch_npu 2.9.0.post1（CANN 8.5.1） |

## 许可

alphafold3 源码遵循 Apache License 2.0；AF3 模型参数的使用受
[WEIGHTS_TERMS_OF_USE.md](alphafold3/WEIGHTS_TERMS_OF_USE.md) 约束。
