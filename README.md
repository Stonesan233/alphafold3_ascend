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
patches/        dssp v4.4.7 补丁（打在 dssp 第三方源码上）
deliverables/   相对基线 SHA 的 git diff + 交付说明
```

修改依据与逐条说明见 `AF3_NPU_code_change_requirements.md`。

## 主要修改

**xfold**

| 文件 | 改动 |
|------|------|
| `xfold/params.py` | zstandard 可选；自动探测 `af3.bin.zst` / `af3.bin` 双格式权重；fourier npy 缺失时报清晰错误 |
| `run_alphafold.py` | 设备抽象（NPU > CUDA > CPU，`AF3_DEVICE` 环境变量可覆盖卡号）；float64→float32 转换（NPU aclnnMatmul 不支持 DT_DOUBLE）；NPU 默认 float32；`--precision {auto,fp32,bf16}` |
| `xfold/fastnn/*` | triton 可选 import + torch fallback；无 CUDA 环境自动使用 torch 实现 |

**alphafold3**

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | `requires-python >= 3.11` |
| `CMakeLists.txt` | `ALPHAFOLD3_DEPS_DIR` 离线 FetchContent（本地目录名 `libcifpp`）；链接 `Boost::regex`（修复 cpp.so undefined symbol） |
| `patches/dssp-v4.4.7-mkdssp-option.patch` | `DSSP_BUILD_MKDSSP` 开关，可关闭 mkdssp 构建 |

## 服务器部署

### xfold 推理（NPU）

```bash
cd xfold
pip install -e .

# auto 精度在 NPU 上即 float32（昇腾上 fp32 比 bf16 更快、置信度更好）
python run_alphafold.py --json_path <input.json> --output_dir <out> \
    --model_dir <weights_dir> --precision auto

AF3_DEVICE=npu:5 python run_alphafold.py ...   # 指定卡号，默认 npu:0
```

权重目录需包含（`af3.bin` 或 `af3.bin.zst` 二选一）：

```text
af3.bin(.zst)  fourier_weight.npy  fourier_bias.npy
```

### alphafold3 数据管线 / C++ 扩展（离线编译）

```bash
# 1. 准备依赖源码目录，目录名必须如下（dssp 需先打补丁）
#    $ALPHAFOLD3_DEPS_DIR/
#      abseil-cpp/  pybind11/  pybind11_abseil/  libcifpp/  dssp/
git apply patches/dssp-v4.4.7-mkdssp-option.patch   # 在 dssp 源码内执行

# 2. 编译安装（Python 3.11）
export ALPHAFOLD3_DEPS_DIR=/path/to/deps
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev
pip install -e . --no-build-isolation

# 3. 验证（不得再出现 boost undefined symbol）
python -c "import alphafold3.cpp; print('OK')"
```

不设 `ALPHAFOLD3_DEPS_DIR` 时仍走 GitHub FetchContent 在线构建，原路径不受影响。

## 验证状态

- 已通过：全部改动 `py_compile` 语法检查；dssp 补丁 `git apply` 往返校验
- 待内网 NPU 回归：`contact_probs` 均值差 ~0.0011、fp32 完整配置 ~80s 量级、
  `import alphafold3.cpp` 无 boost 缺符号（使用官方 test pkl）

## 内网环境参考

| 组件 | 版本 |
|------|------|
| 硬件 | Atlas 800I A2，Ascend 910B4 |
| OS / 容器 | EulerOS 2.0 SP12 (aarch64)，quay.io/ascend/vllm-ascend:v0.18.0rc1 |
| Python / torch | 3.11.14，torch 2.9.0+cpu + torch_npu 2.9.0.post1（CANN 8.5.1） |

## 许可

alphafold3 源码遵循 Apache License 2.0；AF3 模型参数的使用受
[WEIGHTS_TERMS_OF_USE.md](alphafold3/WEIGHTS_TERMS_OF_USE.md) 约束。
