# AlphaFold3 昇腾 NPU 迁移 — 交付说明

## 基线 SHA（修改基于此提交，见 patch 文件名后缀）

- xfold      : 22bdeed (update license follow by alphafold v3.0.3)
- alphafold3 : c0f97ed (Add validation that templates is a list)

本仓库直接包含修改后的完整源码树（`xfold/`、`alphafold3/`）；
`deliverables/` 下另附相对基线的 `git diff` 补丁，便于核对或打给
上游 checkout 的源码树。

## 修改清单

| 文件 | 改动 | 对应章节 | 优先级 |
|------|------|----------|--------|
| xfold/xfold/params.py | zstd 可选；自动选 af3.bin / af3.bin.zst；fourier npy 缺失报清晰错误 | 修改 1 | P0 |
| xfold/run_alphafold.py | 设备抽象（NPU>CUDA>CPU，AF3_DEVICE 覆盖）、fp64→fp32、NPU 默认 fp32、--precision | 修改 2、4 | P0 |
| xfold/xfold/fastnn/* | triton 可选 import + torch fallback；config 自动检测 | 修改 3 | P0 |
| alphafold3/pyproject.toml | requires-python >=3.11 | 修改 6 | P0 |
| alphafold3/CMakeLists.txt | 离线 ALPHAFOLD3_DEPS_DIR + Boost::regex | 修改 7、8 | P0 |
| alphafold3/patches/dssp-v4.4.7-mkdssp-option.patch | DSSP_BUILD_MKDSSP 开关（打在 dssp 第三方源码上） | 修改 9 | P1 |

## 服务器部署要点

```bash
# xfold NPU 推理（auto 精度在 NPU 上即 fp32）
python run_alphafold.py ... --precision auto
AF3_DEVICE=npu:5 python run_alphafold.py ...   # 指定卡号，默认 npu:0

# alphafold3 离线编译（$ALPHAFOLD3_DEPS_DIR 含
# abseil-cpp/ pybind11/ pybind11_abseil/ libcifpp/ dssp/，
# 其中 dssp/ 需先 git apply patches/dssp-v4.4.7-mkdssp-option.patch）
export ALPHAFOLD3_DEPS_DIR=/path/to/deps
pip install -e . --no-build-isolation
python -c "import alphafold3.cpp; print('OK')"   # 不得再出现 boost 缺符号

# 外网编译机依赖（Debian/Ubuntu）
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev
```

## 验证状态

- 外网门禁：Python 语法检查（py_compile）全部通过；dssp 补丁已
  git-apply 往返校验，与目标结果逐字节一致；未做完整 pip install /
  import / 编译验证。
- NPU 回归：未做（需内网用同一官方 test pkl 回归：
  contact_probs 均值差 ~0.0011、fp32 完整配置 ~80s 量级）。
