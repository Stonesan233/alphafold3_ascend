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
| xfold/run_alphafold.py | 适配 alphafold3 3.0.4 API：base_model/Diffuser 改为 alphafold3.model.model（Model.get_inference_result / ModelResult / InferenceResult）；cached_ccd → Ccd；新增 --num_recycles/--diffusion_steps | 事实核查补充 | P0 |
| xfold/pyproject.toml | 补充 run_alphafold.py 运行时依赖 jax / dm-haiku / tokamax（CPU 版即可） | 事实核查补充 | P1 |
| alphafold3/pyproject.toml | requires-python >=3.11 | 修改 6 | P0 |
| alphafold3/CMakeLists.txt | 离线 ALPHAFOLD3_DEPS_DIR + Boost::regex | 修改 7、8 | P0 |
| alphafold3/patches/dssp-v4.4.7-mkdssp-option.patch | DSSP_BUILD_MKDSSP 开关（打在 dssp 第三方源码上） | 修改 9 | P1 |

### 事实核查修复说明（相对首版交付）

xfold 基线针对 alphafold3 3.0.2 时代 API 编写，官方 3.0.4 重构删除了
`alphafold3.model.components.base_model` 与 `alphafold3.model.diffusion.model`
（Diffuser），导致 run_alphafold.py 在 import 阶段即 ModuleNotFoundError。
内网未暴露是因为验证走的是 af3_service.py 路径（只 import
xfold.alphafold3 + xfold.params）。本次已按 3.0.4 API 修复
（get_inference_result 等价物位于 model.py:356），并核实了
featurise_input / DataPipelineConfig / write_output / folding_input /
`__identifier__` 契约均与 3.0.4 一致。

部署注意（非 bug）：

- run_alphafold.py 完整数据管线需要 `chemical_components.pickle`（约 546MB）
  与 `components.cif`，由 build_data / ccd_pickle_gen.py 生成，勿遗漏。
- 快速冒烟可降低参数：`--num_recycles 1 --diffusion_steps 20`。

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

- 外网门禁：Python 语法检查（py_compile）全部通过；run_alphafold.py 对
  alphafold3 3.0.4 的全部 API 引用已逐一静态核实（import 均可解析）；
  dssp 补丁已 git-apply 往返校验，与目标结果逐字节一致；未做完整
  pip install / 端到端推理 / 编译验证。
- NPU 回归：未做（需内网用同一官方 test pkl 回归：
  contact_probs 均值差 ~0.0011、fp32 完整配置 ~80s 量级）。
- af3_service.py 已按部署手册重写入库（service/af3_service.py），
  相对内网冒烟版：路径/参数环境变量化、默认 fp32、完整配置默认值、
  推理加锁、sys.executable 子进程，待内网回归。
