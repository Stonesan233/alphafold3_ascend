# alphafold3-ascend 部署指南

AlphaFold 3 昇腾 NPU 推理部署（xfold PyTorch 实现 + torch_npu）。
在 Atlas 800I A2 上从零拉起 OpenAI 兼容推理服务的端到端流程。

> 本文档面向部署人员，按章节顺序执行即可完成部署。
> 技术选型、代码改动明细见 `AF3_NPU_code_change_requirements.md`
> 与 `deliverables/README.md`。

---

## 一、数据与权重下载（代码之外）

以下文件不属于代码仓，需单独获取并放置到部署机的数据目录
（下文以 `/mnt/weight/alphafold3` 为例，可按实际调整）。

### 1.1 模型权重

| 文件 | 下载地址 | 大小 | 放置位置 |
|------|----------|------|----------|
| af3.bin.zst | https://storage.googleapis.com/alphafold3/af3.bin.zst | 1.02 GB | `/mnt/weight/alphafold3/af3.bin.zst` |
| fourier_weight.npy | 由 alphafold3-decoded 项目提供（`data/params/diff_fourier_weight.pt` 转换） | 1.2 KB | `/mnt/weight/alphafold3/fourier_weight.npy` |
| fourier_bias.npy | 同上（`diff_fourier_bias.pt` 转换） | 1.2 KB | `/mnt/weight/alphafold3/fourier_bias.npy` |

说明：
- 权重受 [AlphaFold 3 模型参数使用条款](alphafold3/WEIGHTS_TERMS_OF_USE.md) 约束，
  需从 Google 官方渠道获取。
- 服务支持 `af3.bin`（解压后）与 `af3.bin.zst`（原始压缩）两种格式自动检测，
  二选一即可，推荐直接使用原始 `.zst`。
- fourier 两个 npy 文件很小，随数据包分发即可。

### 1.2 CCD 化学组件字典

| 文件 | 下载地址 | 大小 | 放置位置 |
|------|----------|------|----------|
| components.cif.gz | https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz | 118 MB（解压 514 MB） | 解压后放 `/mnt/weight/alphafold3/components.cif` |

用于生成 CCD pickle（数据管线必需，见第四章 4.3 节）。

### 1.3 遗传数据库（完整数据管线需要）

数据源均为 Google 镜像：
`https://storage.googleapis.com/alphafold-databases/v3.0/`

| 文件 | 大小（解压后） | 用途 |
|------|---------------|------|
| pdb_2022_09_28_mmcif_files.tar.zst | ~238 GB（19.6 万个 cif） | 模板结构 |
| uniref90_2022_05.fa | 67 GB | 蛋白 MSA |
| bfd-first_non_consensus_sequences.fasta | 17 GB | 蛋白 MSA |
| mgy_clusters_2022_05.fa | 120 GB | 蛋白 MSA |
| uniprot_all_2021_04.fa | 101 GB | 配对 MSA |
| nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta | 75 GB | RNA MSA |
| rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta | 0.2 GB | RNA MSA |
| rnacentral_active_seq_id_90_cov_80_linclust.fasta | 13 GB | RNA MSA |
| pdb_seqres_2022_09_28.fasta | 0.2 GB | 模板序列 |

总下载量约 143 GB，解压后约 630 GB。统一放置到
`/mnt/weight/alphafold3/databases/`（mmcif 解压到其下 `mmcif_files/`）。

> 数据库仅需跑完整数据管线（输入氨基酸序列 → MSA/模板搜索 → 特征化）时使用。
> 若只做预特征化 pkl 的推理验证，可暂不准备。

### 1.4 目标目录结构

全部就位后：

```text
/mnt/weight/alphafold3/
├── alphafold3-ascend/          # 本仓库代码（git clone / 解压）
├── af3.bin.zst                 # 模型权重
├── fourier_weight.npy
├── fourier_bias.npy
├── components.cif              # CCD 字典（514 MB）
└── databases/                  # 遗传数据库
    ├── mmcif_files/
    ├── uniref90_2022_05.fa
    ├── bfd-first_non_consensus_sequences.fasta
    ├── mgy_clusters_2022_05.fa
    ├── uniprot_all_2021_04.fa
    ├── nt_rna_*.fasta
    ├── rfam_*.fasta
    ├── rnacentral_*.fasta
    └── pdb_seqres_2022_09_28.fasta
```

---

## 二、启动 Docker 容器

部署使用 vllm-ascend 官方镜像（自带 CANN / torch / torch_npu）。
以下假设使用 NPU 4（按现场空闲卡调整 `/dev/davinci4`）。

```bash
IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-a3
CONTAINER=af3-xfold
WEIGHT_DIR=/mnt/weight/alphafold3

docker run -d \
  --restart unless-stopped \
  --name ${CONTAINER} \
  --network host \
  --shm-size 1g \
  --privileged \
  --device /dev/davinci4 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v ${WEIGHT_DIR}:${WEIGHT_DIR} \
  --entrypoint /bin/bash \
  ${IMAGE} \
  -lc 'sleep infinity'
```

要点：
- `--device /dev/davinci4`：把目标卡透给容器，容器内即 `npu:0`
- 驱动相关挂载（dcmi / npu-smi / driver/lib64）缺一会报
  `libascend_hal.so not found`
- 权重目录整目录挂载，代码、权重、数据库容器内均可见

进入容器并验证 NPU：

```bash
docker exec -it ${CONTAINER} bash

python3 -c "
import torch, torch_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('NPU available:', torch.npu.is_available())
"
# 预期：NPU available: True（LD_PRELOAD 警告可忽略）

npu-smi info   # 应只可见挂载的 NPU 4
```

---

## 三、安装依赖与编译 C++ 扩展

### 3.1 系统包

```bash
# 编译依赖（Boost 用系统 1.74 即可）
apt-get install -y libboost-regex-dev cmake ninja-build python3-dev
```

> apt 源按客户现场内源镜像自行配置（aarch64 平台对应 ubuntu-ports 路径）。

### 3.2 Python 依赖

```bash
# 运行依赖。注意：tokamax 必须钉 0.0.12，
# >=0.1.0 移除了 DotProductAttentionImplementation，会破坏 import
pip install fastapi uvicorn pydantic zstandard rdkit etils absl-py tqdm
pip install tokamax==0.0.12
```

> pip 源按客户现场内源镜像自行配置（`~/.pip/pip.conf` 或
> `pip install -i <镜像地址>`）。

### 3.3 安装 xfold

```bash
# 代码目录由宿主机用户 clone，容器内 git 因属主不一致会报
# "dubious ownership"（xfold 版本号经 setuptools_scm 读 git，必须先加白名单）
git config --global --add safe.directory /mnt/weight/alphafold3/alphafold3-ascend

cd /mnt/weight/alphafold3/alphafold3-ascend/xfold
pip install -e . --no-build-isolation
```

（会连带安装 jax / dm-haiku 的 CPU 版，NPU 环境不会拉 CUDA 包）

### 3.4 编译 alphafold3 C++ 扩展（离线编译）

```bash
cd /mnt/weight/alphafold3/alphafold3-ascend

# build backend（--no-build-isolation 模式下需手动安装，否则报
# "Cannot import 'scikit_build_core.build'"）
pip install scikit_build_core

# 依赖源码已预打补丁在 deps/ 目录，无需任何手工 patch。
# 唯一前置操作：预放置 CCD 文件（必须非空，空文件会被 libcifpp 删除重下）
cp /mnt/weight/alphafold3/components.cif deps/libcifpp/rsrc/components.cif
ls -lh deps/libcifpp/rsrc/components.cif   # 应约 514MB，禁止 0 字节

# 设置离线依赖目录。
# 变量名是 ALPHAFOLD3_DEPS_DIR（敲成 ALPHAFAFOLD3 会静默走在线下载，
# 顶层 CMake 对空值有 WARNING 兜底）
export ALPHAFOLD3_DEPS_DIR=$PWD/deps

cd alphafold3
pip install -e . --no-build-isolation \
  --config-settings="cmake.args=-DBUILD_TESTING=OFF;-DCIFPP_DOWNLOAD_CCD=OFF"
# 顶层 CMake 已对相关开关做 OFF FORCE，--config-settings 为双保险，可省略但建议保留
```

编译约 5~15 分钟（abseil + libcifpp 全量编译，314 个编译单元）。

> 注意：cpp.so 按容器 Python ABI 编译，**3.11 与 3.12 的产物不能混用**，
> 换 Python 版本的容器必须重编。

### 3.5 验收

```bash
# cpp.so 运行时通过 LIBCIFPP_DATA_DIR 定位 components.cif，
# 未设会报 "Could not find the libcifpp components.cif file"
export LIBCIFPP_DATA_DIR=/mnt/weight/alphafold3/alphafold3-ascend/deps/libcifpp/rsrc

python3 -c "import alphafold3.cpp; print('OK', [x for x in dir(alphafold3.cpp) if not x.startswith('_')])"
# 预期：OK + 13 个子模块（cif_dict、fasta_iterator、msa_conversion 等）
# 不得出现 boost undefined symbol；断网时不得访问 github / gitlab / wwpdb
```

> `LIBCIFPP_DATA_DIR` 后续（生成 CCD pickle、跑推理服务）均需要，
> 建议写入 `~/.bashrc` 一劳永逸。

---

## 四、生成 CCD pickle 与权重准备

### 4.1 环境变量

```bash
# 3.5 节已设置过；若换了 shell 会话需重新 export
export LIBCIFPP_DATA_DIR=/mnt/weight/alphafold3/alphafold3-ascend/deps/libcifpp/rsrc
```

（后续所有 Python 操作均需要；CANN 环境容器已自带，无需额外 source）

### 4.2 生成 CCD pickle（约 1~2 分钟）

```bash
cd /mnt/weight/alphafold3/alphafold3-ascend/alphafold3/src

python3 -c "
from alphafold3.constants.converters.ccd_pickle_gen import main
main(['g', '/mnt/weight/alphafold3/components.cif',
      'alphafold3/constants/converters/chemical_components.pickle'])"

# 建立文件名兼容软链 + 生成 sets pickle
ln -sf chemical_components.pickle alphafold3/constants/converters/ccd.pickle
python3 -c "
from alphafold3.constants.converters.chemical_component_sets_gen import main
main(['g', 'alphafold3/constants/converters/chemical_component_sets.pickle'])"
```

### 4.3 权重目录检查

```bash
ls -la /mnt/weight/alphafold3/af3.bin.zst \
       /mnt/weight/alphafold3/fourier_weight.npy \
       /mnt/weight/alphafold3/fourier_bias.npy
```

三个文件齐即可，服务启动时自动完成 Haiku → PyTorch 权重转换与加载
（无需单独的转换步骤，约 30 秒）。

### 4.4 验证数据管线（可选）

```bash
python3 -c "
import sys, pickle
sys.path.insert(0, '/mnt/weight/alphafold3/alphafold3-ascend/alphafold3/src')
with open('/mnt/weight/alphafold3/alphafold3-ascend/alphafold3/src/alphafold3/test_data/featurised_example.pkl', 'rb') as f:
    batch = pickle.load(f)
print('PKL loaded OK:', type(batch), len(batch))"
# 预期：PKL loaded OK: <class 'list'> 1
```

---

## 五、拉起推理服务

### 5.1 启动（冒烟配置）

```bash
cd /mnt/weight/alphafold3/alphafold3-ascend/service

# 路径与推理参数（冒烟用低参数，快速验证）
export AF3_SRC=/mnt/weight/alphafold3/alphafold3-ascend/alphafold3/src
export XFOLD_SRC=/mnt/weight/alphafold3/alphafold3-ascend/xfold
export AF3_MODEL_DIR=/mnt/weight/alphafold3
export LIBCIFPP_DATA_DIR=/mnt/weight/alphafold3/alphafold3-ascend/deps/libcifpp/rsrc
export AF3_NUM_RECYCLES=1 AF3_NUM_SAMPLES=1 AF3_DIFFUSION_STEPS=10

nohup python3 af3_service.py > /tmp/af3_service.log 2>&1 &

# 模型加载约 30~45 秒，等待后验证
sleep 45
```

关键环境变量说明（均有默认值，完整清单见 `service/README.md`）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `AF3_MODEL_DIR` | `/home` | 权重目录（af3.bin(.zst) + fourier npy） |
| `AF3_PRECISION` | `fp32` | 昇腾上 fp32 比 bf16 更快、置信度更好 |
| `AF3_NUM_RECYCLES` / `AF3_NUM_SAMPLES` / `AF3_DIFFUSION_STEPS` | 10 / 5 / 200 | 官方完整配置；冒烟可设 1/1/10 |

### 5.2 切换生产配置

冒烟通过后，去掉低参数覆盖重启即为官方完整配置
（10 recycle / 5 sample / 200 step，150 tokens 约 80 秒/次）：

```bash
kill %1   # 停掉冒烟进程
unset AF3_NUM_RECYCLES AF3_NUM_SAMPLES AF3_DIFFUSION_STEPS
nohup python3 af3_service.py > /tmp/af3_service.log 2>&1 &
```

---

## 六、冒烟测试与验收

### 6.1 健康检查

```bash
curl -s http://localhost:9800/health
```

预期返回：

```json
{"status":"ok","model":"alphafold3-3.0.4","device":"npu:0","precision":"fp32",
 "npu_available":true,"parameters":368384666}
```

### 6.2 推理冒烟（内置测试数据）

```bash
curl -s -X POST http://localhost:9800/predict/test
```

预期返回（关键指标）：

```json
{"status":"success","inference_time_sec":~1.1,
 "result_summary":{"tensors":{"predicted_lddt":{"shape":[1,150,24],...},
   "full_pae":{"shape":[1,150,150],...},...},
 "metrics":{"predicted_lddt_mean":~83.6}}
```

验收基线（现场实测，A3-syn-31 / NPU 4 / fp32）：

| 指标 | 基线值 |
|------|--------|
| cpp.so import | OK，13 个子模块，无 boost 缺符号 |
| 冒烟推理时间（1/1/10 配置） | ~1.1 s |
| predicted_lddt_mean | ~83.6 |
| 参数量 | 368,384,666 |
| 完整配置（10/5/200） | ~80 s |

### 6.3 OpenAI SDK 调用验证

```bash
pip install openai
python3 -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:9800/v1', api_key='none')
resp = client.chat.completions.create(
    model='alphafold3-3.0.4',
    messages=[{'role': 'user', 'content':
        '/mnt/weight/alphafold3/alphafold3-ascend/alphafold3/src/alphafold3/test_data/featurised_example.pkl'}])
print(resp.choices[0].message.content)"
```

> 服务接口接受的是**预特征化 pkl 路径**（不是原始氨基酸序列）。
> 从序列出发的完整流程（MSA → 模板 → 特征化 → 推理）用
> `xfold/run_alphafold.py`，需要第一章 1.3 节的数据库与 HMMER 工具。

### 6.4 API 一览

| 端点 | 说明 |
|------|------|
| `GET /health` | 服务与模型状态 |
| `GET /v1/models` | OpenAI 模型列表 |
| `POST /v1/chat/completions` | OpenAI 兼容；`messages[-1].content` 传预特征化 pkl 路径 |
| `POST /predict` | `{"pkl_path": "..."}` |
| `POST /predict/test` | 内置测试数据冒烟 |

---

## 七、故障排查

| 现象 | 原因与处理 |
|------|-----------|
| `libascend_hal.so not found` | 容器缺驱动挂载，检查 dmesg/npud-smi/driver/lib64 三个 -v |
| 安装 xfold 报 `fatal: detected dubious ownership` | 容器内 git 属主与宿主机不一致；`git config --global --add safe.directory <仓库路径>`（见 3.3 节） |
| 装 alphafold3 报 `Cannot import 'scikit_build_core.build'` | `--no-build-isolation` 下 build backend 需手动装：`pip install scikit_build_core`（见 3.4 节） |
| pip 找不到 setuptools / 装包巨慢 | 容器内 pip 走了外网；确认 pip 源已指向客户内源镜像（`~/.pip/pip.conf`） |
| pip 报 `--config-settings` 不存在 | 用了系统旧 pip；改用 `/usr/local/python3*/bin/pip` 或改设 `export CMAKE_ARGS="-DBUILD_TESTING=OFF"` |
| CMake configure 阶段卡在 github.com / gitlab.com / wwpdb | `ALPHAFOLD3_DEPS_DIR` 未设或敲错（多敲了个 F），或 components.cif 是空文件被删重下 |
| 编译报 `'unreachable' is not a member of 'std'` | deps 里的 libmcfp 版本不对（2.0.4 需 C++23）；必须用 1.3.4 |
| 编译报 `fatal error: Core: No such file` | deps/eigen 不完整（缺 Eigen/Core 入口或 src/Core 目录）；从官方 eigen-3.4.0.zip 补齐 |
| `import alphafold3.cpp` 报 boost undefined symbol | 容器缺 `libboost-regex-dev`；装后重编 |
| `import alphafold3.cpp` 报 `Could not find the libcifpp components.cif file` | `LIBCIFPP_DATA_DIR` 未设；指向 `deps/libcifpp/rsrc`（见 3.5 节） |
| 服务启动 40 秒左右退出 | 查 `/tmp/af3_service.log`；权重目录三个文件是否齐全 |
| `NameError: ChatRequest is not defined` | af3_service.py 版本过旧（Pydantic 类在函数内），git pull 拿最新 |
| import 报 tokamax 相关错误 | tokamax 版本不对；必须 `pip install tokamax==0.0.12` |
| 换了容器后 import alphafold3.cpp 失败 | cpp.so 与容器 Python 小版本不匹配（3.11≠3.12）；重编 |

---

## 八、环境参考

| 组件 | 版本 |
|------|------|
| 硬件 | Atlas 800I A2，Ascend 910B4 |
| 镜像 | quay.io/ascend/vllm-ascend:v0.23.0-a3（Python 3.12 / torch 2.10 / torch_npu 2.10.0.post4） |
| 已验证环境 | A3-syn-31，NPU 4，EulerOS 宿主机 |
| 历史验证环境 | 189 开发机（v0.18.0rc1 镜像，Python 3.11 / torch 2.9），精度基线同 |

> 精度基线：`contact_probs` 与 JAX 参考输出均值差 ~0.0011（权重转换正确性）；
> fp32 推理优于 bf16（更快且置信度更好，昇腾实测）。

## 九、精度测试
TODO

## 十、性能测试
TODO

## 许可

alphafold3 源码遵循 Apache License 2.0；模型参数使用受
[WEIGHTS_TERMS_OF_USE.md](alphafold3/WEIGHTS_TERMS_OF_USE.md) 约束。
****
