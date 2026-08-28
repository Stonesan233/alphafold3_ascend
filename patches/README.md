# 依赖补丁（仅供参考 / 审计）

**部署不需要执行本目录任何操作。** 仓库 `deps/` 目录与 `deps.tar.gz` 中的
`libcifpp/`、`dssp/` 已经打好全部补丁，客户现场只需：

```bash
tar xzf deps.tar.gz
export ALPHAFOLD3_DEPS_DIR=$PWD/deps
cd alphafold3 && pip install -e . --no-build-isolation
```

本目录补丁仅用于审计改动内容，或在从官方上游源码自行准备依赖时使用
（`patch -p1`，在对应依赖源码根目录执行）：

| 补丁 | 对象 / 版本 | 内容 |
|------|-------------|------|
| `libcifpp-boost-eigen.patch` | pdb-redo/libcifpp @ `ac98531a` (v7.0.3) | ① `find_package(Boost 1.80` → `1.74`（现场常见 1.74）；② eigen FetchContent（gitlab，离线不可达）改为 `DEPS_DIR`/`ALPHAFOLD3_DEPS_DIR` 存在时用本地 `SOURCE_DIR`，否则保留原 git 路径 |
| `dssp-mkdssp-offline.patch` | PDB-REDO/dssp @ `5756047` (v4.4.7) | ① 新增 `DSSP_BUILD_MKDSSP` option（默认 ON，alphafold3 顶层默认传 OFF），mkdssp target 及其 install rules 包进 if；② libmcfp、libcifpp 的 FetchContent 在 `DEPS_DIR` 存在时改本地 `SOURCE_DIR` |

```bash
# 用法示例（从官方源码开始时）
cd libcifpp && patch -p1 < ../patches/libcifpp-boost-eigen.patch
cd dssp     && patch -p1 < ../patches/dssp-mkdssp-offline.patch
```

注意：官方 zip/tar 解压后的目录名带 commit hash（如
`libcifpp-ac98531a.../`），需改名为短目录名 `libcifpp/`，与 alphafold3
顶层 `CMakeLists.txt` 的 `EXISTS "${DEPS_DIR}/libcifpp"` 判断对齐。
`deps/` 目录树（含 `eigen/` 3.4.0 与 `libmcfp/` v2.0.4）已按此命名。
