# AlphaFold3-Ascend 离线依赖包（deps.tar.gz）追加需求

> 对象：外网 PC / https://github.com/Stonesan233/alphafold3_ascend  
> 目的：客户现场 **零 GitHub、零手工 patch、零 git apply**，一条命令编过 `alphafold3.cpp`。  
> 背景：内网 189 已用 zip 解压的源码编过；zip **没有 `.git`**，`git apply` 会失败。依赖相关坑未全部进外网仓。

---

## 一、结论先说

当前仓只覆盖了 **dssp 关 mkdssp** 一类问题，不够。

客户现场正确形态：

```bash
tar xzf deps.tar.gz
export ALPHAFOLD3_DEPS_DIR=/path/to/deps
pip install -e alphafold3/ --no-build-isolation
```

不要让客户再对第三方 CMakeLists 做 sed / git apply。

**推荐方案 A**：分发已经改好、目录名已对齐的 `deps.tar.gz`（约 30–40MB 源码）。  
**不推荐方案 B**：CMake 配置阶段 `execute_process` 自动 patch（fragile）。

---

## 二、现状缺口

内网编译实际打过的补丁 vs 外网仓覆盖情况：

| 内网踩的坑 | 内网解法 | 外网仓是否覆盖 |
|------------|----------|----------------|
| libcifpp 要求 Boost ≥ 1.80，现场常见 1.74 | libcifpp `CMakeLists.txt`：`find_package(Boost 1.80` → `1.74` | ❌ 无 |
| libcifpp FetchContent eigen3（gitlab，离线不可达） | eigen `GIT_REPOSITORY` → 本地 `SOURCE_DIR` | ❌ 无 |
| dssp 同样 FetchContent eigen / libcifpp，断网失败 | 指向本地 `libcifpp/`、`eigen/` | ❌ 外网 dssp patch 只关了 mkdssp |
| zip 解压源码无 `.git`，`git apply` 失败 | 内网用 Python 直接改文件 | ⚠️ 若只交 `.patch` 且要求 `git apply` 则现场失败 |
| 目录名是 `abseil-cpp-<sha>/`，CMake 期望 `abseil-cpp/` | 内网写死带 hash 的 `SOURCE_DIR` | ⚠️ 需统一成短目录名 |

Boost regex **不要**打进 deps：用系统包 `libboost-regex-dev`（或 Euler/CentOS 等价包）。

---

## 三、目标目录（必须按此命名）

```text
alphafold3-ascend/
├── alphafold3/              # 已改官方代码（含 ALPHAFOLD3_DEPS_DIR / Boost::regex）
├── xfold/
├── service/
├── deps/                    # 本需求要交付的内容，也可单独 deps.tar.gz
│   ├── abseil-cpp/          # zip 解压后去掉 -<sha> 后缀
│   ├── pybind11/
│   ├── pybind11_abseil/
│   ├── libcifpp/            # 已打 Boost 1.74 + eigen 本地化
│   ├── dssp/                # 已打 mkdssp 开关 + libcifpp/eigen 本地化
│   └── eigen/               # 新增第 6 个：Eigen 3.4.0 源码
└── patches/                 # 仅作参考，客户部署不依赖
    ├── libcifpp-boost-eigen.patch
    └── dssp-mkdssp-offline.patch
```

顶层 `FetchContent` 五个（CMake 声明名 / 本地目录）：

| CMake 声明名 | 本地目录 | 上游 tag / commit |
|--------------|----------|-------------------|
| `abseil-cpp` | `deps/abseil-cpp` | `d7aaad83b488fd62bd51c81ecf16cd938532cc0a` |
| `pybind11` | `deps/pybind11` | `2e0815278cb899b20870a67ca8205996ef47e70f`（v2.12.0） |
| `pybind11_abseil` | `deps/pybind11_abseil` | `bddf30141f9fec8e577f515313caec45f559d319` |
| `cifpp` | `deps/libcifpp` | `ac98531a2fc8daf21131faa0c3d73766efa46180`（v7.0.3） |
| `dssp` | `deps/dssp` | `57560472b4260dc41f457706bc45fc6ef0bc0f10`（v4.4.7） |

额外（被 libcifpp / dssp FetchContent，**上一版需求文档漏了**）：

| 用途 | 本地目录 | 版本 |
|------|----------|------|
| Eigen3 | `deps/eigen` | 3.4.0（`libeigen/eigen` tag `3.4.0`） |

`ALPHAFOLD3_DEPS_DIR` 指向 `deps/` 这一层，不要指向某个子目录。

---

## 四、必须打进 deps 源码的修改

补丁一律用 **`patch -p1`** 可应用的 diff，或直接把改好的文件打进 tar。  
**禁止**把 `git apply` 写成唯一用法。

### 4.1 libcifpp/CMakeLists.txt（两处）

Boost：

```cmake
# 原
find_package(Boost 1.80 QUIET COMPONENTS regex)

# 改
find_package(Boost 1.74 QUIET COMPONENTS regex)
```

若还有非 QUIET / REQUIRED 的 `find_package(Boost 1.80`，一并改为 1.74。

Eigen FetchContent：

```cmake
# 原
FetchContent_Declare(my-eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)

# 改（DEPS_DIR 由上层 CMake 传入，或写相对/绝对本地路径）
FetchContent_Declare(my-eigen3
  SOURCE_DIR "${ALPHAFOLD3_DEPS_DIR}/eigen")
```

实现时注意：`ALPHAFOLD3_DEPS_DIR` 必须在配置 alphafold3 时传到子项目，或在 libcifpp 里写死为与 `deps/eigen` 的相对关系。推荐 alphafold3 顶层：

```cmake
set(DEPS_DIR "$ENV{ALPHAFOLD3_DEPS_DIR}" CACHE PATH
    "Pre-downloaded third-party sources")
```

并把 `DEPS_DIR` / `ALPHAFOLD3_DEPS_DIR` cache 传给子 CMake（`CMAKE_ARGS` 或编译前已改好的 libcifpp，使 `SOURCE_DIR` 为 `${DEPS_DIR}/eigen`）。

最省事、现场最稳：**deps 包里的 libcifpp 已改成 `SOURCE_DIR` 指向同级 `../eigen` 或 `${DEPS_DIR}/eigen`，客户不再改文件。**

### 4.2 dssp/CMakeLists.txt（三处）

1. **mkdssp 可关**（已有思路，必须进包）：

```cmake
option(DSSP_BUILD_MKDSSP "Build the mkdssp executable" ON)

if(DSSP_BUILD_MKDSSP)
  add_executable(mkdssp ...)
  target_link_libraries(mkdssp PRIVATE libmcfp::libmcfp dssp::dssp)
endif()
```

alphafold3 配置加：`-DDSSP_BUILD_MKDSSP=OFF`。  
更省事：deps 里的 dssp **直接删掉 / 注释掉** `add_executable(mkdssp ...)`（内网就是这么过的）。两种选一种，README 写清楚。

2. **libmcfp / libcifpp 不要再 git 拉**：

```cmake
# 原（示例）
FetchContent_Declare(... GIT_REPOSITORY https://github.com/mhekkel/libmcfp ...)
FetchContent_Declare(... GIT_REPOSITORY https://github.com/pdb-redo/libcifpp.git GIT_TAG v7.0.3)

# 改
# libcifpp → SOURCE_DIR ${DEPS_DIR}/libcifpp 或已 patch 的同级目录
```

现场 **只需要 dssp 静态库**。dssp 配置阶段无条件 FetchContent libmcfp，因此必须本地化：用 **v1.3.4**（1.3.x 为 C++17/20，g++ 11 可编；v2.0.4 需 C++23，现场编不过）放进 `deps/libmcfp` 并改 `SOURCE_DIR`。

3. **eigen**：dssp 若自己 FetchContent eigen，同样改到 `deps/eigen`。

### 4.3 alphafold3/CMakeLists.txt（仓内已应具备，请复核）

- `ALPHAFOLD3_DEPS_DIR` 存在且目录名为上表短名（`libcifpp` 不是 `cifpp`）。
- 五个官方依赖均为：有 `DEPS_DIR` 用 `SOURCE_DIR`，否则才 `GIT_REPOSITORY`。
- `find_package(Boost 1.74 REQUIRED COMPONENTS regex)` + `target_link_libraries(cpp PRIVATE ... Boost::regex)`。
- 配置传入 `-DDSSP_BUILD_MKDSSP=OFF`（若 dssp 仍保留 option）。

---

## 五、deps.tar.gz 制作步骤（给外网 PC）

```bash
# 1. 下载官方 zip（不要 git clone）
#    解压后把  <name>-<sha>/  改名为  <name>/

# 2. Eigen
#    https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
#    解压为 deps/eigen/（根目录能看到 Eigen/ 头文件即可）

# 3. 对 deps/libcifpp、deps/dssp 应用 4.1 / 4.2
#    优先：直接改文件后打包
#    备选：patch -p1 < patches/xxx.patch
#    cd deps/libcifpp && patch -p1 < ../../patches/libcifpp-boost-eigen.patch

# 4. 打包（不要带 .git，不要带编译产物）
tar czf deps.tar.gz deps/
```

体积预期约 30–40MB。不要把 Boost 源码打进去。

---

## 六、客户现场（README 必须写成这样）

系统包（Euler / Ubuntu 示例，按现场改包名）：

```bash
# Ubuntu/Debian
sudo apt-get install -y libboost-regex-dev cmake ninja-build python3-dev

# 现场已有 Boost 1.74 即可；不要再要求 1.80
```

编译：

```bash
tar xzf deps.tar.gz
export ALPHAFOLD3_DEPS_DIR=$PWD/deps
cd alphafold3
pip install -e . --no-build-isolation
```

验收：

```bash
python -c "import alphafold3.cpp; print('OK')"
# 不得出现 boost undefined symbol
# 不得访问 github.com / gitlab.com
```

断网模拟：配置阶段抓网络，不应再出现 `gitlab.com/libeigen` 或 `github.com/pdb-redo`。

---

## 七、交付物清单

| 项 | 要求 |
|----|------|
| `deps.tar.gz` | 短目录名；含 `eigen/`；`libcifpp`/`dssp` 已改好 |
| `patches/*.patch` | **可选**，`patch -p1` 格式，仅文档/审计用 |
| README 离线编译小节 | 只保留 tar + `ALPHAFOLD3_DEPS_DIR` + pip，无手工改第三方 |
| alphafold3 CMake | 复核 `DEPS_DIR` 短名、`Boost::regex`、`DSSP_BUILD_MKDSSP` |

不要交付：

- 只写「请 git clone 五个依赖再 git apply」
- 只有 dssp-mkdssp patch、没有 libcifpp Boost/eigen
- 目录名仍带 commit hash，和 CMake `EXISTS "${DEPS_DIR}/abseil-cpp"` 对不上

---

## 八、和旧文档的差异（避免外网漏项）

| 项 | 旧需求 | 本追加 |
|----|--------|--------|
| 依赖个数 | 5 个 FetchContent | **6 个目录：+ eigen** |
| libcifpp Boost | 只在顶层链 `Boost::regex` | **还要改 libcifpp 自己的 1.80→1.74** |
| eigen | 未单列 | **必须进 tar** |
| dssp | 只关 mkdssp | **还要本地化 libcifpp/eigen（及必要时 libmcfp）** |
| 打补丁方式 | 易写成 git apply | **改好再打包，或 patch -p1** |
| 客户动作 | 多步手工 | **解压 + 环境变量 + pip** |

---

## 九、给外网的最短任务列表

1. 按第三节短目录名准备 6 份源码（5 个 AF3 FetchContent + eigen 3.4.0）。  
2. 改 libcifpp：Boost 1.74 + eigen `SOURCE_DIR` 到 `deps/eigen`。  
3. 改 dssp：关/删 mkdssp；libcifpp、eigen（及需要时的 libmcfp）全部本地。  
4. 打 `deps.tar.gz`，自己在无外网环境跑一遍 `pip install -e alphafold3/`。  
5. 更新 README：现场零 patch。  
6. patch 文件可留在 `patches/`，但不是部署前置条件。
