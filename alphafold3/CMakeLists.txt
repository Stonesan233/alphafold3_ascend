# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

cmake_minimum_required(VERSION 3.28)

# This forces Git to use the 'files' backend for all FetchContent operations.
# This fixes libcifpp and dssp incompatibility with newer git versions.
set(ENV{GIT_CONFIG_PARAMETERS} "'init.defaultRefFormat=files'")

project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(ABSL_PROPAGATE_CXX_STD ON)

# Remove support for scan deps, which is only useful when using C++ modules.
unset(CMAKE_CXX_SCANDEP_SOURCE)

# Pre-downloaded dependency sources for offline builds.
# Set ALPHAFOLD3_DEPS_DIR to a directory containing:
#   abseil-cpp/      (abseil/abseil-cpp @ d7aaad83b488fd62bd51c81ecf16cd938532cc0a)
#   pybind11/        (pybind/pybind11 @ 2e0815278cb899b20870a67ca8205996ef47e70f, v2.12.0)
#   pybind11_abseil/ (pybind/pybind11_abseil @ bddf30141f9fec8e577f515313caec45f559d319)
#   libcifpp/        (pdb-redo/libcifpp @ ac98531a2fc8daf21131faa0c3d73766efa46180, v7.0.3;
#                     pre-patched: Boost >= 1.74, eigen from ${DEPS_DIR}/eigen)
#   dssp/            (PDB-REDO/dssp @ 57560472b4260dc41f457706bc45fc6ef0bc0f10, v4.4.7;
#                     pre-patched: optional mkdssp, libmcfp/libcifpp from ${DEPS_DIR})
#   eigen/           (libeigen/eigen @ 3.4.0, header-only usage)
#   libmcfp/         (mhekkel/libmcfp @ v2.0.4; needed only by dssp's configure stage)
# The pre-patched trees are shipped in the repository's deps/ directory and
# in deps.tar.gz. When ALPHAFOLD3_DEPS_DIR is unset (or a dependency
# directory is missing), the original GitHub FetchContent path is used.
set(DEPS_DIR "$ENV{ALPHAFOLD3_DEPS_DIR}" CACHE PATH
    "Directory containing pre-downloaded dependency sources")

# AlphaFold3 only needs the dssp::dssp library, not the mkdssp executable
# (whose libmcfp linkage is broken with newer libmcfp). Default OFF; the
# deps/dssp patch keeps the option available for standalone builds.
set(DSSP_BUILD_MKDSSP OFF CACHE BOOL
    "Build the mkdssp executable (not needed by AlphaFold3)")

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/abseil-cpp")
  FetchContent_Declare(abseil-cpp SOURCE_DIR "${DEPS_DIR}/abseil-cpp")
else()
  FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG d7aaad83b488fd62bd51c81ecf16cd938532cc0a # 20240116.2
    EXCLUDE_FROM_ALL)
endif()

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/pybind11")
  FetchContent_Declare(pybind11 SOURCE_DIR "${DEPS_DIR}/pybind11")
else()
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG 2e0815278cb899b20870a67ca8205996ef47e70f # v2.12.0
    EXCLUDE_FROM_ALL)
endif()

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/pybind11_abseil")
  FetchContent_Declare(pybind11_abseil SOURCE_DIR "${DEPS_DIR}/pybind11_abseil")
else()
  FetchContent_Declare(
    pybind11_abseil
    GIT_REPOSITORY https://github.com/pybind/pybind11_abseil
    GIT_TAG bddf30141f9fec8e577f515313caec45f559d319 # HEAD @ 2024-08-07
    EXCLUDE_FROM_ALL)
endif()

# Note: the FetchContent name is `cifpp` but the local dependency directory
# must be named `libcifpp` (matching the upstream repository name).
if(DEPS_DIR AND EXISTS "${DEPS_DIR}/libcifpp")
  FetchContent_Declare(cifpp SOURCE_DIR "${DEPS_DIR}/libcifpp")
else()
  FetchContent_Declare(
    cifpp
    GIT_REPOSITORY https://github.com/pdb-redo/libcifpp
    GIT_TAG ac98531a2fc8daf21131faa0c3d73766efa46180 # v7.0.3
    # Don't `EXCLUDE_FROM_ALL` as necessary for build_data.
  )
endif()

if(DEPS_DIR AND EXISTS "${DEPS_DIR}/dssp")
  FetchContent_Declare(dssp SOURCE_DIR "${DEPS_DIR}/dssp")
else()
  FetchContent_Declare(
    dssp
    GIT_REPOSITORY https://github.com/PDB-REDO/dssp
    GIT_TAG 57560472b4260dc41f457706bc45fc6ef0bc0f10 # v4.4.7
    EXCLUDE_FROM_ALL)
endif()

FetchContent_MakeAvailable(pybind11 abseil-cpp pybind11_abseil cifpp dssp)

find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)

# libcifpp uses boost::regex; without this link the built cpp module fails to
# import with `undefined symbol: _ZN5boost16re_detail...verify_optionsE...`.
# Needs the system package libboost-regex-dev (>= 1.74).
find_package(Boost 1.74 REQUIRED COMPONENTS regex)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(src/)

file(GLOB_RECURSE cpp_srcs src/alphafold3/*.cc)
list(FILTER cpp_srcs EXCLUDE REGEX ".*\(_test\|_main\|_benchmark\).cc$")

add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

pybind11_add_module(cpp ${cpp_srcs})

target_link_libraries(
  cpp
  PRIVATE absl::check
          absl::flat_hash_map
          absl::node_hash_map
          absl::strings
          absl::status
          absl::statusor
          absl::log
          pybind11_abseil::absl_casters
          Python3::NumPy
          dssp::dssp
          cifpp::cifpp
          Boost::regex)

target_compile_definitions(cpp PRIVATE VERSION_INFO=${PROJECT_VERSION})
install(TARGETS cpp LIBRARY DESTINATION alphafold3)
install(
  FILES LICENSE
        OUTPUT_TERMS_OF_USE.md
        WEIGHTS_PROHIBITED_USE_POLICY.md
        WEIGHTS_TERMS_OF_USE.md
  DESTINATION alphafold3)
