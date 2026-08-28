#!/usr/bin/env python3
"""AlphaFold 3 Ascend NPU inference service with an OpenAI-compatible API.

Architecture (see AF3_Ascend_NPU_field_deployment_handbook.md):

  xfold (PyTorch AF3) + af3.bin weights -> torch_npu -> FastAPI server

Process layout is critical on NPU: deserializing a featurised ``.pkl`` pulls
in ``alphafold3.cpp``, and ``torch_npu`` + ``alphafold3.cpp`` + pickle in the
same process segfaults. Therefore:

  Phase 1 (no torch_npu imported yet): the built-in test pkl is converted
          to ``.npz`` (plain numpy arrays).
  Phase 2 (torch_npu imported): the model is loaded and serves from npz.
          User-supplied pkls are deserialized in a *clean subprocess*.

The service accepts pre-featurised pkl paths only. Running the full MSA /
template data pipeline is out of scope (use xfold ``run_alphafold.py``
with ``--run_data_pipeline`` for that).

Configuration via environment variables (all optional):

  AF3_SRC           alphafold3 source root holding ``src/``   (data pipeline)
  XFOLD_SRC         xfold source root
  AF3_MODEL_DIR     directory with af3.bin(.zst) + fourier npy files
  AF3_DEVICE        torch device spec, e.g. npu:0 (default npu:0)
  AF3_PRECISION     fp32 (default, faster & better confidence on Ascend)
                    or bf16
  AF3_NUM_RECYCLES  model recycles (default 10, official config)
  AF3_NUM_SAMPLES   diffusion samples (default 5)
  AF3_DIFFUSION_STEPS  diffusion sampler steps (default 200)
  AF3_HOST / AF3_PORT  listen address (default 0.0.0.0:9800)
  LIBCIFPP_DATA_DIR libcifpp resource dir (components.cif)
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AF3_SRC = os.environ.get("AF3_SRC", "/home/alphafold3-3.0.4/src")
XFOLD_SRC = os.environ.get("XFOLD_SRC", "/home/xfold/xfold-main")
MODEL_DIR = os.environ.get("AF3_MODEL_DIR", "/home")
LIBCIFPP_DATA_DIR_DEFAULT = (
    "/home/cpp_deps/libcifpp-ac98531a2fc8daf21131faa0c3d73766efa46180/rsrc"
)
TEST_PKL = os.environ.get(
    "AF3_TEST_PKL",
    str(pathlib.Path(AF3_SRC) / "alphafold3/test_data/featurised_example.pkl"),
)
TEST_NPZ = os.environ.get("AF3_TEST_NPZ", "/tmp/af3_test_batch.npz")

DEVICE_SPEC = os.environ.get("AF3_DEVICE", "npu:0")
PRECISION = os.environ.get("AF3_PRECISION", "fp32").lower()  # fp32 | bf16
NUM_RECYCLES = int(os.environ.get("AF3_NUM_RECYCLES", "10"))
NUM_SAMPLES = int(os.environ.get("AF3_NUM_SAMPLES", "5"))
DIFFUSION_STEPS = int(os.environ.get("AF3_DIFFUSION_STEPS", "200"))

HOST = os.environ.get("AF3_HOST", "0.0.0.0")
PORT = int(os.environ.get("AF3_PORT", "9800"))

MODEL_ID = "alphafold3-3.0.4"

# ---------------------------------------------------------------------------
# Phase 1: load test pkl to npz BEFORE importing torch_npu
# ---------------------------------------------------------------------------


def _batch_to_arrays(batch: dict) -> dict[str, np.ndarray]:
    """Keep numpy arrays (and scalars as 0-d arrays); drop everything else."""
    arrays = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (int, float, bool)):
            arrays[k] = np.array(v)
    return arrays


def load_pkl(pkl_path: str) -> dict[str, np.ndarray]:
    """Deserialize a featurised pkl (must run without torch_npu loaded)."""
    with open(pkl_path, "rb") as f:
        batch = pickle.load(f)
    if isinstance(batch, list):
        batch = batch[0]
    return _batch_to_arrays(batch)


def preload_test_data() -> None:
    if not os.path.exists(TEST_PKL):
        print(f"[preload] test pkl not found, skipping: {TEST_PKL}", flush=True)
        return
    print(f"[preload] converting test pkl to npz: {TEST_PKL}", flush=True)
    arrays = load_pkl(TEST_PKL)
    np.savez(TEST_NPZ, **arrays)
    print(f"[preload] saved {len(arrays)} arrays -> {TEST_NPZ}", flush=True)


# alphafold3.cpp must be importable for pickle deserialization.
sys.path.insert(0, str(pathlib.Path(AF3_SRC)))
os.environ.setdefault("LIBCIFPP_DATA_DIR", LIBCIFPP_DATA_DIR_DEFAULT)
preload_test_data()

# ---------------------------------------------------------------------------
# Phase 2: import torch_npu and build the model
# ---------------------------------------------------------------------------

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401,E402
except ImportError:
    torch_npu = None

sys.path.insert(0, str(pathlib.Path(XFOLD_SRC)))

from xfold.alphafold3 import AlphaFold3  # noqa: E402
from xfold.fastnn import config as fastnn_config  # noqa: E402
from xfold.params import import_jax_weights_  # noqa: E402

# NPU has no triton; force the pure torch implementations. (The adapted
# fastnn already auto-detects this, but be explicit in a long-lived service.)
fastnn_config.layer_norm_implementation = "torch"
fastnn_config.dot_product_attention_implementation = "torch"
fastnn_config.gated_linear_unit_implementation = "torch"

DEVICE = torch.device(DEVICE_SPEC)
MODEL: AlphaFold3 | None = None
_INFER_LOCK = threading.Lock()


def device_sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_model() -> None:
    global MODEL
    print(
        f"[model] building AlphaFold3(num_recycles={NUM_RECYCLES}, "
        f"num_samples={NUM_SAMPLES}, diffusion_steps={DIFFUSION_STEPS})...",
        flush=True,
    )
    model = AlphaFold3(
        num_recycles=NUM_RECYCLES,
        num_samples=NUM_SAMPLES,
        diffusion_steps=DIFFUSION_STEPS,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters: {n_params:,}", flush=True)

    print(f"[model] loading weights from {MODEL_DIR}...", flush=True)
    import_jax_weights_(model, pathlib.Path(MODEL_DIR))

    model.eval()
    model = model.to(device=DEVICE)
    MODEL = model
    print(
        f"[model] ready on {DEVICE}, precision={PRECISION}", flush=True,
    )


def npz_to_torch_batch(npz_path: str) -> dict[str, torch.Tensor]:
    """npz -> device tensors, downcasting dtypes unsupported by the NPU."""
    data = np.load(npz_path, allow_pickle=True)
    batch = {}
    for k in data.files:
        arr = data[k]
        if arr.dtype == np.object_:
            continue
        # aclnnMatmul does not support float64; npz stores no bf16.
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)
        t = torch.from_numpy(arr)
        if t.dtype in (torch.bfloat16, torch.float64):
            t = t.to(torch.float32)
        batch[k] = t.to(device=DEVICE)
    return batch


def load_pkl_via_subprocess(
    pkl_path: str, npz_path: str | None = None
) -> str:
    """Deserialize a user pkl in a clean subprocess (no torch_npu loaded).

    torch_npu + alphafold3.cpp + pickle in one process segfaults, so custom
    pkls must never be deserialized inside the service process.
    """
    npz_path = npz_path or f"/tmp/af3_batch_{uuid.uuid4().hex}.npz"
    script = f"""
import sys, pickle, numpy as np
sys.path.insert(0, {AF3_SRC!r})
with open({pkl_path!r}, "rb") as f:
    batch = pickle.load(f)
if isinstance(batch, list):
    batch = batch[0]
arrays = {{}}
for k, v in batch.items():
    if isinstance(v, np.ndarray):
        arrays[k] = v
    elif isinstance(v, (int, float, bool)):
        arrays[k] = np.array(v)
np.savez({npz_path!r}, **arrays)
"""
    env = dict(os.environ)
    env.setdefault("LIBCIFPP_DATA_DIR", LIBCIFPP_DATA_DIR_DEFAULT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pkl deserialization failed: {result.stderr[-500:]}")
    return npz_path


def run_inference(batch: dict[str, torch.Tensor]) -> dict:
    if MODEL is None:
        raise RuntimeError("model not loaded")

    use_bf16 = PRECISION == "bf16" and DEVICE.type in ("npu", "cuda")
    with _INFER_LOCK, torch.inference_mode():
        if use_bf16:
            with torch.amp.autocast(
                device_type=DEVICE.type, dtype=torch.bfloat16
            ):
                result = MODEL(batch)
        else:
            result = MODEL(batch)
    device_sync(DEVICE)

    # bf16 outputs are downcast for downstream consumers.
    result = {
        k: v.to(torch.float32) if v.dtype == torch.bfloat16 else v
        for k, v in result.items()
        if isinstance(v, torch.Tensor)
    }
    return result


def summarize_result(result: dict) -> dict:
    """Compact JSON-safe summary: shape/dtype/norm per tensor + key metrics."""

    def tensor_summary(t: torch.Tensor) -> dict:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(t.float().norm().item()) if t.numel() else 0.0,
        }

    summary = {k: tensor_summary(v) for k, v in result.items()}

    metrics = {}
    for key in ("contact_probs", "predicted_lddt"):
        t = result.get(key)
        if isinstance(t, torch.Tensor) and t.numel():
            metrics[f"{key}_mean"] = float(t.float().mean().item())
    return {"tensors": summary, "metrics": metrics}


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------


def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="AlphaFold3 NPU Service", version="1.1.0")

    def chat_response(content: dict, finish_reason: str = "stop",
                      status_code: int = 200) -> JSONResponse:
        return JSONResponse(
            {
                "id": f"af3-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content, ensure_ascii=False),
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
            status_code=status_code,
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "device": str(DEVICE),
            "precision": PRECISION,
            "num_recycles": NUM_RECYCLES,
            "num_samples": NUM_SAMPLES,
            "diffusion_steps": DIFFUSION_STEPS,
            "npu_available": (
                torch.npu.is_available() if torch_npu is not None else False
            ),
            "parameters": (
                sum(p.numel() for p in MODEL.parameters())
                if MODEL is not None
                else 0
            ),
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "xfold-npu",
                }
            ],
        }

    def _infer_npz(npz_path: str) -> dict:
        batch = npz_to_torch_batch(npz_path)
        t0 = time.time()
        result = run_inference(batch)
        elapsed = time.time() - t0
        return {
            "status": "success",
            "inference_time_sec": round(elapsed, 2),
            "result_summary": summarize_result(result),
        }

    def _infer_pkl(pkl_path: str) -> dict:
        npz_path = load_pkl_via_subprocess(pkl_path)
        try:
            return _infer_npz(npz_path)
        finally:
            try:
                os.remove(npz_path)
            except OSError:
                pass

    class ChatRequest(BaseModel):
        model: str = MODEL_ID
        messages: list
        temperature: float | None = 1.0
        max_tokens: int | None = 4096
        stream: bool | None = False

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        user_msg = req.messages[-1]["content"] if req.messages else ""
        pkl_path = str(user_msg).strip()
        if not os.path.exists(pkl_path):
            return chat_response(
                {"error": f"File not found: {pkl_path} (expected a "
                          f"pre-featurised pkl path)"},
                finish_reason="error",
            )
        try:
            return chat_response(_infer_pkl(pkl_path))
        except Exception as e:  # noqa: BLE001 - report via API body
            return chat_response(
                {"error": str(e)}, finish_reason="error", status_code=500
            )

    class PredictRequest(BaseModel):
        pkl_path: str

    @app.post("/predict")
    async def predict(req: PredictRequest):
        if not os.path.exists(req.pkl_path):
            raise HTTPException(
                status_code=404, detail=f"File not found: {req.pkl_path}"
            )
        try:
            return _infer_pkl(req.pkl_path)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/test")
    async def predict_test():
        if not os.path.exists(TEST_NPZ):
            raise HTTPException(
                status_code=503,
                detail=f"test npz not preloaded: {TEST_NPZ}",
            )
        try:
            return _infer_npz(TEST_NPZ)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main() -> None:
    import uvicorn

    if PRECISION not in ("fp32", "bf16"):
        raise SystemExit(f"AF3_PRECISION must be fp32 or bf16, got {PRECISION}")

    load_model()
    app = build_app()
    print(f"[server] listening on {HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
