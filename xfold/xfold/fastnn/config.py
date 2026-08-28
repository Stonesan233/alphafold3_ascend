try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    triton = None
    HAS_TRITON = False


def _detect_default_implementation() -> str:
    """Pick triton kernels only when triton is installed and CUDA is available.

    NPU environments have no CUDA, so the result there must be "torch".
    """
    if HAS_TRITON:
        try:
            import torch
            if torch.cuda.is_available():
                return "triton"
        except ImportError:
            pass
    return "torch"


# options: ["torch", "triton"]
layer_norm_implementation = _detect_default_implementation()

# options: ["torch", "triton"]
dot_product_attention_implementation = _detect_default_implementation()

# options: ["torch", "triton"]
gated_linear_unit_implementation = _detect_default_implementation()
