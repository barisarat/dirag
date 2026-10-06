"""The local models (the embedder here, the cross-encoder in rerank.py) and where they run.

They run on an NVIDIA GPU when one is present and working, else on the CPU.
Present means an NVIDIA driver is loaded and onnxruntime has its CUDA provider;
working means a first inference succeeds. A GPU that fails that check is
reported once and the CPU is used instead.

    DIRAG_DEVICE=cpu        never use the GPU
    DIRAG_EMBED_THREADS     CPU threads for onnxruntime; default all cores
"""

import os
import shutil
import subprocess
import sys
import threading
import warnings
from pathlib import Path

from . import config

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
THREADS = int(os.getenv("DIRAG_EMBED_THREADS")) if os.getenv("DIRAG_EMBED_THREADS") else None

_model = None
_lock = threading.Lock()
_gpu = None                       # None until decided, then True or False


def gpu_name():
    """The NVIDIA GPU's name, or None when there is no usable one. Does not load onnxruntime.

    nvidia-smi naming a GPU is the test; without nvidia-smi, the GPU device file is.
    Driver files alone (as in a container without the GPU passed through) do not count.
    """
    if os.getenv("DIRAG_DEVICE", "").lower() == "cpu":
        return None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            names = out.stdout.strip().splitlines()
            return names[0] if out.returncode == 0 and names else None
        except (OSError, subprocess.SubprocessError):
            return None
    return "NVIDIA GPU" if Path("/dev/nvidia0").exists() else None


def use_gpu():
    """Whether the local models should try the GPU."""
    global _gpu
    if _gpu is None:
        _gpu = False
        if gpu_name():
            import onnxruntime
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                # CUDA and cuDNN installed as pip packages load from here.
                try:
                    onnxruntime.preload_dlls()
                except Exception:
                    pass
                _gpu = True
    return _gpu


def build(make, probe):
    """make(**options) on the GPU when it works, else on the CPU. probe(model) runs one inference.

    onnxruntime falls back to the CPU on its own when the CUDA provider cannot
    start, with only a warning; that warning counts as the GPU not working.
    """
    global _gpu
    import onnxruntime
    # dirag reports the device itself; onnxruntime's own provider errors are noise here.
    onnxruntime.set_default_logger_severity(4)
    if use_gpu():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = make(cuda=True, device_ids=[0])
                probe(model)
            failed = [str(w.message) for w in caught if "CUDAExecutionProvider" in str(w.message)]
            if not failed:
                return model
            reason = failed[0]
        except Exception as exc:
            reason = str(exc)
        print(f"dirag: the GPU did not work ({reason.splitlines()[0][:160]}); using the CPU", file=sys.stderr, flush=True)
        _gpu = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = make(threads=THREADS, providers=["CPUExecutionProvider"])
        probe(model)
    return model


def _load():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = build(lambda **o: TextEmbedding(model_name=MODEL, cache_dir=str(config.MODELS), **o),
                           lambda m: list(m.embed(["probe"])))
    return _model


def batch_size():
    """Passages per embedding call: large on the GPU, small on the CPU so a stop is honoured within seconds."""
    _load()
    return 256 if _gpu else 32


def passages(texts):
    """Vectors for a batch of passages, as lists of floats."""
    texts = list(texts)
    return [vector.tolist() for vector in _load().embed(texts)] if texts else []


def query(text):
    """Vector for one query. bge models prefix queries with an instruction; query_embed adds it."""
    return list(_load().query_embed([text]))[0].tolist()


def preflight():
    """Load the embedder before a long run, so a broken setup fails before the first book."""
    try:
        _load()
    except Exception as exc:
        raise SystemExit(f"Embedding is not working:\n  {exc}")
