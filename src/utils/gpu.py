import ctypes
import importlib.util
import logging
import os
import site
from pathlib import Path

logger = logging.getLogger(__name__)

_PRELOAD_LIBS = ["nvidia/cusolver/lib/libcusolver.so.11"]


def _preload_cuda_libs() -> None:
    for site_dir in site.getsitepackages():
        for rel_path in _PRELOAD_LIBS:
            lib = Path(site_dir) / rel_path
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _preload_triton() -> None:
    if importlib.util.find_spec("triton") is not None:
        importlib.import_module("triton")


def setup_gpu(quiet: bool = True) -> list:
    if quiet:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    _preload_triton()
    _preload_cuda_libs()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        logger.info("TensorFlow utilise %d GPU : %s", len(gpus), gpus)
    else:
        logger.warning("Aucun GPU détecté : TensorFlow tourne sur CPU")
    return gpus
