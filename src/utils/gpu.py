"""Prépare TensorFlow pour utiliser le GPU NVIDIA.

À appeler AVANT tout `import tensorflow` (racine du code : src/) :

    from utils.gpu import setup_gpu
    setup_gpu()
"""

import ctypes
import logging
import os
import site
from pathlib import Path

logger = logging.getLogger(__name__)

# Bug de packaging de TF 2.21 : libcusolver est absent du chemin de recherche
# de libtensorflow_framework, ce qui désactive tout le GPU. On la précharge.
_PRELOAD_LIBS = ["nvidia/cusolver/lib/libcusolver.so.11"]


def _preload_cuda_libs() -> None:
    for site_dir in site.getsitepackages():
        for rel_path in _PRELOAD_LIBS:
            lib = Path(site_dir) / rel_path
            if lib.exists():
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def setup_gpu(quiet: bool = True) -> list:
    """Configure TensorFlow et retourne la liste des GPU visibles."""
    if quiet:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    _preload_cuda_libs()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # Allocation progressive : laisse de la mémoire à PyTorch (YOLOv8)
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus:
        logger.info("TensorFlow utilise %d GPU : %s", len(gpus), gpus)
    else:
        logger.warning("Aucun GPU détecté : TensorFlow tourne sur CPU")
    return gpus
