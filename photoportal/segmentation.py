"""MediaPipe selfie-segmentation model construction."""
from importlib import import_module


def create_selfie_segmentation():
    """Create the legacy MediaPipe model used by PhotoPortal.

    Some MediaPipe distributions do not expose ``solutions`` on the package's
    top level even though the implementation is still importable.  Importing
    the implementation module directly keeps PhotoPortal working in both
    layouts.
    """
    try:
        module = import_module("mediapipe.python.solutions.selfie_segmentation")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Установлена несовместимая версия MediaPipe. Выполните "
            "`python -m pip install --force-reinstall -r requirements.txt`."
        ) from error
    return module.SelfieSegmentation(model_selection=0)
