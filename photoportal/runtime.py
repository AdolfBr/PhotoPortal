"""Runtime defaults that must be applied before importing numerical libraries."""
import os

DEFAULT_NUM_THREADS = min(4, os.cpu_count() or 1)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = str(DEFAULT_NUM_THREADS)

APPDATA_LOCAL = os.getenv("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
TEMP_DIR = os.path.join(APPDATA_LOCAL, "PhotoPortal", "Temp")
BG = "#F5F5E6"
