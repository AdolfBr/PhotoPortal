"""Final and temporary image storage."""
from datetime import datetime
import logging
import os
import tempfile
from tkinter import messagebox
from .runtime import TEMP_DIR


def initialize_temp_directory(temp_dir=TEMP_DIR):
    os.makedirs(temp_dir, exist_ok=True)
    for entry in os.scandir(temp_dir):
        try:
            if entry.is_file(follow_symlinks=False): os.remove(entry.path)
        except OSError: logging.exception("Не удалось удалить временный файл: %s", entry.path)


def create_temp_path(suffix=".jpg", temp_dir=TEMP_DIR):
    os.makedirs(temp_dir, exist_ok=True)
    file = tempfile.NamedTemporaryFile(prefix="photoportal_", suffix=suffix, dir=temp_dir, delete=False); file.close()
    return file.name


def remove_temp_file(path, temp_dir=TEMP_DIR):
    if not path: return
    root, candidate = os.path.normcase(os.path.realpath(temp_dir)), os.path.normcase(os.path.realpath(path))
    try:
        if os.path.commonpath((root, candidate)) != root: return
    except ValueError: return
    try: os.remove(path)
    except FileNotFoundError: pass
    except OSError: logging.exception("Не удалось удалить временный файл: %s", path)


class PhotoStorage:
    def __init__(self, save_dir, error_reporter=None):
        self.save_dir = os.path.abspath(os.fspath(save_dir)); self.error_reporter = error_reporter or messagebox.showerror
    def save(self, image):
        staging_path = None
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            if not os.access(self.save_dir, os.W_OK): raise PermissionError
            stamp = datetime.now().strftime("photo_%Y%m%d_%H%M%S_%f")[:-3]
            final_path = os.path.join(self.save_dir, f"{stamp}.jpg")
            file = tempfile.NamedTemporaryFile(prefix=".photoportal_", suffix=".tmp", dir=self.save_dir, delete=False)
            staging_path = file.name; file.close()
            image.save(staging_path, "JPEG", quality=95, dpi=(300, 300)); os.replace(staging_path, final_path); staging_path = None
            return final_path
        except PermissionError: self.error_reporter("Ошибка сохранения", "Нет доступа к выбранной папке.")
        except OSError: self.error_reporter("Ошибка сохранения", "Не удалось сохранить фотографию.")
        finally:
            if staging_path:
                try: os.remove(staging_path)
                except OSError: pass
        return None
