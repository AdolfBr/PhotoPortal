"""Compatibility launcher for the modular PhotoPortal application.

Public legacy helpers remain importable; runtime UI state lives in PhotoPortalApp.
"""
import configparser
from datetime import datetime
import logging
import os
import tempfile
import threading
import tkinter as tk
from tkinter import messagebox

from photoportal.runtime import APPDATA_LOCAL as appdata_local, DEFAULT_NUM_THREADS, TEMP_DIR

try:
    import cv2
except ImportError:
    cv2 = None


def set_window_icon(window):
    """Best-effort application icon setup for a Tk window."""
    if not ICO_DIR:
        logging.warning('Путь к иконке не задан')
        return
    try:
        if not os.path.exists(ICO_DIR):
            logging.warning('Иконка не найдена по пути: %s', ICO_DIR)
            return
        window.iconbitmap(ICO_DIR)
        logging.info('Иконка установлена: %s', ICO_DIR)
    except (OSError, TypeError, ValueError, tk.TclError):
        logging.exception('Не удалось установить иконку окна: %s', ICO_DIR)

class AppConfig:
    """Validated, fault-tolerant application configuration."""
    SaveDir = appdata_local
    CameraIndex = 0
    IcoDir = 'C:\\Program Files\\PhotoPortal\\icon.ico'
    MirrorHorizontal = False
    NumThreads = DEFAULT_NUM_THREADS
    Sensitivity = 0.6
    BlurStrength = 5.5

    def __init__(self, path=None):
        self.path = path or os.path.join(appdata_local, 'PhotoPortal_config.ini')
        for name in self._field_names():
            setattr(self, name, getattr(type(self), name))

    @staticmethod
    def _field_names():
        return ('SaveDir', 'CameraIndex', 'IcoDir', 'MirrorHorizontal', 'NumThreads', 'Sensitivity', 'BlurStrength')

    def _read_value(self, section, name, converter, validator):
        try:
            value = converter(section[name])
            if not validator(value):
                raise ValueError(f'недопустимое значение {value!r}')
            setattr(self, name, value)
        except (KeyError, ValueError, configparser.Error) as error:
            logging.warning('Некорректная настройка %s; используется default %r: %s', name, getattr(self, name), error)

    def load(self):
        """Load every valid field, retaining only that field's default on error."""
        parser = configparser.ConfigParser()
        try:
            with open(self.path, 'r', encoding='utf-8') as config_file:
                parser.read_file(config_file)
            if not parser.has_section('Settings'):
                logging.warning('В конфигурационном файле %s отсутствует [Settings]; используются defaults', self.path)
                return self
        except FileNotFoundError:
            logging.info('Конфигурационный файл %s не найден; используются defaults', self.path)
            return self
        except (configparser.Error, OSError, ValueError) as error:
            logging.warning('Не удалось прочитать конфигурацию %s; используются defaults: %s', self.path, error)
            return self
        section = parser['Settings']
        cpu_limit = os.cpu_count() or 1
        self._read_value(section, 'SaveDir', os.fspath, lambda value: bool(value.strip()))
        self._read_value(section, 'CameraIndex', int, lambda value: value >= 0)
        self._read_value(section, 'IcoDir', os.fspath, lambda value: bool(value.strip()))
        self._read_value(section, 'MirrorHorizontal', lambda value: parser.BOOLEAN_STATES[value.lower()], lambda value: isinstance(value, bool))
        self._read_value(section, 'NumThreads', int, lambda value: 1 <= value <= cpu_limit)
        self._read_value(section, 'Sensitivity', float, lambda value: 0.1 <= value <= 0.9)
        self._read_value(section, 'BlurStrength', float, lambda value: 0 <= value <= 10)
        return self

    def save(self):
        """Atomically replace the INI, leaving an existing file intact on error."""
        parser = configparser.ConfigParser()
        parser['Settings'] = {name: str(getattr(self, name)) for name in self._field_names()}
        temporary_path = self.path + '.tmp'
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(temporary_path, 'w', encoding='utf-8') as config_file:
                parser.write(config_file)
            os.replace(temporary_path, self.path)
            logging.info('Настройки успешно сохранены в %s', self.path)
            return True
        except (configparser.Error, OSError, ValueError):
            logging.exception('Не удалось сохранить настройки в %s', self.path)
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                logging.exception('Не удалось удалить временный файл %s', temporary_path)
            return False

def initialize_temp_directory():
    """Create the private workspace and remove files left by earlier runs."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    for entry in os.scandir(TEMP_DIR):
        try:
            if entry.is_file(follow_symlinks=False):
                os.remove(entry.path)
                logging.info('Удалён устаревший временный файл: %s', entry.path)
        except OSError:
            logging.exception('Не удалось удалить временный файл: %s', entry.path)

def create_temp_path(suffix='.jpg'):
    """Reserve and return a unique path inside PhotoPortal's private workspace."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(prefix='photoportal_', suffix=suffix, dir=TEMP_DIR, delete=False)
    temporary_file.close()
    return temporary_file.name

def remove_temp_file(path):
    """Best-effort removal restricted to the application's private workspace."""
    if not path:
        return
    temp_root = os.path.normcase(os.path.realpath(TEMP_DIR))
    candidate = os.path.normcase(os.path.realpath(path))
    try:
        if os.path.commonpath((temp_root, candidate)) != temp_root:
            logging.warning('Отказ от удаления файла вне временной папки: %s', path)
            return
    except ValueError:
        logging.warning('Отказ от удаления файла на другом диске: %s', path)
        return
    try:
        os.remove(path)
        logging.info('Временный файл удалён: %s', path)
    except FileNotFoundError:
        pass
    except OSError:
        logging.exception('Не удалось удалить временный файл: %s', path)

class PhotoStorage:
    """Atomically save finished JPEG files in a configured directory."""

    def __init__(self, save_dir, error_reporter=None):
        self.save_dir = os.path.abspath(os.fspath(save_dir))
        self.error_reporter = error_reporter or messagebox.showerror

    def save(self, image):
        """Save *image* and return its final path, or ``None`` on an I/O error."""
        staging_path = None
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            if not os.access(self.save_dir, os.W_OK):
                raise PermissionError(f'Нет доступа на запись в каталог: {self.save_dir}')
            timestamp = datetime.now().strftime('photo_%Y%m%d_%H%M%S_%f')[:-3]
            final_path = os.path.join(self.save_dir, f'{timestamp}.jpg')
            temporary_file = tempfile.NamedTemporaryFile(prefix='.photoportal_', suffix='.tmp', dir=self.save_dir, delete=False)
            staging_path = temporary_file.name
            temporary_file.close()
            image.save(staging_path, 'JPEG', quality=95, dpi=(300, 300))
            os.replace(staging_path, final_path)
            staging_path = None
            logging.info('Изображение сохранено: %s', final_path)
            return final_path
        except PermissionError:
            logging.exception('Нет доступа для сохранения фотографии в %s', self.save_dir)
            self.error_reporter('Ошибка сохранения', 'Нет доступа к выбранной папке.')
        except OSError:
            logging.exception('Не удалось сохранить фотографию в %s', self.save_dir)
            self.error_reporter('Ошибка сохранения', 'Не удалось сохранить фотографию.')
        finally:
            if staging_path:
                try:
                    os.remove(staging_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logging.exception('Не удалось удалить временный файл: %s', staging_path)
        return None

class CameraManager:
    """Owns camera discovery and the complete ``VideoCapture`` lifecycle."""
    RESOLUTIONS = ((1920, 1080), (1280, 960), (1280, 720), (640, 480))

    def __init__(self, capture_factory=None):
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture = None
        self._camera_id = None
        self._resolution = None
        self._lock = threading.RLock()

    def discover(self, limit=10):
        """Return camera descriptors without retaining any probe captures."""
        logging.info('Поиск доступных камер')
        cameras = []
        for camera_id in range(limit):
            probe = self._capture_factory(camera_id)
            try:
                if probe.isOpened():
                    cameras.append({'id': camera_id, 'name': f'Камера {camera_id}'})
            finally:
                probe.release()
        logging.info('Доступные камеры: %s', cameras)
        return cameras

    def open(self, camera_id):
        """Open a camera and select the first requested mode yielding a frame."""
        with self._lock:
            self.close()
            capture = self._capture_factory(camera_id)
            if not capture.isOpened():
                capture.release()
                logging.error('Не удалось открыть камеру %s', camera_id)
                return False
            for requested_width, requested_height in self.RESOLUTIONS:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ok, frame = capture.read()
                if ok and frame is not None and getattr(frame, 'size', 0):
                    self._capture = capture
                    self._camera_id = camera_id
                    self._resolution = (actual_width, actual_height)
                    logging.info('Камера %s открыта: запрошено %sx%s, реально выбрано %sx%s', camera_id, requested_width, requested_height, actual_width, actual_height)
                    return True
                logging.warning('Камера %s не вернула кадр при %sx%s (фактически %sx%s)', camera_id, requested_width, requested_height, actual_width, actual_height)
            capture.release()
            logging.error('Камера %s не поддерживает ни один рабочий режим', camera_id)
            return False

    def switch(self, camera_id):
        return self.open(camera_id)

    def read(self):
        with self._lock:
            if self._capture is None or not self._capture.isOpened():
                return (False, None)
            ok, frame = self._capture.read()
            return (ok, frame)

    def is_open(self):
        with self._lock:
            return self._capture is not None and self._capture.isOpened()

    def get_resolution(self):
        with self._lock:
            return self._resolution

    def close(self):
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                logging.info('Камера %s освобождена', self._camera_id)
            self._capture = None
            self._camera_id = None
            self._resolution = None

def calculate_crop_rectangle(image_size, zoom, offset_x, offset_y, canvas_size=(300, 400)):
    """Map the canvas viewport to one bounded, integer 3:4 source rectangle."""
    image_width, image_height = image_size
    canvas_width, canvas_height = canvas_size
    if zoom <= 0:
        raise ValueError('zoom must be positive')
    ratio_unit = int(min(canvas_width / (3 * zoom), canvas_height / (4 * zoom)) + 1e-09)
    if ratio_unit < 1:
        raise ValueError('zoom is too large to produce an integer 3:4 crop')
    crop_width, crop_height = (3 * ratio_unit, 4 * ratio_unit)
    viewport_left = -offset_x / zoom
    viewport_top = -offset_y / zoom
    left = round(viewport_left + (canvas_width / zoom - crop_width) / 2)
    top = round(viewport_top + (canvas_height / zoom - crop_height) / 2)
    left = max(0, min(left, image_width - crop_width))
    top = max(0, min(top, image_height - crop_height))
    return (left, top, left + crop_width, top + crop_height)

# Canonical implementations and controller are provided by the package modules.
from photoportal.image_processor import ImageProcessor
from photoportal.app import PhotoPortalApp

def main():
    log_path = os.path.abspath(os.path.join(appdata_local, "PhotoPortal.log"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=log_path, filemode="a")
    logging.info("Программа запущена")
    PhotoPortalApp().run()

if __name__ == "__main__":
    main()
