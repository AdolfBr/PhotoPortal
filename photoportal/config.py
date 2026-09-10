"""Persistent PhotoPortal configuration."""
import configparser
import logging
import os

from .runtime import APPDATA_LOCAL, DEFAULT_NUM_THREADS


class AppConfig:
    """Validated, fault-tolerant application configuration."""

    SaveDir = APPDATA_LOCAL
    CameraIndex = 0
    IcoDir = r"C:\Program Files\PhotoPortal\icon.ico"
    MirrorHorizontal = False
    NumThreads = DEFAULT_NUM_THREADS
    Sensitivity = 0.6
    BlurStrength = 5.5

    def __init__(self, path=None):
        self.path = path or os.path.join(APPDATA_LOCAL, "PhotoPortal_config.ini")
        for name in self._field_names():
            setattr(self, name, getattr(type(self), name))

    @staticmethod
    def _field_names():
        return ("SaveDir", "CameraIndex", "IcoDir", "MirrorHorizontal", "NumThreads", "Sensitivity", "BlurStrength")

    def _read_value(self, section, name, converter, validator):
        try:
            value = converter(section[name])
            if not validator(value):
                raise ValueError(f"недопустимое значение {value!r}")
            setattr(self, name, value)
        except (KeyError, ValueError, configparser.Error) as error:
            logging.warning("Некорректная настройка %s; используется default %r: %s", name, getattr(self, name), error)

    def load(self):
        parser = configparser.ConfigParser()
        try:
            with open(self.path, "r", encoding="utf-8") as config_file:
                parser.read_file(config_file)
            if not parser.has_section("Settings"):
                logging.warning("В конфигурационном файле %s отсутствует [Settings]; используются defaults", self.path)
                return self
        except FileNotFoundError:
            logging.info("Конфигурационный файл %s не найден; используются defaults", self.path)
            return self
        except (configparser.Error, OSError, ValueError) as error:
            logging.warning("Не удалось прочитать конфигурацию %s; используются defaults: %s", self.path, error)
            return self
        section = parser["Settings"]
        cpu_limit = os.cpu_count() or 1
        self._read_value(section, "SaveDir", os.fspath, lambda value: bool(value.strip()))
        self._read_value(section, "CameraIndex", int, lambda value: value >= 0)
        self._read_value(section, "IcoDir", os.fspath, lambda value: bool(value.strip()))
        self._read_value(section, "MirrorHorizontal", lambda value: parser.BOOLEAN_STATES[value.lower()], lambda value: isinstance(value, bool))
        self._read_value(section, "NumThreads", int, lambda value: 1 <= value <= cpu_limit)
        self._read_value(section, "Sensitivity", float, lambda value: 0.1 <= value <= 0.9)
        self._read_value(section, "BlurStrength", float, lambda value: 0 <= value <= 10)
        return self

    def save(self):
        parser = configparser.ConfigParser()
        parser["Settings"] = {name: str(getattr(self, name)) for name in self._field_names()}
        temporary_path = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(temporary_path, "w", encoding="utf-8") as config_file:
                parser.write(config_file)
            os.replace(temporary_path, self.path)
            logging.info("Настройки успешно сохранены в %s", self.path)
            return True
        except (configparser.Error, OSError, ValueError):
            logging.exception("Не удалось сохранить настройки в %s", self.path)
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            except OSError:
                logging.exception("Не удалось удалить временный файл %s", temporary_path)
            return False
