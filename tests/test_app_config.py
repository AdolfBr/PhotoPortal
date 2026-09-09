import ast
import configparser
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def load_app_config_class():
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AppConfig"
    )
    namespace = {
        "appdata_local": tempfile.gettempdir(),
        "configparser": configparser,
        "DEFAULT_NUM_THREADS": min(4, os.cpu_count() or 1),
        "logging": logging,
        "os": os,
    }
    exec(
        compile(ast.Module([class_node], type_ignores=[]), "PhotoPortal.py", "exec"),
        namespace,
    )
    return namespace["AppConfig"]


AppConfig = load_app_config_class()


class AppConfigTests(unittest.TestCase):
    def test_invalid_fields_do_not_discard_other_valid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "PhotoPortal_config.ini")
            Path(path).write_text(
                "[Settings]\n"
                "SaveDir = /valid/photos\n"
                "CameraIndex = broken\n"
                "IcoDir = /valid/icon.ico\n"
                "MirrorHorizontal = yes\n"
                "NumThreads = 0\n"
                "Sensitivity = 0.75\n"
                "BlurStrength = 7.5\n",
                encoding="utf-8",
            )

            settings = AppConfig(path).load()

            self.assertEqual(settings.SaveDir, "/valid/photos")
            self.assertEqual(settings.CameraIndex, AppConfig.CameraIndex)
            self.assertEqual(settings.IcoDir, "/valid/icon.ico")
            self.assertTrue(settings.MirrorHorizontal)
            self.assertEqual(settings.NumThreads, AppConfig.NumThreads)
            self.assertEqual(settings.Sensitivity, 0.75)
            self.assertEqual(settings.BlurStrength, 7.5)

    def test_malformed_file_and_missing_section_use_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "PhotoPortal_config.ini")
            Path(path).write_text("not an ini", encoding="utf-8")
            self.assertEqual(AppConfig(path).load().CameraIndex, 0)

            Path(path).write_text("[Other]\nvalue=1\n", encoding="utf-8")
            self.assertEqual(AppConfig(path).load().SaveDir, AppConfig.SaveDir)

    def test_save_closes_temporary_file_before_atomic_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "PhotoPortal_config.ini")
            settings = AppConfig(path)
            real_replace = os.replace

            def assert_closed_and_replace(source, destination):
                with open(source, "a", encoding="utf-8"):
                    pass
                real_replace(source, destination)

            with mock.patch.object(os, "replace", side_effect=assert_closed_and_replace):
                self.assertTrue(settings.save())

            self.assertTrue(Path(path).is_file())
            self.assertFalse(Path(path + ".tmp").exists())

    def test_replace_failure_preserves_existing_ini(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "PhotoPortal_config.ini")
            Path(path).write_text("original", encoding="utf-8")

            with mock.patch.object(os, "replace", side_effect=OSError("busy")):
                self.assertFalse(AppConfig(path).save())

            self.assertEqual(Path(path).read_text(encoding="utf-8"), "original")
            self.assertFalse(Path(path + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
