import ast
from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def load_photo_storage_class():
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PhotoStorage"
    )
    namespace = {
        "datetime": datetime,
        "logging": logging,
        "messagebox": mock.Mock(),
        "os": os,
        "tempfile": tempfile,
    }
    exec(compile(ast.Module([class_node], type_ignores=[]), "PhotoPortal.py", "exec"), namespace)
    return namespace["PhotoStorage"]


PhotoStorage = load_photo_storage_class()


class FakeImage:
    def __init__(self, payload=b"complete jpeg"):
        self.payload = payload

    def save(self, path, _format, **_options):
        with open(path, "wb") as output:
            output.write(self.payload)


class PhotoStorageTests(unittest.TestCase):
    def test_creates_directory_and_atomically_returns_timestamped_path(self):
        with tempfile.TemporaryDirectory() as workspace:
            destination = os.path.join(workspace, "new", "photos")

            path = PhotoStorage(destination).save(FakeImage())

            self.assertIsNotNone(path)
            self.assertRegex(
                os.path.basename(path),
                r"^photo_\d{8}_\d{6}_\d{3}\.jpg$",
            )
            self.assertEqual(Path(path).read_bytes(), b"complete jpeg")
            self.assertEqual(os.listdir(destination), [os.path.basename(path)])

    def test_reports_write_failure_and_removes_partial_file(self):
        reporter = mock.Mock()
        with tempfile.TemporaryDirectory() as destination:
            image = mock.Mock()
            image.save.side_effect = OSError("disk full")

            path = PhotoStorage(destination, reporter).save(image)

            self.assertIsNone(path)
            reporter.assert_called_once_with(
                "Ошибка сохранения", "Не удалось сохранить фотографию."
            )
            self.assertEqual(os.listdir(destination), [])

    def test_reports_missing_write_access(self):
        reporter = mock.Mock()
        with tempfile.TemporaryDirectory() as destination:
            with mock.patch.object(os, "access", return_value=False):
                path = PhotoStorage(destination, reporter).save(FakeImage())

            self.assertIsNone(path)
            reporter.assert_called_once_with(
                "Ошибка сохранения", "Нет доступа к выбранной папке."
            )


if __name__ == "__main__":
    unittest.main()
