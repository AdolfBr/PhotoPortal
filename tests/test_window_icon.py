import ast
import logging
import os
from pathlib import Path
import tkinter as tk
import unittest
from unittest import mock


def load_set_window_icon(icon_path):
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_window_icon"
    )
    namespace = {
        "ICO_DIR": icon_path,
        "logging": logging,
        "os": os,
        "tk": tk,
    }
    exec(
        compile(ast.Module([function_node], type_ignores=[]), "PhotoPortal.py", "exec"),
        namespace,
    )
    return namespace["set_window_icon"]


class SetWindowIconTests(unittest.TestCase):
    def test_sets_existing_icon(self):
        window = mock.Mock()

        with mock.patch.object(os.path, "exists", return_value=True):
            load_set_window_icon("icon.ico")(window)

        window.iconbitmap.assert_called_once_with("icon.ico")

    def test_does_not_set_missing_or_unconfigured_icon(self):
        for icon_path in (None, "", "missing.ico"):
            with self.subTest(icon_path=icon_path):
                window = mock.Mock()
                with mock.patch.object(os.path, "exists", return_value=False):
                    load_set_window_icon(icon_path)(window)
                window.iconbitmap.assert_not_called()

    def test_tcl_error_is_logged_and_suppressed(self):
        window = mock.Mock()
        window.iconbitmap.side_effect = tk.TclError("unsupported icon")

        with (
            mock.patch.object(os.path, "exists", return_value=True),
            mock.patch.object(logging, "exception") as log_exception,
        ):
            load_set_window_icon("icon.ico")(window)

        log_exception.assert_called_once()

    def test_filesystem_error_is_logged_and_suppressed(self):
        window = mock.Mock()

        with (
            mock.patch.object(os.path, "exists", side_effect=OSError("unavailable")),
            mock.patch.object(logging, "exception") as log_exception,
        ):
            load_set_window_icon("icon.ico")(window)

        window.iconbitmap.assert_not_called()
        log_exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
