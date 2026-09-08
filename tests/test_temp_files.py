import ast
import logging
import os
from pathlib import Path
import tempfile
import unittest


def load_temp_functions(temp_dir):
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    names = {
        "initialize_temp_directory",
        "create_temp_path",
        "remove_temp_file",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "TEMP_DIR": temp_dir,
        "logging": logging,
        "os": os,
        "tempfile": tempfile,
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), "PhotoPortal.py", "exec"), namespace)
    return namespace


class TempFileTests(unittest.TestCase):
    def test_paths_are_unique_and_created_in_private_directory(self):
        with tempfile.TemporaryDirectory() as workspace:
            temp_dir = os.path.join(workspace, "PhotoPortal", "Temp")
            functions = load_temp_functions(temp_dir)

            first = functions["create_temp_path"]()
            second = functions["create_temp_path"]()

            self.assertNotEqual(first, second)
            self.assertEqual(os.path.dirname(first), temp_dir)
            self.assertTrue(os.path.isfile(first))

    def test_startup_cleanup_only_removes_files_in_private_directory(self):
        with tempfile.TemporaryDirectory() as workspace:
            temp_dir = os.path.join(workspace, "PhotoPortal", "Temp")
            os.makedirs(temp_dir)
            stale_file = os.path.join(temp_dir, "stale.jpg")
            outside_file = os.path.join(workspace, "keep.jpg")
            nested_dir = os.path.join(temp_dir, "keep-directory")
            Path(stale_file).touch()
            Path(outside_file).touch()
            os.mkdir(nested_dir)
            functions = load_temp_functions(temp_dir)

            functions["initialize_temp_directory"]()

            self.assertFalse(os.path.exists(stale_file))
            self.assertTrue(os.path.exists(outside_file))
            self.assertTrue(os.path.isdir(nested_dir))

    def test_removal_refuses_a_path_outside_private_directory(self):
        with tempfile.TemporaryDirectory() as workspace:
            temp_dir = os.path.join(workspace, "PhotoPortal", "Temp")
            os.makedirs(temp_dir)
            outside_file = os.path.join(workspace, "keep.jpg")
            Path(outside_file).touch()
            functions = load_temp_functions(temp_dir)

            functions["remove_temp_file"](outside_file)

            self.assertTrue(os.path.exists(outside_file))


if __name__ == "__main__":
    unittest.main()
