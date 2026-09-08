import ast
import logging
from pathlib import Path
import threading
import unittest

class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4


def load_camera_manager_class():
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CameraManager"
    )
    namespace = {"cv2": FakeCV2, "logging": logging, "threading": threading}
    exec(compile(ast.Module([class_node], type_ignores=[]), "PhotoPortal.py", "exec"), namespace)
    return namespace["CameraManager"]


CameraManager = load_camera_manager_class()


class FakeCapture:
    def __init__(self, camera_id, valid_on_attempt=1, opened=True):
        self.camera_id = camera_id
        self.opened = opened
        self.valid_on_attempt = valid_on_attempt
        self.read_count = 0
        self.width = 0
        self.height = 0
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def set(self, prop, value):
        if prop == FakeCV2.CAP_PROP_FRAME_WIDTH:
            self.width = value
        else:
            self.height = value
        return True

    def get(self, prop):
        return self.width if prop == FakeCV2.CAP_PROP_FRAME_WIDTH else self.height

    def read(self):
        self.read_count += 1
        if self.read_count < self.valid_on_attempt:
            return False, None
        return True, type("Frame", (), {"size": 12})()

    def release(self):
        self.released = True


class CameraManagerTests(unittest.TestCase):
    def test_open_tries_modes_until_a_valid_frame_and_reports_actual_size(self):
        capture = FakeCapture(2, valid_on_attempt=3)
        manager = CameraManager(lambda _camera_id: capture)

        self.assertTrue(manager.open(2))
        self.assertEqual(capture.read_count, 3)
        self.assertEqual(manager.get_resolution(), (1280, 720))

    def test_switch_releases_previous_capture(self):
        captures = []

        def factory(camera_id):
            capture = FakeCapture(camera_id)
            captures.append(capture)
            return capture

        manager = CameraManager(factory)
        manager.open(0)
        manager.switch(1)

        self.assertTrue(captures[0].released)
        self.assertTrue(manager.is_open())

    def test_discovery_releases_every_probe(self):
        probes = []

        def factory(camera_id):
            probe = FakeCapture(camera_id, opened=camera_id == 1)
            probes.append(probe)
            return probe

        manager = CameraManager(factory)
        self.assertEqual(manager.discover(limit=3), [{"id": 1, "name": "Камера 1"}])
        self.assertTrue(all(probe.released for probe in probes))


if __name__ == "__main__":
    unittest.main()
