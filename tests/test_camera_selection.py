import unittest

from camera_selection import select_camera_id


class SelectCameraIdTests(unittest.TestCase):
    def test_single_camera(self):
        self.assertEqual(select_camera_id([0], 0), 0)

    def test_contiguous_camera_ids(self):
        self.assertEqual(select_camera_id([0, 1], 1), 1)

    def test_non_contiguous_camera_ids(self):
        self.assertEqual(select_camera_id([0, 2], 2), 2)

    def test_missing_saved_id_uses_first_available_id(self):
        self.assertEqual(select_camera_id([0, 2], 1), 0)


if __name__ == "__main__":
    unittest.main()
