import ast
from pathlib import Path
import unittest


def load_crop_rectangle_function():
    source = Path(__file__).parents[1].joinpath("PhotoPortal.py").read_text()
    tree = ast.parse(source)
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "calculate_crop_rectangle"
    )
    namespace = {}
    exec(compile(ast.Module([function_node], type_ignores=[]), "PhotoPortal.py", "exec"), namespace)
    return namespace["calculate_crop_rectangle"]


calculate_crop_rectangle = load_crop_rectangle_function()


class CropRectangleTests(unittest.TestCase):
    def assert_three_by_four(self, rectangle):
        left, top, right, bottom = rectangle
        self.assertEqual((right - left) * 4, (bottom - top) * 3)

    def test_portrait_image_uses_the_centered_canvas_window(self):
        rectangle = calculate_crop_rectangle((600, 1000), 0.5, 0, -50)
        self.assertEqual(rectangle, (0, 100, 600, 900))
        self.assert_three_by_four(rectangle)

    def test_landscape_image_uses_the_centered_canvas_window(self):
        rectangle = calculate_crop_rectangle((1000, 600), 2 / 3, -550 / 3, 0)
        self.assertEqual(rectangle, (275, 0, 725, 600))
        self.assert_three_by_four(rectangle)

    def test_zoom_maps_to_a_smaller_source_rectangle(self):
        rectangle = calculate_crop_rectangle((600, 1000), 1, -150, -300)
        self.assertEqual(rectangle, (150, 300, 450, 700))
        self.assert_three_by_four(rectangle)

    def test_drag_to_left_edge(self):
        self.assertEqual(
            calculate_crop_rectangle((600, 1000), 1, 0, -300),
            (0, 300, 300, 700),
        )

    def test_drag_to_right_edge(self):
        self.assertEqual(
            calculate_crop_rectangle((600, 1000), 1, -300, -300),
            (300, 300, 600, 700),
        )

    def test_drag_to_top_edge(self):
        self.assertEqual(
            calculate_crop_rectangle((600, 1000), 1, -150, 0),
            (150, 0, 450, 400),
        )

    def test_drag_to_bottom_edge(self):
        self.assertEqual(
            calculate_crop_rectangle((600, 1000), 1, -150, -600),
            (150, 600, 450, 1000),
        )


if __name__ == "__main__":
    unittest.main()
