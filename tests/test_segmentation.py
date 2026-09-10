import types
import unittest
from unittest import mock

from photoportal.segmentation import create_selfie_segmentation


class SegmentationFactoryTests(unittest.TestCase):
    def test_imports_implementation_without_top_level_solutions_attribute(self):
        model = object()
        constructor = mock.Mock(return_value=model)
        implementation = types.SimpleNamespace(SelfieSegmentation=constructor)

        with mock.patch(
            "photoportal.segmentation.import_module", return_value=implementation
        ) as import_module:
            result = create_selfie_segmentation()

        self.assertIs(result, model)
        import_module.assert_called_once_with(
            "mediapipe.python.solutions.selfie_segmentation"
        )
        constructor.assert_called_once_with(model_selection=0)

    def test_reports_how_to_repair_an_incompatible_installation(self):
        with mock.patch(
            "photoportal.segmentation.import_module",
            side_effect=ModuleNotFoundError("solutions"),
        ):
            with self.assertRaisesRegex(RuntimeError, "force-reinstall"):
                create_selfie_segmentation()


if __name__ == "__main__":
    unittest.main()
