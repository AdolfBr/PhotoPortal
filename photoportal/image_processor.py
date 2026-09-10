"""Photo segmentation and RGB processing."""
import threading

from . import runtime as _runtime

import cv2
import numpy as np
from PIL import Image

_MEDIAPIPE_LOCK = threading.Lock()


class ImageProcessor:
    def __init__(self, segmentation_model, sensitivity=0.6):
        self.segmentation_model = segmentation_model
        self.sensitivity = float(sensitivity)
        self._source_rgb = None
        self._segmentation_mask = None

    @staticmethod
    def _as_rgb_array(image):
        array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if array.ndim != 3 or array.shape[2] < 3: raise ValueError("Ожидается RGB-изображение")
        return np.clip(array[:, :, :3], 0, 255).astype(np.uint8)

    def segment(self, image):
        rgb = self._as_rgb_array(image)
        with _MEDIAPIPE_LOCK: result = self.segmentation_model.process(rgb)
        mask = np.asarray(result.segmentation_mask, dtype=np.float32)
        if mask.shape != rgb.shape[:2]: mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._source_rgb, self._segmentation_mask = rgb.copy(), np.clip(mask, 0.0, 1.0)
        return self._segmentation_mask.copy()

    def _soft_mask(self, softness):
        if self._segmentation_mask is None: raise ValueError("Сначала необходимо выполнить сегментацию")
        softness = max(0.0, float(softness)); mask = self._segmentation_mask
        if softness:
            kernel = max(3, int(softness * 4) | 1)
            mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=softness * 2, sigmaY=softness * 2)
        return np.clip(mask, 0.0, 1.0)

    def adjust_brightness(self, brightness=0, softness=5.5, image=None):
        if image is not None:
            rgb = self._as_rgb_array(image)
            if self._segmentation_mask is None or self._source_rgb is None or rgb.shape != self._source_rgb.shape: self.segment(rgb)
            else: self._source_rgb = rgb.copy()
        if self._source_rgb is None: raise ValueError("Изображение не задано")
        foreground = np.clip(self._source_rgb.astype(np.float32) + float(brightness), 0, 255)
        mask = self._soft_mask(softness)[..., None]
        composed = foreground * mask + 255.0 * (1.0 - mask)
        return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8), "RGB")

    def remove_background(self, image=None, softness=5.5): return self.adjust_brightness(0, softness, image)

    def resize(self, image, min_width=600, min_height=800):
        rgb = Image.fromarray(self._as_rgb_array(image), "RGB"); width, height = rgb.size
        if width >= min_width and height >= min_height: return rgb
        ratio = max(min_width / width, min_height / height)
        return rgb.resize((int(width * ratio), int(height * ratio)), Image.Resampling.BICUBIC).convert("RGB")

    def prepare_final(self, image, brightness=0, softness=5.5, min_width=600, min_height=800):
        self.segment(image)
        return self.resize(self.adjust_brightness(brightness, softness), min_width, min_height)
