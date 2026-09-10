"""Camera discovery and capture lifecycle."""
import logging
import threading

from . import runtime as _runtime

import cv2


class CameraManager:
    RESOLUTIONS = ((1920, 1080), (1280, 960), (1280, 720), (640, 480))

    def __init__(self, capture_factory=None):
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture = None
        self._camera_id = None
        self._resolution = None
        self._lock = threading.RLock()

    def discover(self, limit=10):
        cameras = []
        for camera_id in range(limit):
            probe = self._capture_factory(camera_id)
            try:
                if probe.isOpened(): cameras.append({"id": camera_id, "name": f"Камера {camera_id}"})
            finally: probe.release()
        return cameras

    def open(self, camera_id):
        with self._lock:
            self.close()
            capture = self._capture_factory(camera_id)
            if not capture.isOpened():
                capture.release(); return False
            for requested_width, requested_height in self.RESOLUTIONS:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                ok, frame = capture.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    self._capture, self._camera_id, self._resolution = capture, camera_id, actual
                    return True
            capture.release(); return False

    def switch(self, camera_id): return self.open(camera_id)
    def read(self):
        with self._lock:
            return (False, None) if self._capture is None or not self._capture.isOpened() else self._capture.read()
    def is_open(self):
        with self._lock: return self._capture is not None and self._capture.isOpened()
    def get_resolution(self):
        with self._lock: return self._resolution
    def close(self):
        with self._lock:
            if self._capture is not None: self._capture.release()
            self._capture = self._camera_id = self._resolution = None
