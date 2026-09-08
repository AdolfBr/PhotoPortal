import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from camera_selection import select_camera_id

try:
    import sys
    import random
    import time
    import logging
    import configparser
    import cv2
    import multiprocessing
    import numpy as np
    import mediapipe as mp
    from PIL import Image as PILImage, ImageTk, ImageDraw, ImageFont
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta
except ImportError as e:
    messagebox.showerror("Ошибка импорта", f"Не удалось импортировать библиотеку: {e}")
    raise


mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

executor = ThreadPoolExecutor(max_workers=1)
segmentation_executor = ThreadPoolExecutor(max_workers=1)

appdata_local = os.getenv('LOCALAPPDATA')
if not appdata_local:
    appdata_local = os.path.expanduser('~\\AppData\\Local')

log_filename = 'PhotoPortal.log'
Log_path = os.path.abspath(os.path.join(appdata_local, log_filename))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=Log_path,
    filemode='a'
)
logging.info('Программа запущена')

root = None
webcam_label = None
webcam_after_id = None
app_closing = False
last_webcam_read_error_log_time = None
WEBCAM_READ_ERROR_LOG_INTERVAL = 5.0
PREVIEW_INTERVAL_MS = 36
SEGMENTATION_INTERVAL = 0.1
segmentation_future = None
last_segmentation_submit = 0.0
latest_segmentation_mask = None
segmentation_generation = 0
segmentation_lock = threading.Lock()
mediapipe_lock = threading.Lock()
BG = "#F5F5E6"
SAVE_DIR = appdata_local
ICO_DIR = r"C:\Program Files\PhotoPortal\icon.ico"
MIRROR_HORIZONTAL = False
NUM_THREADS = 4
MAX_THREADS = 4
SENSITIVITY_THRESHOLD = 0.6
BLUR_STRENGTH = 5.5


class CameraManager:
    """Owns camera discovery and the complete ``VideoCapture`` lifecycle."""

    RESOLUTIONS = ((1920, 1080), (1280, 960), (1280, 720), (640, 480))

    def __init__(self, capture_factory=None):
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._capture = None
        self._camera_id = None
        self._resolution = None
        self._lock = threading.RLock()

    def discover(self, limit=10):
        """Return camera descriptors without retaining any probe captures."""
        logging.info("Поиск доступных камер")
        cameras = []
        for camera_id in range(limit):
            probe = self._capture_factory(camera_id)
            try:
                if probe.isOpened():
                    cameras.append({"id": camera_id, "name": f"Камера {camera_id}"})
            finally:
                probe.release()
        logging.info("Доступные камеры: %s", cameras)
        return cameras

    def open(self, camera_id):
        """Open a camera and select the first requested mode yielding a frame."""
        with self._lock:
            self.close()
            capture = self._capture_factory(camera_id)
            if not capture.isOpened():
                capture.release()
                logging.error("Не удалось открыть камеру %s", camera_id)
                return False

            for requested_width, requested_height in self.RESOLUTIONS:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ok, frame = capture.read()
                if ok and frame is not None and getattr(frame, "size", 0):
                    self._capture = capture
                    self._camera_id = camera_id
                    self._resolution = (actual_width, actual_height)
                    logging.info(
                        "Камера %s открыта: запрошено %sx%s, реально выбрано %sx%s",
                        camera_id, requested_width, requested_height,
                        actual_width, actual_height,
                    )
                    return True
                logging.warning(
                    "Камера %s не вернула кадр при %sx%s (фактически %sx%s)",
                    camera_id, requested_width, requested_height,
                    actual_width, actual_height,
                )

            capture.release()
            logging.error("Камера %s не поддерживает ни один рабочий режим", camera_id)
            return False

    def switch(self, camera_id):
        return self.open(camera_id)

    def read(self):
        with self._lock:
            if self._capture is None or not self._capture.isOpened():
                return False, None
            ok, frame = self._capture.read()
            return ok, frame

    def is_open(self):
        with self._lock:
            return self._capture is not None and self._capture.isOpened()

    def get_resolution(self):
        with self._lock:
            return self._resolution

    def close(self):
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                logging.info("Камера %s освобождена", self._camera_id)
            self._capture = None
            self._camera_id = None
            self._resolution = None


class ImageProcessor:
    """RGB-only image processing with a reusable MediaPipe segmentation mask."""

    def __init__(self, segmentation_model=None, sensitivity=SENSITIVITY_THRESHOLD):
        self.segmentation_model = segmentation_model or selfie_segmentation
        self.sensitivity = float(sensitivity)
        self._source_rgb = None
        self._segmentation_mask = None

    @staticmethod
    def _as_rgb_array(image):
        if isinstance(image, PILImage.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] < 3:
                raise ValueError("Ожидается RGB-изображение")
            array = array[:, :, :3]
        return np.clip(array, 0, 255).astype(np.uint8)

    def segment(self, image):
        """Run MediaPipe once and cache its continuous float mask."""
        rgb = self._as_rgb_array(image)
        with mediapipe_lock:
            result = self.segmentation_model.process(rgb)
        mask = np.asarray(result.segmentation_mask, dtype=np.float32)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        self._source_rgb = rgb.copy()
        self._segmentation_mask = np.clip(mask, 0.0, 1.0)
        return self._segmentation_mask.copy()

    def _soft_mask(self, softness):
        if self._segmentation_mask is None:
            raise ValueError("Сначала необходимо выполнить сегментацию")
        softness = max(0.0, float(softness))
        mask = self._segmentation_mask
        if softness:
            kernel_size = max(3, int(softness * 4) | 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size),
                                    sigmaX=softness * 2,
                                    sigmaY=softness * 2)
        # Deliberately keep the alpha continuous: no final binary threshold.
        return np.clip(mask, 0.0, 1.0)

    def remove_background(self, image=None, softness=BLUR_STRENGTH):
        """Composite the foreground over white and return an uint8 RGB image."""
        if image is not None:
            rgb = self._as_rgb_array(image)
            if self._segmentation_mask is None or rgb.shape != self._source_rgb.shape:
                self.segment(rgb)
            else:
                self._source_rgb = rgb.copy()
        elif self._source_rgb is None:
            raise ValueError("Изображение не задано")

        mask = self._soft_mask(softness)[..., None]
        foreground = self._source_rgb.astype(np.float32)
        white_background = np.full_like(foreground, 255.0)
        composed = foreground * mask + white_background * (1.0 - mask)
        return PILImage.fromarray(np.clip(composed, 0, 255).astype(np.uint8), "RGB")

    def adjust_brightness(self, brightness=0, softness=BLUR_STRENGTH,
                          image=None):
        """Adjust cached source RGB and composite with the cached soft mask."""
        if image is not None and self._segmentation_mask is None:
            self.segment(image)
        if self._source_rgb is None:
            raise ValueError("Изображение не задано")
        foreground = np.clip(
            self._source_rgb.astype(np.float32) + float(brightness), 0, 255
        )
        mask = self._soft_mask(softness)[..., None]
        white_background = np.full_like(foreground, 255.0)
        composed = foreground * mask + white_background * (1.0 - mask)
        return PILImage.fromarray(np.clip(composed, 0, 255).astype(np.uint8), "RGB")

    def resize(self, image, min_width=600, min_height=800):
        """Enlarge an RGB image just enough to meet the requested dimensions."""
        rgb_image = PILImage.fromarray(self._as_rgb_array(image), "RGB")
        width, height = rgb_image.size
        if width >= min_width and height >= min_height:
            return rgb_image
        ratio = max(min_width / width, min_height / height)
        size = (int(width * ratio), int(height * ratio))
        return rgb_image.resize(size, PILImage.Resampling.BICUBIC).convert("RGB")

    def prepare_final(self, image, brightness=0, softness=BLUR_STRENGTH,
                      min_width=600, min_height=800):
        """Segment once, composite, adjust brightness and produce final RGB."""
        self.segment(image)
        composed = self.adjust_brightness(brightness, softness)
        return self.resize(composed, min_width, min_height)


camera_manager = CameraManager()






class Tooltip:
    """Класс для всплывающих подсказок с задержкой"""
    def __init__(self, widget, text, delay=2000):  # Задержка в миллисекундах (2 секунды)
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip = None
        self.after_id = None
        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def schedule_tooltip(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self):
        if not self.after_id:  # Если событие отменено, не показываем
            return
        x, y, _, _ = self.widget.bbox("insert") if "insert" in dir(self.widget) else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        self.tooltip.attributes('-topmost', True)  # Подсказка поверх всех окон
        label = tk.Label(self.tooltip, text=self.text, bg="#FFFFDD", fg="#623B2A",
                         font=("Arial", 10), borderwidth=1, relief="solid", justify="left")
        label.pack()
        self.after_id = None

    def hide_tooltip(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def load_settings():
    """Загрузка настроек из конфигурационного файла или создание нового"""
    global SAVE_DIR, ICO_DIR, MIRROR_HORIZONTAL, NUM_THREADS, MAX_THREADS
    logging.info("Начало загрузки настроек")
    config = configparser.ConfigParser()
    config_path = os.path.join(appdata_local, "PhotoPortal_config.ini")

    MAX_THREADS = multiprocessing.cpu_count()
    logging.info(f"Обнаружено ядер: {MAX_THREADS}")

    if not os.path.exists(config_path):
        logging.info("Конфигурационный файл не найден, создание нового")
        config["Settings"] = {
            "SaveDir": SAVE_DIR,
            "IcoDir": ICO_DIR,
            "CameraIndex": "0",
            "MirrorHorizontal": "False",
            "NumThreads": str(MAX_THREADS)
        }
        with open(config_path, "w") as configfile:
            config.write(configfile)
        logging.info("Создан новый конфигурационный файл")

    config.read(config_path)
    SAVE_DIR = config.get("Settings", "SaveDir", fallback=appdata_local)
    camera_index = config.getint("Settings", "CameraIndex", fallback=0)
    cameras = get_available_cameras()
    selected_camera_id = select_camera_id(
        [camera["id"] for camera in cameras], camera_index
    )
    if selected_camera_id is not None:
        camera_index = selected_camera_id
    ICO_DIR = config.get("Settings", "IcoDir", fallback=r"C:\Program Files\PhotoPortal\icon.ico")
    MIRROR_HORIZONTAL = config.getboolean("Settings", "MirrorHorizontal", fallback=False)
    NUM_THREADS = config.getint("Settings", "NumThreads", fallback=MAX_THREADS)

    NUM_THREADS = min(NUM_THREADS, MAX_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
    cv2.setNumThreads(NUM_THREADS)

    logging.info(
        f"Настройки загружены: SaveDir={SAVE_DIR}, CameraIndex={camera_index}, IcoDir={ICO_DIR}, MirrorHorizontal={MIRROR_HORIZONTAL}, NumThreads={NUM_THREADS}")
    return SAVE_DIR, camera_index, ICO_DIR, MIRROR_HORIZONTAL


def save_settings(save_dir, camera_index, ICO_DIR, mirror_horizontal, num_threads):
    """Сохранение настроек в конфигурационный файл"""
    logging.info(
        f"Сохранение настроек: SaveDir={save_dir}, CameraIndex={camera_index}, IcoDir={ICO_DIR}, MirrorHorizontal={mirror_horizontal}, NumThreads={num_threads}")
    config = configparser.ConfigParser()
    config["Settings"] = {
        "SaveDir": save_dir,
        "CameraIndex": str(camera_index),
        "IcoDir": ICO_DIR,
        "MirrorHorizontal": str(mirror_horizontal),
        "NumThreads": str(num_threads)
    }
    config_path = os.path.join(appdata_local, "PhotoPortal_config.ini")

    with open(config_path, "w") as configfile:
        config.write(configfile)
    logging.info("Настройки успешно сохранены")


def open_settings():
    """Открытие окна настроек для изменения параметров"""
    logging.info("Открытие окна настроек")
    settings_window = tk.Toplevel()
    settings_window.title("Настройки")
    settings_window.iconbitmap(ICO_DIR)
    settings_window.geometry("400x600")
    settings_window.configure(bg=BG)
    settings_window.resizable(False, False)
    settings_window.grab_set()
    settings_window.attributes('-topmost', True)  # Дочернее окно поверх главного

    def select_save_dir():
        global SAVE_DIR
        new_dir = filedialog.askdirectory()
        if new_dir:
            SAVE_DIR = new_dir
            save_dir_value.config(text=SAVE_DIR)
            logging.info(f"Выбрана новая папка сохранения: {SAVE_DIR}")



    main_frame = tk.Frame(settings_window, bg=BG)
    main_frame.pack(pady=20, padx=20, fill="both", expand=True)

    save_dir_frame = tk.Frame(main_frame, bg=BG, bd=1, relief="solid")
    save_dir_frame.pack(fill="x", pady=5)
    tk.Label(save_dir_frame, text="Папка сохранения", fg="#623B2A", bg=BG, font=("Arial", 12, "bold")).pack(anchor="w",
                                                                                                            padx=5,
                                                                                                            pady=2)
    save_dir_value = tk.Label(save_dir_frame, text=SAVE_DIR, fg="#623B2A", bg=BG, font=("Arial", 12))
    save_dir_value.pack(anchor="w", padx=5)
    save_dir_button = tk.Button(save_dir_frame, text="Выбрать", command=select_save_dir,
                                bg="#E04E39", fg=BG, font=("Arial", 12, "bold"), padx=8, pady=4,
                                relief="flat", activebackground="#623B2A")
    save_dir_button.pack(anchor="e", padx=5, pady=2)
    Tooltip(save_dir_button, "Выберите папку, куда будут сохраняться готовые фотографии.")

    camera_frame = tk.Frame(main_frame, bg=BG, bd=1, relief="solid")
    camera_frame.pack(fill="x", pady=5)
    tk.Label(camera_frame, text="Камера", fg="#623B2A", bg=BG, font=("Arial", 12, "bold")).pack(anchor="w", padx=5,
                                                                                                pady=2)
    cameras = get_available_cameras()
    camera_names = [camera["name"] for camera in cameras]
    camera_combobox = ttk.Combobox(camera_frame, values=camera_names, width=20)
    camera_combobox.pack(anchor="w", padx=5, pady=2)
    if cameras:
        selected_camera_id = select_camera_id(
            [camera["id"] for camera in cameras], camera_index
        )
        selected_position = next(
            position
            for position, camera in enumerate(cameras)
            if camera["id"] == selected_camera_id
        )
        camera_combobox.current(selected_position)
    else:
        logging.warning("Не найдено доступных камер")
        messagebox.showwarning("Предупреждение", "Не найдено доступных камер.")
    Tooltip(camera_combobox, "Выберите камеру для съемки. Обычно Камера 0 — встроенная веб-камера.")

    mirror_frame = tk.Frame(main_frame, bg=BG, bd=1, relief="solid")
    mirror_frame.pack(fill="x", pady=5)
    tk.Label(mirror_frame, text="Зеркальное отображение", fg="#623B2A", bg=BG, font=("Arial", 12, "bold")).pack(
        anchor="w", padx=5, pady=2)
    mirror_var = tk.BooleanVar(value=MIRROR_HORIZONTAL)
    mirror_check = tk.Checkbutton(mirror_frame, text="Включить", variable=mirror_var,
                                  fg="#623B2A", bg=BG, font=("Arial", 12), activebackground="#FFFFFF")
    mirror_check.pack(anchor="w", padx=5, pady=2)
    Tooltip(mirror_check, "Включите, если изображение с камеры нужно отразить по горизонтали.")



    threads_frame = tk.Frame(main_frame, bg=BG, bd=1, relief="solid")
    tk.Label(threads_frame, text="Количество потоков", fg="#623B2A", bg=BG, font=("Arial", 12, "bold")).pack(anchor="w",
                                                                                                             padx=5,
                                                                                                             pady=2)
    threads_options = [1]
    i = 2
    while i <= MAX_THREADS:
        threads_options.append(i)
        i += 2
    threads_combobox = ttk.Combobox(threads_frame, values=threads_options, width=20)
    threads_combobox.pack(anchor="w", padx=5, pady=2)
    threads_combobox.set(NUM_THREADS)
    Tooltip(threads_combobox, "Число потоков для обработки. Больше — быстрее, но зависит от вашего устройства.")

    apply_button = tk.Button(main_frame, text="Применить", command=lambda: apply_settings(),
                             bg="#623B2A", fg="#FFFFFF", font=("Arial", 14, "bold"), padx=10, pady=6,
                             relief="flat", activebackground="#E04E39")
    apply_button.pack(pady=20)
    Tooltip(apply_button, "Сохраните настройки и закройте окно.")

    advanced_label = tk.Label(settings_window, text="Расширенные параметры", fg="#623B2A", bg=BG,
                              font=("Arial", 10, "underline"), cursor="hand2")
    advanced_label.place(relx=1.0, rely=1.0, x=-10, y=-10, anchor="se")
    Tooltip(advanced_label, "Нажмите, чтобы показать или скрыть дополнительные настройки.")

    def toggle_advanced():
        if threads_frame.winfo_ismapped():
            threads_frame.pack_forget()
            advanced_label.config(text="Расширенные параметры")
        else:
            threads_frame.pack(fill="x", pady=5, before=apply_button)
            advanced_label.config(text="Скрыть расширенные параметры")

    advanced_label.bind("<Button-1>", lambda e: toggle_advanced())

    def apply_settings():
        global SAVE_DIR, camera_index, ICO_DIR, MIRROR_HORIZONTAL, NUM_THREADS
        selected_position = camera_combobox.current()
        selected_camera_id = (
            cameras[selected_position]["id"]
            if cameras and selected_position >= 0
            else None
        )
        MIRROR_HORIZONTAL = mirror_var.get()
        NUM_THREADS = int(threads_combobox.get())
        os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
        cv2.setNumThreads(NUM_THREADS)
        if selected_camera_id is not None:
            camera_index = selected_camera_id
        save_settings(
            SAVE_DIR, camera_index, ICO_DIR, MIRROR_HORIZONTAL, NUM_THREADS
        )
        messagebox.showinfo("Успех", "Настройки сохранены!")
        settings_window.destroy()
        if selected_camera_id is not None:
            update_camera(selected_camera_id)
        else:
            logging.warning("Не выбрана камера или камеры отсутствуют")
            messagebox.showwarning("Предупреждение", "Камера не выбрана или отсутствует.")


def get_available_cameras():
    """Получение доступных камер"""
    return camera_manager.discover()


SAVE_DIR, camera_index, ICO_DIR, MIRROR_HORIZONTAL = load_settings()


def show_webcam():
    """Отображение изображения с камеры"""
    global webcam_label, MIRROR_HORIZONTAL
    global webcam_after_id, last_webcam_read_error_log_time
    global segmentation_future, last_segmentation_submit
    global segmentation_generation

    # Запланированный callback уже начал выполняться, поэтому его ID больше
    # нельзя отменить через after_cancel().
    webcam_after_id = None
    if app_closing:
        return
    if not camera_manager.is_open():
        logging.warning("Камера не инициализирована")
        return
    ret, frame = camera_manager.read()
    if not ret or frame is None:
        current_time = time.monotonic()
        if (last_webcam_read_error_log_time is None or
                current_time - last_webcam_read_error_log_time >= WEBCAM_READ_ERROR_LOG_INTERVAL):
            logging.error("Не удалось получить кадр с камеры")
            last_webcam_read_error_log_time = current_time
        schedule_webcam_update()
        return

    if MIRROR_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(frame_gray)
    text_lines = None
    if mean_brightness < 70:
        text_lines = ["Слишком темно!", "Увеличьте освещение."]
    elif mean_brightness > 200:
        text_lines = ["Слишком ярко за спиной!", "Поробуйте сменить ракурс"]

    height, width = frame.shape[:2]
    target_width = int(height * 3 / 4)
    if width > target_width:
        left = (width - target_width) // 2
        frame = frame[:, left:left + target_width]
    now = time.monotonic()
    if (segmentation_future is None or segmentation_future.done()) and \
            now - last_segmentation_submit >= SEGMENTATION_INTERVAL:
        last_segmentation_submit = now
        segmentation_future = segmentation_executor.submit(
            analyse_preview_frame, frame.copy(), SENSITIVITY_THRESHOLD
        )
        generation = segmentation_generation
        segmentation_future.add_done_callback(
            lambda future, generation=generation:
            store_segmentation_result(future, generation)
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with segmentation_lock:
        mask_smoothed = (None if latest_segmentation_mask is None
                         else latest_segmentation_mask.copy())
    if mask_smoothed is not None:
        mask_smoothed = cv2.resize(
            mask_smoothed, (frame_rgb.shape[1], frame_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        frame_rgb[mask_smoothed == 0] = [255, 255, 255]

    frame_rgb = cv2.resize(frame_rgb, (300, 400), interpolation=cv2.INTER_LANCZOS4)
    frame_pil = PILImage.fromarray(frame_rgb)

    if text_lines:
        draw = ImageDraw.Draw(frame_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for
                        line in text_lines]
        line_widths = [draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for
                       line in text_lines]
        total_height = sum(line_heights)
        frame_width, frame_height = 300, 400
        start_y = (frame_height - total_height) // 2
        current_y = start_y
        for line, line_width, line_height in zip(text_lines, line_widths, line_heights):
            text_x = (frame_width - line_width) // 2
            draw.text((text_x, current_y), line, fill=(255, 0, 0), font=font)
            current_y += line_height

    frame_tk = ImageTk.PhotoImage(frame_pil)
    webcam_label.config(image=frame_tk)
    webcam_label.image = frame_tk
    schedule_webcam_update()


def schedule_webcam_update():
    """Планирует единственное следующее обновление изображения с камеры."""
    global webcam_after_id
    if not app_closing and webcam_label is not None:
        webcam_after_id = webcam_label.after(PREVIEW_INTERVAL_MS, show_webcam)


def analyse_preview_frame(frame, sensitivity):
    """Run preview segmentation in the sole MediaPipe preview worker."""
    height, width = frame.shape[:2]
    scale = min(240 / width, 320 / height)
    scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    scaled = cv2.resize(frame, scaled_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    with mediapipe_lock:
        results = selfie_segmentation.process(rgb)
    mask = (results.segmentation_mask > sensitivity).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.5, sigmaY=1.5)
    return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]


def store_segmentation_result(future, generation):
    """Replace, rather than enqueue, the latest worker result."""
    global latest_segmentation_mask
    try:
        mask = future.result()
    except Exception:
        logging.exception("Ошибка сегментации preview")
        return
    with segmentation_lock:
        if generation == segmentation_generation:
            latest_segmentation_mask = mask


def update_camera(index):
    """Подключение к камере"""
    global webcam_after_id, latest_segmentation_mask, segmentation_generation
    logging.info(f"Переключение на камеру {index}")
    if webcam_after_id is not None:
        try:
            webcam_label.after_cancel(webcam_after_id)
        except tk.TclError:
            logging.debug("Callback камеры уже был удалён")
        finally:
            webcam_after_id = None

    if app_closing:
        camera_manager.close()
        return
    with segmentation_lock:
        segmentation_generation += 1
        latest_segmentation_mask = None
    if not camera_manager.switch(index):
        messagebox.showerror("Ошибка", "Не удалось открыть камеру")
    else:
        show_webcam()


def load_image():
    """Открытие окна настроек для изменения параметров"""
    logging.info("Открытие диалога для загрузки изображения")
    file_path = filedialog.askopenfilename(filetypes=[("Изображения", "*.jpg;*.jpeg;*.png")])
    if file_path:
        logging.info(f"Выбрано изображение: {file_path}")
        cropped_path = crop_interactively(file_path)
        if cropped_path:
            try:
                result = process_image(cropped_path, from_webcam=False)
                if result == "retry":
                    load_image()
            finally:
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)
                    logging.info(f"Временный файл удален: {cropped_path}")


def process_image(image_path, from_webcam=False):
    """обработка изображения с камеры"""
    global SENSITIVITY_THRESHOLD
    try:
        logging.info(f"Начало обработки изображения: {image_path}")
        input_image = PILImage.open(image_path).convert("RGB")
        processor = ImageProcessor(sensitivity=SENSITIVITY_THRESHOLD)
        processor.segment(input_image)
        return adjust_brightness(processor, from_webcam=from_webcam)
    except Exception as e:
        logging.error(f"Ошибка обработки изображения: {e}")
        messagebox.showerror("Ошибка", f"Ошибка обработки: {e}")
        return None


def adjust_brightness(processor, from_webcam=False):
    """Настройка яркости результата"""
    global SENSITIVITY_THRESHOLD, BLUR_STRENGTH
    logging.info("Открытие окна корректировки яркости и сглаживания")
    brightness_window = tk.Toplevel()
    brightness_window.title("Настройка яркости и сглаживания")
    brightness_window.iconbitmap(ICO_DIR)
    brightness_window.configure(bg=BG)
    brightness_window.grab_set()
    brightness_window.attributes('-topmost', True)  # Дочернее окно поверх главного

    instruction_label = tk.Label(brightness_window,
                                 text="Настройте параметры изображения с помощью ползунков.",
                                 font=("Arial", 10), fg="#623B2A", bg=BG)
    instruction_label.pack(pady=10)

    image_frame = tk.Frame(brightness_window, bd=1, bg="#C39367")
    image_frame.pack(pady=10)
    initial_image = processor.adjust_brightness(0, BLUR_STRENGTH)
    thumbnail = initial_image.copy()
    thumbnail.thumbnail((300, 400))
    img_tk = ImageTk.PhotoImage(thumbnail)
    label = tk.Label(image_frame, image=img_tk, bg="#FFFFFF")
    label.image = img_tk
    label.pack()
    image_frame.config(width=thumbnail.width + 10, height=thumbnail.height + 10)
    image_frame.pack_propagate(False)

    brightness_value = tk.IntVar(value=0)
    brightness_slider = tk.Scale(brightness_window, from_=-50, to=50, orient=tk.HORIZONTAL, label="Яркость фото",
                                 variable=brightness_value, length=300, fg="#623B2A", bg=BG,
                                 troughcolor="#C39367", activebackground="#E04E39",
                                 command=lambda val: update_preview())
    brightness_slider.pack(pady=10)
    Tooltip(brightness_slider, "Сделайте фото светлее или темнее. Сдвиньте влево, чтобы затемнить, вправо — чтобы осветлить.")

    blur_value = tk.DoubleVar(value=BLUR_STRENGTH)
    blur_slider = tk.Scale(brightness_window, from_=0, to=10, resolution=0.5, orient=tk.HORIZONTAL,
                           label="Мягкость краёв",
                           variable=blur_value, length=300, fg="#623B2A", bg=BG,
                           troughcolor="#C39367", activebackground="#E04E39",
                           command=lambda val: update_preview())
    blur_slider.pack(pady=10)
    Tooltip(blur_slider,
            "Настройте, насколько плавно края объекта переходят в фон. Сдвиньте вправо для более мягкого перехода.")

    adjusted_image = [initial_image]

    def update_preview(*args):
        brightness = brightness_value.get()
        blur_strength = blur_value.get()
        adjusted_image[0] = processor.adjust_brightness(brightness, blur_strength)

        thumbnail = adjusted_image[0].copy()
        thumbnail.thumbnail((300, 400))
        new_img_tk = ImageTk.PhotoImage(thumbnail)
        label.config(image=new_img_tk)
        label.image = new_img_tk

    result = [None]

    def save():
        final_image = processor.resize(adjusted_image[0], 600, 800)
        save_image(final_image, brightness_window)
        result[0] = "saved"

    def retry():
        brightness_window.destroy()
        result[0] = "retry"
        logging.info("Пользователь выбрал переснять")

    save_button = tk.Button(brightness_window, text="Сохранить", command=save, bg="#623B2A", fg="#FFFFFF",
                            font=("Arial", 14, "bold"), padx=10, pady=5, relief="flat", activebackground="#E04E39")
    save_button.pack(pady=5)
    Tooltip(save_button, "Сохраните обработанное фото в выбранную папку.")

    retry_button = tk.Button(brightness_window, text="Переснять", command=retry, bg="#E04E39", fg="#FFFFFF",
                             font=("Arial", 14, "bold"), padx=10, pady=5, relief="flat", activebackground="#623B2A")
    retry_button.pack(pady=5)
    Tooltip(retry_button, "Вернитесь к съемке, чтобы сделать новый снимок.")

    update_preview()
    brightness_window.update_idletasks()
    brightness_window.geometry("")
    brightness_window.wait_window()
    return result[0]


def save_image(image, result_window):
    """Сохраняет обработанное изображение и обновляет статистику."""
    logging.info("Начало сохранения изображения")
    index = 1
    while os.path.exists(save_path := os.path.join(SAVE_DIR, f"photo_{index}.jpg")):
        index += 1
    image.save(save_path, "JPEG", quality=95, dpi=(300, 300))
    logging.info(f"Изображение сохранено: {save_path}")
    temp_path = os.path.join(SAVE_DIR, "webcam_photo.jpg")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Обновляем локальную статистику по дням
    update_local_daily_stats()

    # Запускаем синхронизацию в фоновом потоке
    executor.submit(sync_network_stats)

    os.startfile(SAVE_DIR)
    result_window.destroy()


def capture_photo():
    """Захват кадра"""
    global MIRROR_HORIZONTAL
    logging.info("Начало съёмки фото")
    if camera_manager.is_open():
        ret, frame = camera_manager.read()
        if not ret or frame is None:
            logging.error("Не удалось захватить кадр с камеры")
            messagebox.showerror("Ошибка", "Не удалось сделать фото")
            return
        if MIRROR_HORIZONTAL:
            frame = cv2.flip(frame, 1)
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(frame_gray)
            if mean_brightness < 50:
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
            elif mean_brightness > 200:
                frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-20)

            height, width = frame.shape[:2]
            target_width = int(height * 3 / 4)
            if width > target_width:
                left = (width - target_width) // 2
                frame = frame[:, left:left + target_width]
            photo_path = os.path.join(SAVE_DIR, "webcam_photo.jpg")
            cv2.imwrite(photo_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            logging.info(f"Фото сохранено: {photo_path}")
            result = process_image(photo_path, from_webcam=True)
            if result == "retry":
                return
        else:
            logging.error("Не удалось захватить кадр с камеры")
            messagebox.showerror("Ошибка", "Не удалось сделать фото")
    else:
        logging.error("Камера недоступна")
        messagebox.showerror("Ошибка", "Камера недоступна")


def sync_network_stats():
    """Синхронизирует локальную статистику с сетевым файлом за последние 30 дней."""
    local_stats_path = os.path.join(appdata_local, "PhotoPortal_daily_stats.txt")
    network_stats_path = r"\\zeu.local\SYSVOL\zeu.local\scripts\Software\scripts\checks\Photo_CKM\photo_stats.csv"
    hostname = os.environ.get('COMPUTERNAME', 'UnknownPC')
    cutoff_date = datetime.now() - timedelta(days=30)  # Ограничение на 30 дней

    # Проверяем наличие локального файла
    if not os.path.exists(local_stats_path):
        logging.info("Локальный файл статистики не найден, синхронизация не требуется")
        return

    try:
        # Читаем локальную статистику
        local_stats = {}
        with open(local_stats_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    date_str, count = line.strip().split(',')
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if date >= cutoff_date:  # Только последние 30 дней
                        local_stats[date_str] = int(count)

        # Проверяем и создаём сетевой файл, если его нет
        if not os.path.exists(network_stats_path):
            logging.info(f"Сетевой файл {network_stats_path} не найден, создание нового файла")
            try:
                with open(network_stats_path, "w", encoding="utf-8") as f:
                    f.write("Hostname,Date,PhotoCount\n")
            except IOError as e:
                logging.error(f"Не удалось создать сетевой файл: {e}")
                return

        # Читаем существующую сетевую статистику
        network_stats = {}
        with open(network_stats_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Пропускаем заголовок
                if line.strip():
                    host, date_str, count = line.strip().split(',')
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if date >= cutoff_date:  # Только последние 30 дней
                        network_stats[(host, date_str)] = int(count)

        # Обновляем сетевую статистику локальными данными и проверяем серверные значения
        updated_local_stats = local_stats.copy()
        for date, count in local_stats.items():
            key = (hostname, date)
            if key in network_stats:
                if network_stats[key] > count:  # Если серверное значение больше
                    updated_local_stats[date] = network_stats[key]  # Обновляем локальное
                elif count > network_stats[key]:  # Если локальное больше
                    network_stats[key] = count  # Обновляем сетевое
            else:
                network_stats[key] = count  # Добавляем новое значение в сеть

        # Записываем обновлённые данные в сетевой файл
        time.sleep(random.uniform(0.1, 0.5))  # Задержка для избежания конфликтов
        with open(network_stats_path, "w", encoding="utf-8") as f:
            f.write("Hostname,Date,PhotoCount\n")
            for (host, date), count in network_stats.items():
                f.write(f"{host},{date},{count}\n")  # Записываем только актуальные данные

        # Обновляем локальный файл с учётом серверных данных
        with open(local_stats_path, "w", encoding="utf-8") as f:
            for date, count in updated_local_stats.items():
                f.write(f"{date},{count}\n")
        logging.info(f"Статистика синхронизирована для {hostname} за последние 30 дней")

    except Exception as e:
        logging.error(f"Ошибка при синхронизации статистики: {e}")

def update_local_daily_stats():
    """Обновляет локальный файл ежедневной статистики."""
    today = time.strftime('%Y-%m-%d')
    local_stats_path = os.path.join(appdata_local, "PhotoPortal_daily_stats.txt")

    # Читаем существующие данные
    stats = {}
    if os.path.exists(local_stats_path):
        with open(local_stats_path, "r") as f:
            for line in f:
                if line.strip():
                    date, count = line.strip().split(',')
                    stats[date] = int(count)

    # Обновляем счётчик для текущего дня
    if today in stats:
        stats[today] += 1
    else:
        stats[today] = 1

    # Записываем данные обратно
    with open(local_stats_path, "w") as f:
        for date, count in stats.items():
            f.write(f"{date},{count}\n")
    logging.info(f"Локальная статистика обновлена: {today}, {stats[today]}")


def crop_interactively(image_path):
    logging.info(f"Открытие окна кадрирования для: {image_path}")
    crop_window = tk.Toplevel()
    crop_window.title("Кадрирование изображения")
    crop_window.iconbitmap(ICO_DIR)
    crop_window.configure(bg=BG)
    crop_window.grab_set()
    crop_window.attributes('-topmost', True)

    display_width, display_height = 300, 400
    original_image = PILImage.open(image_path).convert("RGB")
    orig_width, orig_height = original_image.size
    target_ratio = 3 / 4
    current_ratio = orig_width / orig_height
    if current_ratio > target_ratio:
        actual_height = orig_height
        actual_width = int(actual_height * target_ratio)
    else:
        actual_width = orig_width
        actual_height = int(actual_width / target_ratio)

    display_scale = display_width / actual_width
    scale_factor = 1.0
    img_display = original_image.copy()
    img_display_resized = img_display.resize((int(original_image.width * scale_factor * display_scale),
                                              int(original_image.height * scale_factor * display_scale)),
                                             PILImage.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_display_resized)

    canvas_frame = tk.Frame(crop_window, bg="#C39367", bd=1)
    canvas_frame.pack(pady=10)
    canvas = tk.Canvas(canvas_frame, width=display_width, height=display_height, bg="#FFFFFF")
    canvas.pack()
    image_id = canvas.create_image(display_width // 2, display_height // 2, image=img_tk)
    canvas.image = img_tk

    dragging = False
    start_x, start_y = 0, 0
    img_x, img_y = display_width // 2, display_height // 2

    def start_drag(event):
        nonlocal dragging, start_x, start_y
        dragging = True
        start_x, start_y = event.x, event.y

    def drag(event):
        nonlocal img_x, img_y, start_x, start_y
        if dragging:
            dx = event.x - start_x
            dy = event.y - start_y
            img_x += dx
            img_y += dy
            canvas.coords(image_id, img_x, img_y)
            start_x, start_y = event.x, event.y

    def stop_drag(event):
        nonlocal dragging
        dragging = False

    def resize(scale):
        nonlocal scale_factor, img_display_resized, img_tk, img_x, img_y
        scale_factor *= scale
        new_width = max(int(original_image.width * scale_factor * display_scale), 10)
        new_height = max(int(original_image.height * scale_factor * display_scale), 10)
        img_display_resized = original_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_display_resized)
        canvas.itemconfig(image_id, image=img_tk)
        canvas.image = img_tk
        canvas.coords(image_id, img_x, img_y)

    canvas.bind("<Button-1>", start_drag)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", stop_drag)
    canvas.bind("<MouseWheel>", lambda e: resize(1.1 if e.delta > 0 else 0.9))

    instruction_label = tk.Label(crop_window, text="Перетаскивайте фото под рамкой (3:4).\n Кнопки или колёсико мыши изменяют масштаб.",
                                 fg="#623B2A", bg=BG, font=("Arial", 10))
    instruction_label.pack(pady=5)
    Tooltip(instruction_label,
            "Переместите изображение, чтобы оно уместилось в рамку 3:4. Используйте колесо мыши или кнопки для изменения размера.")

    scale_frame = tk.Frame(crop_window, bg=BG)
    scale_frame.pack(pady=5)
    zoom_in_button = tk.Button(scale_frame, text="+", command=lambda: resize(1.1),
                               bg="#E04E39", fg="#FFFFFF", font=("Arial", 12, "bold"), width=2, relief="flat",
                               activebackground="#623B2A")
    zoom_in_button.pack(side="left", padx=5)
    Tooltip(zoom_in_button, "Увеличьте изображение для точной настройки.")
    zoom_out_button = tk.Button(scale_frame, text="−", command=lambda: resize(0.9),
                                bg="#E04E39", fg="#FFFFFF", font=("Arial", 12, "bold"), width=2, relief="flat",
                                activebackground="#623B2A")
    zoom_out_button.pack(side="left", padx=5)
    Tooltip(zoom_out_button, "Уменьшите изображение, если оно слишком большое.")

    button_frame = tk.Frame(crop_window, bg=BG)
    button_frame.pack(pady=10)
    confirm_button = tk.Button(button_frame, text="Подтвердить", command=crop_window.destroy,
                               bg="#623B2A", fg="#FFFFFF", font=("Arial", 14, "bold"), padx=10, pady=5,
                               relief="flat", activebackground="#E04E39")
    confirm_button.pack(side="left", padx=5)
    Tooltip(confirm_button, "Подтвердите кадрирование и перейдите к следующему шагу.")
    cancel_button = tk.Button(button_frame, text="Отмена",
                              command=lambda: crop_window.destroy() or setattr(crop_window, "result", "cancel"),
                              bg="#E04E39", fg="#FFFFFF", font=("Arial", 14, "bold"), padx=10, pady=5,
                              relief="flat", activebackground="#623B2A")
    cancel_button.pack(side="left", padx=5)
    Tooltip(cancel_button, "Отмените кадрирование и вернитесь к выбору изображения.")

    crop_window.result = None
    crop_window.wait_window()

    if getattr(crop_window, "result", None) == "cancel":
        return None

    current_width = int(original_image.width * scale_factor)
    current_height = int(original_image.height * scale_factor)
    frame_left = (current_width // 2) - int(img_x / display_scale)
    frame_top = (current_height // 2) - int(img_y / display_scale)
    orig_left = max(0, int(frame_left / scale_factor))
    orig_top = max(0, int(frame_top / scale_factor))
    orig_right = orig_left + int(actual_width / scale_factor)
    orig_bottom = orig_top + int(actual_height / scale_factor)
    orig_right = min(orig_right, original_image.width)
    orig_bottom = min(orig_bottom, original_image.height)

    crop_coords = (orig_left, orig_top, orig_right, orig_bottom)
    logging.info(f"Координаты обрезки: {crop_coords}")
    cropped_image = original_image.crop(crop_coords)

    scaled_width = cropped_image.width
    scaled_height = cropped_image.height
    target_ratio = 3 / 4
    current_ratio = scaled_width / scaled_height
    if current_ratio > target_ratio:
        scaled_width = int(scaled_height * target_ratio)
    else:
        scaled_height = int(scaled_width / target_ratio)
    scaled_image = cropped_image.resize((scaled_width, scaled_height), PILImage.Resampling.LANCZOS)

    final_image = PILImage.new("RGB", (scaled_width, scaled_height), (255, 255, 255))
    paste_x = int(img_x / display_scale)
    paste_y = int(img_y / display_scale)
    paste_x = max(0, min(paste_x, scaled_width - scaled_width))
    paste_y = max(0, min(paste_y, scaled_height - scaled_height))
    final_image.paste(scaled_image, (paste_x, paste_y))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_path = os.path.join(SAVE_DIR, f"cropped_temp_{timestamp}.jpg")
    final_image.save(temp_path, "JPEG")
    logging.info(f"Обрезанное изображение для обработки сохранено: {temp_path}, размер: {final_image.size}")
    return temp_path


def main():
    global root, webcam_label, app_closing
    app_closing = False
    logging.info("Запуск основного интерфейса")
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", shutdown_app)
    logging.info("Главное окно Tkinter создано")
    root.title("Подготовка фото для портала Mos.ru")
    try:
        icon_path = ICO_DIR
        logging.info(f"Путь к иконке: {icon_path}")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            logging.info("Иконка установлена")
        else:
            logging.warning(f"Иконка не найдена по пути: {icon_path}")
    except Exception as e:
        logging.error(f"Ошибка при установке иконки: {e}")
    root.geometry("800x750")
    root.resizable(False, False)
    root.configure(bg="#F5F5E6")
    logging.info("Параметры окна установлены")

    webcam_frame = tk.Frame(root, bd=1, bg="#C39367")
    webcam_frame.pack(pady=20)
    webcam_label = tk.Label(webcam_frame, bg="#2C2A29")
    webcam_label.pack()

    sensitivity_frame = tk.Frame(root, bg="#F5F5E6")
    sensitivity_frame.pack(pady=5)
    sensitivity_label = tk.Label(sensitivity_frame, text="Степень удаления фона",
                                 fg="#623B2A", bg="#F5F5E6", font=("Arial", 12), padx=5, pady=2)
    sensitivity_label.pack(side="left")
    sensitivity_value = tk.DoubleVar(value=SENSITIVITY_THRESHOLD)
    sensitivity_slider = tk.Scale(sensitivity_frame, from_=0.1, to=0.9, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=sensitivity_value,
                                  length=200, fg="#623B2A", bg="#F5F5E6",
                                  troughcolor="#C39367", activebackground="#E04E39",
                                  command=lambda val: debounce_update_sensitivity(val))
    sensitivity_slider.pack(side="left", padx=10)
    Tooltip(sensitivity_slider, "Настройте, насколько сильно убирать фон. Сдвиньте влево, чтобы убрать больше фона, вправо — чтобы оставить больше деталей.")

    update_camera(camera_index)

    capture_button = tk.Button(root, text="Сделать фото", command=capture_photo,
                               width=17, height=2, bg="#E04E39", fg="#FFFFFF",
                               font=("Arial", 16, "bold"), padx=10, pady=5, relief="flat",
                               activebackground="#623B2A")
    capture_button.pack(pady=10)
    Tooltip(capture_button, "Сделайте снимок с камеры для подготовки фото.")

    load_button = tk.Button(root, text="Загрузить свое фото", command=load_image,
                            width=17, height=2, bg="#623B2A", fg="#FFFFFF",
                            font=("Arial", 16, "bold"), padx=10, pady=5, relief="flat",
                            activebackground="#E04E39")
    load_button.pack(pady=10)
    Tooltip(load_button, "Загрузите готовое фото с вашего устройства для обработки.")

    settings_button = tk.Button(root, text="⚙", command=open_settings, bg="#E04E39", fg="#FFFFFF",
                                font=("Arial", 16, "bold"), padx=10, pady=5, relief="flat",
                                activebackground="#623B2A")
    settings_button.place(x=20, y=640)
    Tooltip(settings_button, "Откройте настройки для выбора камеры, папки сохранения и других параметров.")

    info_panel = tk.Frame(root, bg="#E0E0E0", bd=1, relief="sunken")
    info_panel.place(relx=0.0, rely=1.0, relwidth=1.0, height=30, anchor="sw")
    info_label = tk.Label(info_panel, text=f"PhotoPortal \u00A9 2025 ГБУ МФЦ г. Москвы ver. 1.4",
                          fg="#623B2A", bg="#E0E0E0", font=("Arial", 10))
    info_label.place(relx=0.5, rely=0.5, anchor="center")

    executor.submit(sync_network_stats)

    try:
        root.mainloop()
    finally:
        shutdown_app()


def shutdown_app():
    """Идемпотентно освобождает ресурсы и закрывает приложение."""
    global app_closing, webcam_after_id, selfie_segmentation

    if app_closing:
        return
    app_closing = True

    if webcam_after_id is not None:
        try:
            callback_owner = webcam_label if webcam_label is not None else root
            if callback_owner is not None:
                callback_owner.after_cancel(webcam_after_id)
        except tk.TclError:
            logging.debug("Callback камеры уже был удалён при завершении")
        except Exception:
            logging.exception("Не удалось отменить callback камеры")
        finally:
            webcam_after_id = None

    debounce_id = getattr(debounce_update_sensitivity, 'debounce_id', None)
    if debounce_id is not None:
        try:
            root.after_cancel(debounce_id)
        except tk.TclError:
            logging.debug("Debounce callback уже был удалён при завершении")
        except Exception:
            logging.exception("Не удалось отменить debounce callback")
        finally:
            delattr(debounce_update_sensitivity, 'debounce_id')

    try:
        camera_manager.close()
    except Exception:
        logging.exception("Не удалось освободить камеру")

    # Дожидаемся уже выполняющейся сегментации до закрытия её MediaPipe-модели.
    try:
        try:
            segmentation_executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            segmentation_executor.shutdown(wait=True)
    except Exception:
        logging.exception("Не удалось завершить segmentation executor")

    if selfie_segmentation is not None:
        try:
            selfie_segmentation.close()
        except Exception:
            logging.exception("Не удалось закрыть MediaPipe SelfieSegmentation")
        finally:
            selfie_segmentation = None

    try:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
    except Exception:
        logging.exception("Не удалось завершить работу executor")

    if root is not None:
        try:
            root.destroy()
        except tk.TclError:
            logging.debug("Главное окно уже было закрыто")
        except Exception:
            logging.exception("Не удалось закрыть главное окно")




def debounce_update_sensitivity(value):
    """Ползунок чуствительности обрьезки"""
    global SENSITIVITY_THRESHOLD, debounce_id
    if hasattr(debounce_update_sensitivity, 'debounce_id'):
        root.after_cancel(debounce_update_sensitivity.debounce_id)
    debounce_update_sensitivity.debounce_id = root.after(200, lambda: update_sensitivity_debounced(value))
def update_sensitivity_debounced(value):
    global SENSITIVITY_THRESHOLD
    SENSITIVITY_THRESHOLD = float(value)
    logging.info(f"Чувствительность обрезки фона изменена на: {SENSITIVITY_THRESHOLD}")
def update_sensitivity(value):
    debounce_update_sensitivity(value)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Аварийное завершение PhotoPortal")
        shutdown_app()
