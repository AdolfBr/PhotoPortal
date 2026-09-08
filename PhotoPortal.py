import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
cap = None
webcam_label = None
BG = "#F5F5E6"
SAVE_DIR = appdata_local
ICO_DIR = "C:\Program Files\PhotoPortal\icon.ico"
MIRROR_HORIZONTAL = False
NUM_THREADS = 4
MAX_THREADS = 4
SENSITIVITY_THRESHOLD = 0.6
BLUR_STRENGTH = 5.5






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


SAVE_DIR, camera_index, ICO_DIR, MIRROR_HORIZONTAL = load_settings()


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
    camera_combobox = ttk.Combobox(camera_frame, values=cameras, width=20)
    camera_combobox.pack(anchor="w", padx=5, pady=2)
    if cameras:
        camera_combobox.current(min(camera_index, len(cameras) - 1))
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
        global SAVE_DIR, ICO_DIR, MIRROR_HORIZONTAL, NUM_THREADS
        selected_index = camera_combobox.current()
        MIRROR_HORIZONTAL = mirror_var.get()
        NUM_THREADS = int(threads_combobox.get())
        os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
        os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
        cv2.setNumThreads(NUM_THREADS)
        save_settings(SAVE_DIR, selected_index if cameras else 0, ICO_DIR, MIRROR_HORIZONTAL, NUM_THREADS)
        messagebox.showinfo("Успех", "Настройки сохранены!")
        settings_window.destroy()
        if selected_index is not None and cameras:
            update_camera(selected_index)
        else:
            logging.warning("Не выбрана камера или камеры отсутствуют")
            messagebox.showwarning("Предупреждение", "Камера не выбрана или отсутствует.")


def get_available_cameras():
    """Получение доступных камер"""
    logging.info("Поиск доступных камер")
    cameras = []
    for i in range(10):
        temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            cameras.append(f"Камера {i}")
            temp_cap.release()
    logging.info(f"Доступные камеры: {cameras}")
    return cameras


def show_webcam():
    """Отображение изображения с камеры"""
    global cap, webcam_label, MIRROR_HORIZONTAL, SENSITIVITY_THRESHOLD
    if cap is None:
        logging.warning("Камера не инициализирована")
        return
    ret, frame = cap.read()
    if MIRROR_HORIZONTAL:
        frame = cv2.flip(frame, 1)
    if ret:
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
        frame_small = cv2.resize(frame, (320, 240))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        results = selfie_segmentation.process(frame_rgb)
        mask = results.segmentation_mask > SENSITIVITY_THRESHOLD
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)
        mask_smoothed = cv2.GaussianBlur(mask_eroded, (3, 3), sigmaX=1.5, sigmaY=1.5)
        _, mask_smoothed = cv2.threshold(mask_smoothed, 127, 255, cv2.THRESH_BINARY)
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
            max_width = max(line_widths)
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
        webcam_label.after(33, show_webcam)
    else:
        logging.error("Не удалось получить кадр с камеры")


def update_camera(index):
    """Подключение к камере"""
    global cap
    logging.info(f"Переключение на камеру {index}")
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть камеру {index}")
        messagebox.showerror("Ошибка", "Не удалось открыть камеру")
        cap = None
    else:
        logging.info(
            f"Камера {index} открыта с разрешением {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
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
        image_np = np.array(input_image)
        results = selfie_segmentation.process(image_np)
        mask = results.segmentation_mask > SENSITIVITY_THRESHOLD
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_eroded = cv2.erode(mask_dilated, kernel, iterations=1)
        mask_smoothed = cv2.GaussianBlur(mask_eroded, (3, 3), sigmaX=1.5, sigmaY=1.5)
        _, mask_smoothed = cv2.threshold(mask_smoothed, 127, 255, cv2.THRESH_BINARY)
        rgb = image_np.copy()
        alpha = mask_smoothed
        rgb[alpha == 0] = [255, 255, 255]
        output_image = PILImage.fromarray(np.dstack((rgb, alpha)))
        enhanced_image = enhance_image(output_image)
        final_image = ensure_min_size(enhanced_image, min_width=600, min_height=800)
        return adjust_brightness(final_image)
    except Exception as e:
        logging.error(f"Ошибка обработки изображения: {e}")
        messagebox.showerror("Ошибка", f"Ошибка обработки: {e}")
        return None


def enhance_image(image):
    """Улучшение результата"""
    logging.info("Улучшение качества изображения")
    image_np = np.array(image.convert("RGB"))
    enhanced = np.clip(image_np, 0, 255).astype(np.uint8)
    enhanced_pil = PILImage.fromarray(enhanced)
    if image.mode == "RGBA":
        alpha = image.split()[3]
        enhanced_pil.putalpha(alpha)
    return enhanced_pil


def ensure_min_size(image, min_width=600, min_height=800):
    """Увелечение до минимального размера"""
    logging.info("Проверка размера изображения")
    width, height = image.size
    if width < min_width or height < min_height:
        ratio = max(min_width / width, min_height / height)
        new_width, new_height = int(width * ratio), int(height * ratio)
        resized = image.resize((new_width, new_height), PILImage.Resampling.BICUBIC)
        logging.info(f"Изображение увеличено до {new_width}x{new_height}")
        return resized
    logging.info(f"Размер изображения сохранен: {width}x{height}")
    return image


def adjust_brightness(image, from_webcam=False):
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
    thumbnail = image.copy()
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

    adjusted_image = [image]
    original_np = np.array(image)
    rgb_original = original_np[:, :, :3]
    alpha = original_np[:, :, 3] if image.mode == "RGBA" else None

    results = selfie_segmentation.process(rgb_original)
    cached_mask = (results.segmentation_mask > SENSITIVITY_THRESHOLD).astype(np.uint8) * 255

    def update_preview(*args):
        brightness = brightness_value.get()
        blur_strength = blur_value.get()
        rgb = rgb_original.copy()

        if blur_strength > 0:
            kernel_size = max(3, int(blur_strength * 4) | 1)
            sigma = blur_strength * 2
            mask_blurred = cv2.GaussianBlur(cached_mask, (kernel_size, kernel_size), sigmaX=sigma)
            _, mask_blurred = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
            rgb[mask_blurred == 0] = [255, 255, 255]
        else:
            rgb[cached_mask == 0] = [255, 255, 255]

        adjusted_rgb = np.clip(rgb.astype(float) + brightness, 0, 255).astype(np.uint8)
        adjusted_image[0] = PILImage.fromarray(
            np.dstack((adjusted_rgb, alpha))) if alpha is not None else PILImage.fromarray(adjusted_rgb)

        thumbnail = adjusted_image[0].copy()
        thumbnail.thumbnail((300, 400))
        new_img_tk = ImageTk.PhotoImage(thumbnail)
        label.config(image=new_img_tk)
        label.image = new_img_tk

    result = [None]

    def save():
        white_bg = PILImage.new("RGBA", adjusted_image[0].size, (255, 255, 255, 255))
        if adjusted_image[0].mode == "RGBA":
            white_bg.paste(adjusted_image[0], (0, 0), adjusted_image[0].split()[3])
        else:
            white_bg.paste(adjusted_image[0], (0, 0))
        white_bg = white_bg.convert("RGB")
        save_image(white_bg, brightness_window)
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
    if cap and cap.isOpened():
        ret, frame = cap.read()
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
    global root, webcam_label
    logging.info("Запуск основного интерфейса")
    root = tk.Tk()
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
        if cap:
            cap.release()
            logging.info("Камера освобождена")




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
    main()