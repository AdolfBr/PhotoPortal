"""Tk application controller for PhotoPortal."""
from concurrent.futures import ThreadPoolExecutor
import logging, os, time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .runtime import BG

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

from camera_selection import select_camera_id
from .camera import CameraManager
from .config import AppConfig
from .image_processor import ImageProcessor
from .segmentation import create_selfie_segmentation
from .statistics import sync_network_stats, update_local_daily_stats
from .storage import PhotoStorage, create_temp_path, initialize_temp_directory, remove_temp_file
from .ui import Tooltip, crop_interactively


class PhotoPortalApp:
    """Own all application services, Tk state, widgets, and callback IDs."""
    PREVIEW_INTERVAL_MS = 36
    SEGMENTATION_INTERVAL = .1

    def __init__(self, root=None, *, config=None, camera_manager=None,
                 segmentation_model=None, executor=None, segmentation_executor=None):
        self.root = root
        self.config = (config or AppConfig()).load()
        if not os.path.exists(self.config.path): self.config.save()
        self.camera_manager = camera_manager or CameraManager()
        if segmentation_model is None:
            segmentation_model = create_selfie_segmentation()
        self.segmentation_model = segmentation_model
        self.image_processor = ImageProcessor(segmentation_model, self.config.Sensitivity)
        self.storage = PhotoStorage(self.config.SaveDir)
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.segmentation_executor = segmentation_executor or ThreadPoolExecutor(max_workers=1)
        self.settings = {
            "save_dir": self.config.SaveDir, "camera_index": self.config.CameraIndex,
            "mirror_horizontal": self.config.MirrorHorizontal, "sensitivity": self.config.Sensitivity,
            "blur_strength": self.config.BlurStrength, "num_threads": self.config.NumThreads,
            "icon_path": self.config.IcoDir,
        }
        self.widgets = {}
        self.callback_ids = {"webcam": None, "sensitivity": None}
        self.closing = False; self.segmentation_future = None; self.last_segmentation_submit = 0.0
        self.latest_segmentation_mask = None; self.segmentation_generation = 0
        self.last_webcam_read_error_log_time = None
        cv2.setNumThreads(self.settings["num_threads"])
        initialize_temp_directory()

    def set_window_icon(self, window):
        path = self.settings["icon_path"]
        if not path: return
        try:
            if os.path.exists(path): window.iconbitmap(path)
            else: logging.warning("Иконка не найдена по пути: %s", path)
        except (OSError, TypeError, ValueError, tk.TclError): logging.exception("Не удалось установить иконку окна: %s", path)

    def build_ui(self):
        if self.root is None: self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown); self.root.title("Подготовка фото для портала Mos.ru"); self.set_window_icon(self.root)
        self.root.geometry("800x750"); self.root.resizable(False, False); self.root.configure(bg=BG)
        webcam_frame=tk.Frame(self.root,bd=1,bg="#C39367"); webcam_frame.pack(pady=20)
        self.widgets["webcam_label"]=tk.Label(webcam_frame,bg="#2C2A29"); self.widgets["webcam_label"].pack()
        frame=tk.Frame(self.root,bg=BG); frame.pack(pady=5); tk.Label(frame,text="Степень удаления фона",fg="#623B2A",bg=BG,font=("Arial",12),padx=5,pady=2).pack(side="left")
        sensitivity=tk.DoubleVar(value=self.settings["sensitivity"]); self.widgets["sensitivity_value"]=sensitivity
        slider=tk.Scale(frame,from_=.1,to=.9,resolution=.05,orient=tk.HORIZONTAL,variable=sensitivity,length=200,fg="#623B2A",bg=BG,troughcolor="#C39367",activebackground="#E04E39",command=self.debounce_sensitivity); slider.pack(side="left",padx=10); Tooltip(slider,"Настройте, насколько сильно убирать фон.")
        capture=tk.Button(self.root,text="Сделать фото",command=self.capture_and_process_photo,width=17,height=2,bg="#E04E39",fg="white",font=("Arial",16,"bold"),padx=10,pady=5,relief="flat",activebackground="#623B2A"); capture.pack(pady=10)
        load=tk.Button(self.root,text="Загрузить свое фото",command=self.load_image,width=17,height=2,bg="#623B2A",fg="white",font=("Arial",16,"bold"),padx=10,pady=5,relief="flat",activebackground="#E04E39"); load.pack(pady=10)
        settings=tk.Button(self.root,text="⚙",command=self.open_settings,bg="#E04E39",fg="white",font=("Arial",16,"bold"),padx=10,pady=5,relief="flat",activebackground="#623B2A"); settings.place(x=20,y=640)
        panel=tk.Frame(self.root,bg="#E0E0E0",bd=1,relief="sunken"); panel.place(relx=0,rely=1,relwidth=1,height=30,anchor="sw"); tk.Label(panel,text="PhotoPortal © 2025 ГБУ МФЦ г. Москвы ver. 1.4",fg="#623B2A",bg="#E0E0E0",font=("Arial",10)).place(relx=.5,rely=.5,anchor="center")
        self.widgets.update(capture_button=capture,load_button=load,settings_button=settings)
        cameras=self.camera_manager.discover(); selected=select_camera_id([item["id"] for item in cameras],self.settings["camera_index"])
        if selected is not None: self.settings["camera_index"]=selected; self.update_camera(selected)
        else: messagebox.showwarning("Предупреждение","Не найдено доступных камер.")
        self.executor.submit(self._sync_network_stats_safely)
        return self.root

    def run(self):
        self.build_ui()
        try: self.root.mainloop()
        finally: self.shutdown()

    def debounce_sensitivity(self, value):
        callback=self.callback_ids["sensitivity"]
        if callback is not None: self.root.after_cancel(callback)
        self.callback_ids["sensitivity"]=self.root.after(200,lambda:self.apply_sensitivity(value))

    def apply_sensitivity(self, value):
        self.callback_ids["sensitivity"]=None; self.settings["sensitivity"]=float(value); self.config.Sensitivity=float(value); self.image_processor.sensitivity=float(value)

    def schedule_webcam_update(self):
        label=self.widgets.get("webcam_label")
        if not self.closing and label is not None: self.callback_ids["webcam"]=label.after(self.PREVIEW_INTERVAL_MS,self.show_webcam)

    def _analyse_preview(self, frame, sensitivity):
        height,width=frame.shape[:2]; scale=min(240/width,320/height); scaled=cv2.resize(frame,(max(1,int(width*scale)),max(1,int(height*scale))),interpolation=cv2.INTER_AREA); rgb=cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)
        result=self.segmentation_model.process(rgb); mask=(result.segmentation_mask>sensitivity).astype(np.uint8)*255; kernel=np.ones((3,3),np.uint8); mask=cv2.dilate(mask,kernel,iterations=1); mask=cv2.erode(mask,kernel,iterations=1); return cv2.threshold(cv2.GaussianBlur(mask,(3,3),1.5),127,255,cv2.THRESH_BINARY)[1]

    def _store_segmentation(self, future, generation):
        try: mask=future.result()
        except Exception: logging.exception("Ошибка сегментации preview"); return
        if generation == self.segmentation_generation: self.latest_segmentation_mask=mask

    def show_webcam(self):
        self.callback_ids["webcam"]=None
        if self.closing or not self.camera_manager.is_open(): return
        ok,frame=self.camera_manager.read()
        if not ok or frame is None: self.schedule_webcam_update(); return
        if self.settings["mirror_horizontal"]: frame=cv2.flip(frame,1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); mean=np.mean(gray); lines=["Слишком темно!","Увеличьте освещение."] if mean<70 else (["Слишком ярко за спиной!","Попробуйте сменить ракурс"] if mean>200 else None)
        height,width=frame.shape[:2]; target=int(height*3/4)
        if width>target: left=(width-target)//2; frame=frame[:,left:left+target]
        now=time.monotonic()
        if (self.segmentation_future is None or self.segmentation_future.done()) and now-self.last_segmentation_submit>=self.SEGMENTATION_INTERVAL:
            self.last_segmentation_submit=now; generation=self.segmentation_generation; self.segmentation_future=self.segmentation_executor.submit(self._analyse_preview,frame.copy(),self.settings["sensitivity"]); self.segmentation_future.add_done_callback(lambda future:self._store_segmentation(future,generation))
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if self.latest_segmentation_mask is not None:
            mask=cv2.resize(self.latest_segmentation_mask,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST); rgb[mask==0]=[255,255,255]
        pil=Image.fromarray(cv2.resize(rgb,(300,400),interpolation=cv2.INTER_LANCZOS4))
        if lines:
            draw=ImageDraw.Draw(pil)
            try: font=ImageFont.truetype("arial.ttf",20)
            except IOError: font=ImageFont.load_default()
            y=170
            for line in lines: draw.text((150, y),line,fill=(255,0,0),font=font,anchor="mm"); y+=25
        photo=ImageTk.PhotoImage(pil); label=self.widgets["webcam_label"]; label.config(image=photo); label.image=photo; self.schedule_webcam_update()

    def update_camera(self, index):
        callback=self.callback_ids["webcam"]
        if callback is not None:
            try: self.widgets["webcam_label"].after_cancel(callback)
            except tk.TclError: pass
            self.callback_ids["webcam"]=None
        self.segmentation_generation+=1; self.latest_segmentation_mask=None
        if not self.closing and self.camera_manager.switch(index): self.show_webcam()
        elif not self.closing: messagebox.showerror("Ошибка","Не удалось открыть камеру")

    def load_image(self):
        path=filedialog.askopenfilename(filetypes=[("Изображения","*.jpg;*.jpeg;*.png")])
        if not path: return
        cropped=crop_interactively(self.root,path,self.set_window_icon)
        if cropped:
            try:
                if self.process_image(cropped,from_webcam=False)=="retry": self.load_image()
            finally: remove_temp_file(cropped)

    def process_image(self, image_path, from_webcam=False):
        """Process an image; origin explicitly controls the retry workflow."""
        try:
            with Image.open(image_path) as image: source=ImageOps.exif_transpose(image).convert("RGB")
            self.image_processor=ImageProcessor(self.segmentation_model,self.settings["sensitivity"]); self.image_processor.segment(source)
            return self.adjust_brightness(from_webcam=from_webcam)
        except Exception as error: logging.exception("Ошибка обработки изображения"); messagebox.showerror("Ошибка",f"Ошибка обработки: {error}")

    def adjust_brightness(self, from_webcam=False):
        window=tk.Toplevel(self.root); window.title("Настройка яркости и сглаживания"); self.set_window_icon(window); window.configure(bg=BG); window.grab_set(); window.attributes("-topmost",True)
        tk.Label(window,text="Настройте параметры изображения с помощью ползунков.",fg="#623B2A",bg=BG).pack(pady=10); frame=tk.Frame(window,bd=1,bg="#C39367"); frame.pack(pady=10)
        label=tk.Label(frame,bg="white"); label.pack(); brightness=tk.IntVar(value=0); softness=tk.DoubleVar(value=self.settings["blur_strength"]); adjusted=[None]; result=[None]
        def update(_value=None):
            adjusted[0]=self.image_processor.adjust_brightness(brightness.get(),softness.get()); thumbnail=adjusted[0].copy(); thumbnail.thumbnail((300,400)); photo=ImageTk.PhotoImage(thumbnail); label.config(image=photo); label.image=photo
        tk.Scale(window,from_=-50,to=50,orient=tk.HORIZONTAL,label="Яркость фото",variable=brightness,length=300,fg="#623B2A",bg=BG,command=update).pack(pady=10)
        tk.Scale(window,from_=0,to=10,resolution=.5,orient=tk.HORIZONTAL,label="Мягкость краёв",variable=softness,length=300,fg="#623B2A",bg=BG,command=update).pack(pady=10)
        def save():
            if self.save_image(self.image_processor.resize(adjusted[0],600,800),window): result[0]="saved"
        def retry(): result[0]="retry"; window.destroy()
        tk.Button(window,text="Сохранить",command=save,bg="#623B2A",fg="white",font=("Arial",14,"bold"),padx=10,pady=5).pack(pady=5)
        # Camera photos return to the live preview; uploaded photos return to file selection.
        retry_text="Переснять" if from_webcam else "Выбрать другое фото"
        tk.Button(window,text=retry_text,command=retry,bg="#E04E39",fg="white",font=("Arial",14,"bold"),padx=10,pady=5).pack(pady=5)
        update(); window.wait_window(); return result[0]

    def capture_photo(self):
        if not self.camera_manager.is_open(): messagebox.showerror("Ошибка","Камера недоступна"); return None
        ok,frame=self.camera_manager.read()
        if not ok or frame is None: messagebox.showerror("Ошибка","Не удалось сделать фото"); return None
        if self.settings["mirror_horizontal"]: frame=cv2.flip(frame,1)
        mean=np.mean(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        if mean<50: frame=cv2.convertScaleAbs(frame,alpha=1.2,beta=20)
        elif mean>200: frame=cv2.convertScaleAbs(frame,alpha=.9,beta=-20)
        height,width=frame.shape[:2]; target=int(height*3/4)
        if width>target: left=(width-target)//2; frame=frame[:,left:left+target]
        path=create_temp_path()
        if not cv2.imwrite(path,frame,[int(cv2.IMWRITE_JPEG_QUALITY),100]): remove_temp_file(path); raise OSError("OpenCV не удалось записать снимок")
        return path

    def capture_and_process_photo(self):
        path=None
        try:
            path=self.capture_photo()
            return self.process_image(path,from_webcam=True) if path else None
        except Exception as error: logging.exception("Ошибка записи или обработки снимка"); messagebox.showerror("Ошибка",f"Ошибка обработки: {error}")
        finally: remove_temp_file(path)

    def save_image(self, image, result_window):
        path=self.storage.save(image)
        if not path: return None
        try:
            try: update_local_daily_stats()
            except Exception: logging.exception("Не удалось обновить локальную статистику")
            self.executor.submit(self._sync_network_stats_safely)
            try: os.startfile(self.settings["save_dir"])
            except (AttributeError,OSError): logging.exception("Не удалось открыть каталог сохранения")
        finally: result_window.destroy()
        return path

    def _sync_network_stats_safely(self):
        try: sync_network_stats()
        except Exception: logging.exception("Ошибка сетевой синхронизации")

    def open_settings(self):
        window=tk.Toplevel(self.root); window.title("Настройки"); self.set_window_icon(window); window.geometry("400x600"); window.configure(bg=BG); window.grab_set(); window.attributes("-topmost",True)
        main=tk.Frame(window,bg=BG); main.pack(pady=20,padx=20,fill="both",expand=True); save_label=tk.Label(main,text=self.settings["save_dir"],fg="#623B2A",bg=BG); save_label.pack()
        def choose():
            directory=filedialog.askdirectory()
            if directory: self.settings["save_dir"]=directory; save_label.config(text=directory)
        tk.Button(main,text="Выбрать папку сохранения",command=choose,bg="#E04E39",fg="white").pack(pady=5)
        cameras=self.camera_manager.discover(); combo=ttk.Combobox(main,values=[c["name"] for c in cameras]); combo.pack(pady=10)
        selected=select_camera_id([c["id"] for c in cameras],self.settings["camera_index"])
        if selected is not None: combo.current(next(i for i,c in enumerate(cameras) if c["id"]==selected))
        mirror=tk.BooleanVar(value=self.settings["mirror_horizontal"]); tk.Checkbutton(main,text="Зеркальное отображение",variable=mirror,bg=BG).pack(pady=10)
        threads=ttk.Combobox(main,values=list(range(1,(os.cpu_count() or 1)+1))); threads.set(self.settings["num_threads"]); threads.pack(pady=10)
        def apply():
            position=combo.current(); camera_id=cameras[position]["id"] if position>=0 and cameras else None
            self.settings.update(mirror_horizontal=mirror.get(),num_threads=int(threads.get()))
            if camera_id is not None: self.settings["camera_index"]=camera_id
            for attr,key in (("SaveDir","save_dir"),("CameraIndex","camera_index"),("MirrorHorizontal","mirror_horizontal"),("NumThreads","num_threads"),("Sensitivity","sensitivity"),("BlurStrength","blur_strength")): setattr(self.config,attr,self.settings[key])
            self.config.save(); self.storage=PhotoStorage(self.settings["save_dir"]); cv2.setNumThreads(self.settings["num_threads"]); window.destroy()
            if camera_id is not None: self.update_camera(camera_id)
        tk.Button(main,text="Применить",command=apply,bg="#623B2A",fg="white",font=("Arial",14,"bold")).pack(pady=20)

    def shutdown(self):
        if self.closing: return
        self.closing=True
        for key,callback in list(self.callback_ids.items()):
            if callback is not None and self.root is not None:
                try: self.root.after_cancel(callback)
                except (tk.TclError,AttributeError): pass
                self.callback_ids[key]=None
        self.camera_manager.close()
        for pool,wait in ((self.segmentation_executor,True),(self.executor,False)):
            try: pool.shutdown(wait=wait,cancel_futures=True)
            except TypeError: pool.shutdown(wait=wait)
        try: self.segmentation_model.close()
        except (AttributeError,RuntimeError): pass
        if self.root is not None:
            try: self.root.destroy()
            except tk.TclError: pass
