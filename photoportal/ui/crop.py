"""Interactive crop dialog helpers."""
import logging
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
from ..storage import create_temp_path, remove_temp_file
from ..runtime import BG

def calculate_crop_rectangle(image_size, zoom, offset_x, offset_y, canvas_size=(300, 400)):
    image_width, image_height = image_size; canvas_width, canvas_height = canvas_size
    if zoom <= 0: raise ValueError("zoom must be positive")
    unit = int(min(canvas_width / (3 * zoom), canvas_height / (4 * zoom)) + 1e-9)
    if unit < 1: raise ValueError("zoom is too large to produce an integer 3:4 crop")
    width, height = 3 * unit, 4 * unit
    left = round(-offset_x / zoom + (canvas_width / zoom - width) / 2)
    top = round(-offset_y / zoom + (canvas_height / zoom - height) / 2)
    left = max(0, min(left, image_width-width)); top = max(0, min(top, image_height-height))
    return left, top, left+width, top+height

def crop_interactively(parent, image_path, icon_setter=lambda _window: None):
    window = tk.Toplevel(parent); window.title("Кадрирование изображения"); icon_setter(window); window.configure(bg=BG); window.grab_set(); window.attributes("-topmost", True)
    display_width, display_height = 300, 400
    with Image.open(image_path) as source: original = ImageOps.exif_transpose(source).convert("RGB")
    ow, oh = original.size; min_zoom = max(display_width/ow, display_height/oh); state = {"zoom": min_zoom, "x": (display_width-ow*min_zoom)/2, "y": (display_height-oh*min_zoom)/2, "drag": None}
    canvas = tk.Canvas(window, width=display_width, height=display_height, bg="#FFFFFF"); canvas.pack(pady=10)
    def redraw():
        z=state["zoom"]; photo=ImageTk.PhotoImage(original.resize((round(ow*z), round(oh*z)), Image.Resampling.LANCZOS)); canvas.delete("all"); canvas.create_image(state["x"], state["y"], image=photo, anchor="nw"); canvas.image=photo
    def constrain(): state["x"]=max(display_width-ow*state["zoom"], min(0,state["x"])); state["y"]=max(display_height-oh*state["zoom"], min(0,state["y"]))
    def zoom(scale):
        old=state["zoom"]; state["zoom"]=max(min_zoom, old*scale); state["x"]=display_width/2-(display_width/2-state["x"])*state["zoom"]/old; state["y"]=display_height/2-(display_height/2-state["y"])*state["zoom"]/old; constrain(); redraw()
    def start(event): state["drag"]=(event.x,event.y)
    def drag(event):
        if state["drag"]:
            x,y=state["drag"]; state["x"] += event.x-x; state["y"] += event.y-y; state["drag"]=(event.x,event.y); constrain(); redraw()
    canvas.bind("<Button-1>", start); canvas.bind("<B1-Motion>", drag); canvas.bind("<ButtonRelease-1>", lambda _e: state.update(drag=None)); canvas.bind("<MouseWheel>", lambda e: zoom(1.1 if e.delta>0 else .9)); redraw()
    tk.Label(window,text="Перетаскивайте фото под рамкой (3:4).\n Кнопки или колёсико мыши изменяют масштаб.",fg="#623B2A",bg=BG).pack(pady=5)
    controls=tk.Frame(window,bg=BG); controls.pack(); tk.Button(controls,text="+",command=lambda:zoom(1.1)).pack(side="left",padx=5); tk.Button(controls,text="−",command=lambda:zoom(.9)).pack(side="left",padx=5)
    window.result=None; buttons=tk.Frame(window,bg=BG); buttons.pack(pady=10); tk.Button(buttons,text="Подтвердить",command=window.destroy,bg="#623B2A",fg="white").pack(side="left",padx=5); tk.Button(buttons,text="Отмена",command=lambda:(setattr(window,"result","cancel"),window.destroy()),bg="#E04E39",fg="white").pack(side="left",padx=5); window.wait_window()
    if window.result == "cancel": return None
    final=original.crop(calculate_crop_rectangle(original.size,state["zoom"],state["x"],state["y"])); path=create_temp_path()
    try: final.save(path,"JPEG")
    except Exception: remove_temp_file(path); raise
    logging.info("Обрезанное изображение сохранено: %s", path); return path
