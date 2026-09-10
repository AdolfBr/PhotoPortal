"""Reusable Tk widgets."""
import tkinter as tk

class Tooltip:
    def __init__(self, widget, text, delay=2000):
        self.widget, self.text, self.delay = widget, text, delay; self.tooltip = None; self.after_id = None
        widget.bind("<Enter>", self.schedule); widget.bind("<Leave>", self.hide)
    def schedule(self, _event=None):
        if self.after_id: self.widget.after_cancel(self.after_id)
        self.after_id = self.widget.after(self.delay, self.show)
    def show(self):
        if not self.after_id: return
        self.tooltip = tk.Toplevel(self.widget); self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{self.widget.winfo_rootx()+25}+{self.widget.winfo_rooty()+25}")
        tk.Label(self.tooltip, text=self.text, bg="#FFFFDD", fg="#623B2A", borderwidth=1, relief="solid").pack(); self.after_id = None
    def hide(self, _event=None):
        if self.after_id: self.widget.after_cancel(self.after_id); self.after_id = None
        if self.tooltip: self.tooltip.destroy(); self.tooltip = None
