#!/usr/bin/env python3

import os
import threading
import shutil
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import time


TEMP_DIR = "temp_watermarked"  # always save processed outputs here
VIDEO_EXT_PREFERRED = ".mp4"   # prefer mp4 for video output when possible


class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark App — visible / invisible (LSB)")
        self.root.geometry("900x720")
        self.root.minsize(800, 600)

        # State
        self.file_path = None
        self.is_video = False
        self.preview_image = None  # PIL image for preview
        self.watermark_text_var = tk.StringVar(value="Your Watermark")
        self.mode_var = tk.StringVar(value="visible")
        self.last_temp_path = None  # path inside TEMP_DIR for last processed file
        self.last_user_saved_dir = None  # folder chosen by user when saving
        self.processing_thread = None
        self.stop_processing = threading.Event()
        self.first_frame_path = None  # path untuk menyimpan frame pertama

        # build UI
        self._build_ui()
        os.makedirs(TEMP_DIR, exist_ok=True)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Top: file selection
        top_frame = ttk.LabelFrame(frm, text="File", padding=8)
        top_frame.pack(fill=tk.X, pady=(0,8))

        btn_select = ttk.Button(top_frame, text="Select Image / Video", command=self.select_file)
        btn_select.grid(row=0, column=0, padx=6, pady=4, sticky=tk.W)

        self.file_label = ttk.Label(top_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, padx=6, sticky=tk.W)

        # Middle: watermark settings
        settings = ttk.LabelFrame(frm, text="Watermark Settings", padding=8)
        settings.pack(fill=tk.X, pady=(0,8))

        ttk.Label(settings, text="Watermark text:").grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Entry(settings, textvariable=self.watermark_text_var, width=50).grid(row=0, column=1, padx=4, sticky=tk.W)

        ttk.Label(settings, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=(6,0))
        ttk.Radiobutton(settings, text="Visible (tiled)", variable=self.mode_var, value="visible").grid(row=1, column=1, sticky=tk.W, padx=4, pady=(6,0))
        ttk.Radiobutton(settings, text="Invisible (LSB)", variable=self.mode_var, value="invisible").grid(row=1, column=1, sticky=tk.W, padx=150, pady=(6,0))

        # Detection frame
        detect_frame = ttk.LabelFrame(frm, text="Invisible Watermark Detection", padding=8)
        detect_frame.pack(fill=tk.X, pady=(0,8))

        detect_btn = ttk.Button(detect_frame, text="Detect Invisible Watermark", command=self.detect_invisible_watermark)
        detect_btn.grid(row=0, column=0, padx=6, pady=4, sticky=tk.W)

        self.detection_label = ttk.Label(detect_frame, text="No detection performed")
        self.detection_label.grid(row=0, column=1, padx=6, sticky=tk.W)

        # Preview area
        preview_frame = ttk.LabelFrame(frm, text="Preview (images: full; videos: first frame)", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0,8))

        self.canvas = tk.Canvas(preview_frame, bg="#ddd")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom: actions and progress
        bottom = ttk.Frame(frm)
        bottom.pack(fill=tk.X)

        self.apply_btn = ttk.Button(bottom, text="Apply Watermark", command=self.apply_watermark)
        self.apply_btn.pack(side=tk.LEFT, padx=6, pady=6)

        self.save_again_btn = ttk.Button(bottom, text="Save Again (copy temp -> folder)", command=self.save_again, state=tk.DISABLED)
        self.save_again_btn.pack(side=tk.LEFT, padx=6, pady=6)

        self.cancel_btn = ttk.Button(bottom, text="Stop", command=self.request_stop)
        self.cancel_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        self.progress = ttk.Progressbar(bottom, orient=tk.HORIZONTAL, length=360, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=6, pady=6)

        self.status_label = ttk.Label(bottom, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=6)

    # ---------- UI Helpers ----------
    def select_file(self):
        types = [("Images & Videos", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(filetypes=types)
        if not path:
            return
        self.file_path = path
        self.is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        self.file_label.config(text=os.path.basename(path))
        self._load_preview()

    def _load_preview(self):
        self.preview_image = None
        self.canvas.delete("all")
        try:
            if self.is_video:
                cap = cv2.VideoCapture(self.file_path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError("Could not read first frame of video.")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb)
            else:
                pil = Image.open(self.file_path)
            self.preview_image = pil.copy()
            self._show_preview_image(self.preview_image)
        except Exception as e:
            messagebox.showerror("Preview error", f"Failed to load preview: {e}")
            self.status_label.config(text="Preview failed")

    def _show_preview_image(self, pil_img):
        # Fit into canvas while preserving aspect ratio
        canvas_w = self.canvas.winfo_width() or 600
        canvas_h = self.canvas.winfo_height() or 360
        img = pil_img.copy()
        img.thumbnail((canvas_w - 20, canvas_h - 20), Image.LANCZOS)
        self._tk_preview = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self._tk_preview, anchor=tk.CENTER)

    # ---------- Watermark Detection ----------
    def detect_invisible_watermark(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Select an image or video first.")
            return
        
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "Processing already running.")
            return

        self.progress['value'] = 0
        self.status_label.config(text="Detecting watermark...")
        self.detection_label.config(text="Detection in progress...")

        # Start detection in thread
        self.processing_thread = threading.Thread(target=self._detect_task, daemon=True)
        self.processing_thread.start()

    def _detect_task(self):
        try:
            if self.is_video:
                # Untuk video, coba deteksi dari frame pertama yang disimpan
                if self.first_frame_path and os.path.exists(self.first_frame_path):
                    # Gunakan frame pertama yang sudah disimpan (tanpa kompresi)
                    pil_img = Image.open(self.first_frame_path)
                    detected_text = self._detect_lsb_watermark(pil_img)
                else:
                    # Fallback: baca frame pertama dari video
                    cap = cv2.VideoCapture(self.file_path)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        raise RuntimeError("Could not read first frame for detection.")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    detected_text = self._detect_lsb_watermark(pil_img)
            else:
                # For image
                pil_img = Image.open(self.file_path)
                detected_text = self._detect_lsb_watermark(pil_img)
            
            self.root.after(0, lambda: self._on_detection_done(detected_text))
            
        except Exception as e:
            self.root.after(0, lambda: self._detection_failed(e))

    def _detect_lsb_watermark(self, pil_image):
        """Extract LSB watermark from image"""
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8)
        flat = arr.flatten()
        
        # Extract LSBs
        bits = ""
        max_bits_to_read = min(10000, flat.size)  # Limit for performance and to avoid garbage
        
        for i in range(max_bits_to_read):
            bits += str(flat[i] & 1)
        
        # Convert bits to text (8 bits per character)
        text = ""
        for i in range(0, len(bits) - 7, 8):
            byte = bits[i:i+8]
            char_code = int(byte, 2)
            if char_code == 0:  # Null terminator
                break
            if 32 <= char_code <= 126:  # Printable ASCII
                text += chr(char_code)
            else:
                # If we hit non-printable character, stop
                break
        
        return text if text else "No readable watermark found"

    def _on_detection_done(self, detected_text):
        self.status_label.config(text="Detection complete")
        self.progress['value'] = 100
        self.detection_label.config(text=f"Detected: {detected_text}")
        
        # Show result in messagebox
        if detected_text and detected_text != "No readable watermark found":
            messagebox.showinfo("Watermark Detection", f"Detected watermark: '{detected_text}'")
        else:
            messagebox.showinfo("Watermark Detection", "No watermark detected or watermark is not readable")

    def _detection_failed(self, exc):
        self.status_label.config(text="Detection failed")
        self.progress['value'] = 0
        self.detection_label.config(text="Detection failed")
        messagebox.showerror("Detection failed", str(exc))

    # ---------- Processing ----------
    def apply_watermark(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Select an image or video first.")
            return
        txt = self.watermark_text_var.get().strip()
        if not txt:
            messagebox.showwarning("No watermark text", "Enter watermark text.")
            return
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Busy", "Processing already running.")
            return

        # reset stop flag
        self.stop_processing.clear()
        self.progress['value'] = 0
        self.status_label.config(text="Processing...")
        self.apply_btn.config(state=tk.DISABLED)
        self.save_again_btn.config(state=tk.DISABLED)

        # start thread
        self.processing_thread = threading.Thread(target=self._process_task, daemon=True)
        self.processing_thread.start()
        # start a small poller to update preview canvas if needed (resizing)
        self._start_preview_resizer()

    def request_stop(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing.set()
            self.status_label.config(text="Stopping...")
        else:
            self.status_label.config(text="No active processing")

    def _process_task(self):
        try:
            os.makedirs(TEMP_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            ext = os.path.splitext(self.file_path)[1].lower()

            # choose temp output path
            if self.is_video:
                out_ext = VIDEO_EXT_PREFERRED if VIDEO_EXT_PREFERRED else ext or ".mp4"
                temp_out = os.path.join(TEMP_DIR, f"{base_name}_watermarked{out_ext}")
                
                # Simpan frame pertama sebagai PNG untuk deteksi (tanpa kompresi)
                self.first_frame_path = os.path.join(TEMP_DIR, f"{base_name}_first_frame.png")
                self._process_video(self.file_path, temp_out, self.first_frame_path)
            else:
                # image
                temp_out = os.path.join(TEMP_DIR, f"{base_name}_watermarked.png")
                self._process_image(self.file_path, temp_out)

            self.last_temp_path = temp_out

            # processing complete: ask user where to save final copy
            self.root.after(0, lambda: self._on_processing_done(temp_out))
        except Exception as e:
            self.root.after(0, lambda: self._processing_failed(e))

    def _processing_failed(self, exc):
        self.apply_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error")
        self.progress['value'] = 0
        messagebox.showerror("Processing failed", str(exc))

    def _on_processing_done(self, temp_out_path):
        self.apply_btn.config(state=tk.NORMAL)
        self.save_again_btn.config(state=tk.NORMAL)
        self.progress['value'] = 100
        self.status_label.config(text="Done")
        # Update preview to show processed file (image or first frame of video)
        try:
            if self.is_video:
                # show first frame of processed video
                cap = cv2.VideoCapture(temp_out_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self._show_preview_image(pil)
            else:
                pil = Image.open(temp_out_path)
                self._show_preview_image(pil)
        except Exception:
            pass

        # Prompt user to pick a folder to save the final output
        dest_folder = filedialog.askdirectory(title="Choose folder to save final processed file (or Cancel to skip)")
        if dest_folder:
            try:
                self._copy_temp_to_folder(temp_out_path, dest_folder)
                self.last_user_saved_dir = dest_folder
                messagebox.showinfo("Saved", f"Saved processed file to:\n{os.path.join(dest_folder, os.path.basename(temp_out_path))}")
            except Exception as e:
                messagebox.showerror("Save error", f"Failed to save to selected folder: {e}")
        else:
            # user cancelled - keep only temp copy
            messagebox.showinfo("Completed", f"Processing complete. A copy is stored in '{TEMP_DIR}'. Use 'Save Again' to copy it elsewhere.")
        self.status_label.config(text="Ready")

    def _copy_temp_to_folder(self, temp_path, folder):
        os.makedirs(folder, exist_ok=True)
        base = os.path.basename(temp_path)
        dest = os.path.join(folder, base)
        # if exists, auto-increment
        if os.path.exists(dest):
            base_name, ext = os.path.splitext(base)
            counter = 1
            while True:
                dest = os.path.join(folder, f"{base_name}_{counter}{ext}")
                if not os.path.exists(dest):
                    break
                counter += 1
        shutil.copy2(temp_path, dest)
        return dest

    def save_again(self):
        if not self.last_temp_path or not os.path.exists(self.last_temp_path):
            messagebox.showwarning("No file", "No processed temp file found. Process a file first.")
            return
        folder = filedialog.askdirectory(title="Choose destination folder to copy the processed file into")
        if not folder:
            return
        try:
            dest = self._copy_temp_to_folder(self.last_temp_path, folder)
            messagebox.showinfo("Saved", f"Copied to: {dest}")
            self.last_user_saved_dir = folder
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy: {e}")

    # ---------- Image watermarking ----------
    def _process_image(self, source_path, out_path):
        pil = Image.open(source_path).convert("RGBA")
        if self.mode_var.get() == "visible":
            result = self._apply_visible_tiled(pil)
        else:
            result = self._apply_invisible_lsb(pil)
        # save result (PNG ensures no quality loss; for visible we still save PNG)
        result.save(out_path)
        # update progress
        self.progress['value'] = 100

    def _apply_visible_tiled(self, pil_image):
        img = pil_image.convert("RGBA")
        width, height = img.size

        overlay = Image.new("RGBA", img.size, (255,255,255,0))
        draw = ImageDraw.Draw(overlay)

        # font sizing
        try:
            font_size = max(20, min(width, height) // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
            # estimate height using default font
            font_size = 12

        text = self.watermark_text_var.get()
        # aim for ~15-20 watermarks across the page; set grid dimension to 5x4 = 20
        cols = 5
        rows = 4
        # compute spacing
        x_step = max(1, width // cols)
        y_step = max(1, height // rows)

        # use semi-transparent black text with small white translucent box behind for readability
        for r in range(rows):
            for c in range(cols):
                x = c * x_step + (x_step // 10)
                y = r * y_step + (y_step // 10)
                # measure text bounding box
                bbox = draw.textbbox((0,0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                pad = max(4, font_size // 8)
                # background rectangle translucent
                rect = (x - pad, y - pad, x + tw + pad, y + th + pad)
                draw.rectangle(rect, fill=(255,255,255,120))
                # draw text
                draw.text((x, y), text, font=font, fill=(0,0,0,160))

        combined = Image.alpha_composite(img, overlay)
        return combined.convert("RGBA")

    def _apply_invisible_lsb(self, pil_image):
        # Convert to RGB array (no alpha) for simpler LSB handling
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8)
        flat = arr.flatten()

        watermark = self.watermark_text_var.get()
        # convert to binary with null terminator
        bits = "".join(format(ord(c), "08b") for c in watermark) + "00000000"
        # ensure we don't exceed capacity
        capacity = flat.size
        if len(bits) > capacity:
            # If too long, truncate bits to fit
            bits = bits[:capacity]

        # encode bits into LSBs
        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | int(bit)

        out_arr = flat.reshape(arr.shape)
        out_img = Image.fromarray(out_arr.astype(np.uint8), mode="RGB")
        return out_img

    # ---------- Video watermarking ----------
    def _process_video(self, source_path, out_path, first_frame_path):
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise RuntimeError("Failed opening video for processing.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Gunakan codec biasa untuk video
        _, ext = os.path.splitext(out_path)
        ext = ext.lower()
        if ext in (".mp4", ".m4v"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Failed to open video writer. Check codec availability.")

        frame_idx = 0
        update_every = max(1, total_frames // 100) if total_frames else 30

        # precompute binary watermark for invisible mode
        watermark_bits = None
        if self.mode_var.get() == "invisible":
            watermark = self.watermark_text_var.get()
            watermark_bits = "".join(format(ord(c), "08b") for c in watermark) + "00000000"

        first_frame_processed = None

        while True:
            if self.stop_processing.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break

            # frame is BGR; convert to PIL via RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            if self.mode_var.get() == "visible":
                pil_water = self._apply_visible_tiled(pil)
            else:
                pil_water = self._apply_invisible_lsb_frame(pil, watermark_bits)

            # Simpan frame pertama yang sudah diproses untuk deteksi
            if frame_idx == 0 and self.mode_var.get() == "invisible":
                first_frame_processed = pil_water

            # convert back to BGR for writing
            arr = np.array(pil_water.convert("RGB"))
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            out.write(bgr)

            frame_idx += 1
            if total_frames:
                percent = int((frame_idx / total_frames) * 100)
                self.progress['value'] = percent
                if frame_idx % max(1, update_every) == 0:
                    self.root.update_idletasks()
            else:
                self.progress['value'] = (self.progress['value'] + 1) % 100

        cap.release()
        out.release()
        
        # Simpan frame pertama sebagai PNG untuk deteksi
        if first_frame_processed and self.mode_var.get() == "invisible":
            first_frame_processed.save(first_frame_path, "PNG")
            print(f"First frame saved for detection: {first_frame_path}")

        self.progress['value'] = 100

    def _apply_invisible_lsb_frame(self, pil_image, watermark_bits):
        """Embed watermark_bits (string of '0'/'1') into the LSBs of the frame's pixel bytes."""
        img_rgb = pil_image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8)
        flat = arr.flatten()
        if watermark_bits is None:
            return pil_image
        bits = watermark_bits
        cap = flat.size
        if len(bits) > cap:
            bits = bits[:cap]
        # encode
        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | int(bit)
        out = flat.reshape(arr.shape)
        return Image.fromarray(out.astype(np.uint8), mode="RGB")

    # ---------- preview resizer poll ----------
    def _start_preview_resizer(self):
        self._preview_resizer_count = 10
        self._preview_resizer_loop()

    def _preview_resizer_loop(self):
        if self._preview_resizer_count <= 0:
            return
        self._preview_resizer_count -= 1
        if self.preview_image:
            try:
                self._show_preview_image(self.preview_image)
            except Exception:
                pass
        self.root.after(250, self._preview_resizer_loop)

# ---------- launch ----------
def main():
    root = tk.Tk()
    app = WatermarkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()