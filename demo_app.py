"""
Interactive AOI Visual Inspection Demo (Tkinter + PIL GUI)
Bypasses opencv-python-headless GUI limitations using Python's standard Tkinter.
Zero additional dependencies required.
"""

import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageTk


class InspectionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Engine running on device: {self.device}")
        
        # Load local ResNet18 backbone
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.to(self.device)
        self.model.eval()

    def inspect(self, bgr_image: np.ndarray):
        start = time.perf_counter()

        # 1. CLAHE (Lighting normalization)
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        # 2. Defect detection (Edge + Contour area filter)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = enhanced_bgr.copy()
        defect_found = False
        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 3000:  # Scratch size criteria
                x, y, cw, ch = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
                cv2.putText(annotated, "DEFECT", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                defect_found = True

        # 3. Model Inference (safe torch.inference_mode)
        resized = cv2.resize(enhanced_bgr, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)
        with torch.inference_mode():
            _ = self.model(tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000
        status_text = "DEFECT DETECTED" if defect_found else "PASS (NORMAL)"
        status_color = "#D32F2F" if defect_found else "#2E7D32"  # Red / Green

        # Prepare side-by-side display image
        combined_bgr = np.hstack([bgr_image, annotated])
        combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
        return combined_rgb, status_text, status_color, elapsed_ms


def generate_sample_images():
    # Normal workpiece
    normal = np.full((350, 350, 3), 190, dtype=np.uint8)
    cv2.circle(normal, (175, 175), 110, (140, 140, 140), -1)
    cv2.imwrite("sample_pass.png", normal)

    # Defect workpiece
    defect = normal.copy()
    cv2.line(defect, (130, 150), (210, 190), (30, 30, 30), 3)
    cv2.line(defect, (190, 210), (230, 220), (20, 20, 20), 2)
    cv2.imwrite("sample_defect.png", defect)


class InspectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial AOI Visual Inspection System")
        self.root.geometry("820x540")
        self.root.configure(bg="#F5F5F5")

        self.engine = InspectionEngine()

        # Header Title
        title_lbl = tk.Label(root, text="Industrial AOI Visual Inspection System (以西結自動化 AI軟體工程師展示)",
                             font=("Segoe UI", 13, "bold"), bg="#F5F5F5", fg="#333333")
        title_lbl.pack(pady=8)

        # Status Banner
        self.status_banner = tk.Label(root, text="READY", font=("Segoe UI", 12, "bold"),
                                      bg="#2E7D32", fg="white", height=2)
        self.status_banner.pack(fill=tk.X, padx=15)

        # Image Canvas (Side-by-side: Original vs Annotated)
        self.img_label = tk.Label(root, bg="#E0E0E0")
        self.img_label.pack(pady=10)

        # Control Panel Buttons
        btn_frame = tk.Frame(root, bg="#F5F5F5")
        btn_frame.pack(pady=5)

        btn_pass = tk.Button(btn_frame, text="[1] 正常品 検査 (PASS)", font=("Segoe UI", 10, "bold"),
                             bg="#4CAF50", fg="white", width=22, height=2, command=self.test_pass)
        btn_pass.grid(row=0, column=0, padx=10)

        btn_defect = tk.Button(btn_frame, text="[2] 不良品 検査 (DEFECT)", font=("Segoe UI", 10, "bold"),
                               bg="#F44336", fg="white", width=22, height=2, command=self.test_defect)
        btn_defect.grid(row=0, column=1, padx=10)

        # Keybindings
        self.root.bind("1", lambda e: self.test_pass())
        self.root.bind("2", lambda e: self.test_defect())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # Initial Run
        self.test_pass()

    def update_view(self, image_path):
        img_bgr = cv2.imread(image_path)
        combined_rgb, status_text, status_color, elapsed_ms = self.engine.inspect(img_bgr)

        # Update Banner
        self.status_banner.config(text=f"STATUS: {status_text} | Latency: {elapsed_ms:.1f} ms", bg=status_color)

        # Update Image in GUI
        pil_img = Image.fromarray(combined_rgb)
        pil_img_tk = ImageTk.PhotoImage(pil_img)
        self.img_label.configure(image=pil_img_tk)
        self.img_label.image = pil_img_tk

    def test_pass(self):
        self.update_view("sample_pass.png")

    def test_defect(self):
        self.update_view("sample_defect.png")


def main():
    generate_sample_images()
    root = tk.Tk()
    app = InspectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
