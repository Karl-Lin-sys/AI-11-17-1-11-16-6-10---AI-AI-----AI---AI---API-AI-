"""
Interactive AOI Visual Inspection Demo (OpenCV Native GUI)
Works immediately on Windows without complex web packages.
"""

import sys
import time
import numpy as np
import cv2
import torch
import torch.nn as nn

class InspectionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Engine running on device: {self.device}")
        
        # Load lightweight evaluation backbone
        self.model = torch.hub.load('pytorch/vision:v0.18.0', 'resnet18', weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.to(self.device)
        self.model.eval()

    def inspect(self, bgr_image: np.ndarray):
        start = time.perf_counter()
        h, w = bgr_image.shape[:2]

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        # 2. Defect simulation detection (Edge + Threshold for visual display)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw visual defect bounding boxes if any anomalous edge area exists
        annotated = enhanced_bgr.copy()
        defect_found = False
        for c in contours:
            area = cv2.contourArea(c)
            if 30 < area < 3000:  # Scratch/dust threshold
                x, y, cw, ch = cv2.boundingRect(c)
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
                cv2.putText(annotated, "DEFECT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                defect_found = True

        # 3. Model Inference (safe inference_mode)
        resized = cv2.resize(enhanced_bgr, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)
        with torch.inference_mode():
            _ = self.model(tensor)

        elapsed_ms = (time.perf_counter() - start) * 1000
        status_text = "STATUS: DEFECT DETECTED" if defect_found else "STATUS: PASS (NORMAL)"
        status_color = (0, 0, 255) if defect_found else (0, 200, 0)

        # Build side-by-side presentation canvas
        canvas = np.hstack([bgr_image, annotated])
        banner = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"{status_text} | Latency: {elapsed_ms:.1f}ms", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        final_view = np.vstack([banner, canvas])
        return final_view, status_text, elapsed_ms


def generate_sample_images():
    """Generates a normal sample and a defective sample for instant demo testing."""
    # Normal workpiece
    normal = np.full((400, 400, 3), 180, dtype=np.uint8)
    cv2.circle(normal, (200, 200), 120, (140, 140, 140), -1)
    cv2.imwrite("sample_pass.png", normal)

    # Defect workpiece (scratches on surface)
    defect = normal.copy()
    cv2.line(defect, (150, 170), (230, 210), (30, 30, 30), 3)
    cv2.line(defect, (210, 230), (250, 240), (20, 20, 20), 2)
    cv2.imwrite("sample_defect.png", defect)
    print("[+] Created sample_pass.png and sample_defect.png")


def main():
    generate_sample_images()
    engine = InspectionEngine()

    print("\n=======================================================")
    print(" Industrial Inspection Live Demo")
    print(" Press [1] to inspect PASS sample")
    print(" Press [2] to inspect DEFECT sample")
    print(" Press [Q] or [ESC] to Exit")
    print("=======================================================\n")

    current_mode = "sample_pass.png"
    img = cv2.imread(current_mode)
    res_view, status, lat = engine.inspect(img)

    cv2.namedWindow("Industrial AOI Inspection Demo", cv2.WINDOW_AUTOSIZE)

    while True:
        cv2.imshow("Industrial AOI Inspection Demo", res_view)
        key = cv2.waitKey(50) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key == ord('1'):
            img = cv2.imread("sample_pass.png")
            res_view, status, lat = engine.inspect(img)
            print(f"[Key 1] Tested PASS sample -> {status} ({lat:.1f}ms)")
        elif key == ord('2'):
            img = cv2.imread("sample_defect.png")
            res_view, status, lat = engine.inspect(img)
            print(f"[Key 2] Tested DEFECT sample -> {status} ({lat:.1f}ms)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
