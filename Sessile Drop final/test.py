
#!/usr/bin/env python3
"""
Improved Sessile Drop Analyzer
Features:
 - Upload image (file dialog)
 - Auto baseline / manual baseline with slider preview
 - Save result image (overlay)
 - Export CSV with angles and basic geometry
 - Includes timestamp and image path in sessile_result.csv
 - Modern two-column layout (controls on left, image on right)
 - Processing status indicator and basic error handling
"""

import sys
import os
import csv
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QSizePolicy, QScrollArea,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# ============================ Image / Geometry Utils ============================

def adaptive_drop_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        51, 2
    )
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask

def choose_droplet_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best, best_score = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        per = cv2.arcLength(c, True)
        circularity = 4.0 * np.pi * area / (per * per + 1e-6)
        score = area * circularity
        if score > best_score:
            best, best_score = c, score
    return best

def find_auto_baseline_from_contour(cnt):
    ys = cnt[:, 0, 1]
    return int(np.percentile(ys, 95))

def refine_contact_point(gray, point, search_radius=5):
    if point is None:
        return None
    x0, y0 = int(point[0]), int(point[1])
    h, w = gray.shape
    best = (x0, y0)
    best_g = 0
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            x, y = x0 + dx, y0 + dy
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
                gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
                g2 = gx * gx + gy * gy
                if g2 > best_g:
                    best_g = g2
                    best = (x, y)
    return best

def fit_circle_and_intersect(cnt, baseline_y, gray=None, refine=True):
    pts = cnt[:, 0, :].astype(np.float32)
    if len(pts) < 10:
        return None, None
    y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])
    cutoff = y_min + 0.70 * (y_max - y_min)
    upper = pts[pts[:, 1] < cutoff]
    if len(upper) < 10:
        upper = pts
    (cx, cy), r = cv2.minEnclosingCircle(upper)
    dy = baseline_y - cy
    if abs(dy) >= r:
        return None, None
    dx = np.sqrt(max(r * r - dy * dy, 0.0))
    left = (int(round(cx - dx)), int(baseline_y))
    right = (int(round(cx + dx)), int(baseline_y))
    if refine and gray is not None:
        left = refine_contact_point(gray, left, 5)
        right = refine_contact_point(gray, right, 5)
    return left, right

def local_poly_contact_angle(cnt, contact_pt, window_px=25):
    if contact_pt is None:
        return None
    cx, _ = contact_pt
    pts = cnt[:, 0, :]
    nb = pts[np.abs(pts[:, 0] - cx) < window_px]
    if len(nb) < 8:
        nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if len(nb) < 8:
        return None
    x = nb[:, 0].astype(np.float64)
    y = nb[:, 1].astype(np.float64)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return None
    m = 2.0 * a * cx + b
    theta = np.degrees(np.arctan(np.abs(m)))
    return float(np.clip(theta, 0.0, 175.0))

def apex_of_contour(cnt):
    if cnt is None or len(cnt) == 0:
        return None
    idx = np.argmin(cnt[:, 0, 1])
    return tuple(cnt[idx, 0, :])

def detect_auto_baseline(gray):
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.uint8(np.absolute(sobel_y))
    edges = cv2.GaussianBlur(edges, (9, 9), 0)
    h = gray.shape[0]
    bottom_half = edges[int(h * 0.5):, :]
    row_strength = np.mean(bottom_half, axis=1)
    y_rel = np.where(row_strength > np.percentile(row_strength, 95))[0]
    if len(y_rel) == 0:
        return int(h * 0.9)
    baseline_y = int(h * 0.5 + np.median(y_rel))
    return baseline_y

# ============================ Main Analyzer GUI ============================

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Contact Angle Analyzer")
        self.setGeometry(80, 80, 1200, 800)

        self.original_image = None
        self.current_pixmap = None
        self.overlay = None
        self.image_path = None
        self.last_path = os.path.expanduser("~")

        main_layout = QHBoxLayout(self)

        # Controls panel
        controls_box = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_box.setLayout(controls_layout)
        controls_box.setMaximumWidth(380)

        row = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_auto_baseline = QPushButton("Auto Baseline")
        self.btn_save = QPushButton("Save Result")
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_auto_baseline)
        row.addWidget(self.btn_save)
        controls_layout.addLayout(row)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Baseline:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        slider_row.addWidget(self.slider)
        controls_layout.addLayout(slider_row)

        self.status_label = QLabel("No image loaded.")
        controls_layout.addWidget(self.status_label)

        # Results section
        result_group = QGroupBox("Results")
        result_layout = QFormLayout()
        result_group.setLayout(result_layout)
        self.left_angle_label = QLabel("--")
        self.right_angle_label = QLabel("--")
        self.base_width_label = QLabel("-- px")
        self.drop_height_label = QLabel("-- px")
        self.apex_pos_label = QLabel("(-, -)")
        self.contour_area_label = QLabel("-- px^2")
        result_layout.addRow("Left Angle:", self.left_angle_label)
        result_layout.addRow("Right Angle:", self.right_angle_label)
        result_layout.addRow("Base Width:", self.base_width_label)
        result_layout.addRow("Drop Height:", self.drop_height_label)
        result_layout.addRow("Apex (x,y):", self.apex_pos_label)
        result_layout.addRow("Contour Area:", self.contour_area_label)
        controls_layout.addWidget(result_group)

        self.btn_export = QPushButton("Export to CSV")
        controls_layout.addWidget(self.btn_export)

        controls_layout.addStretch(1)
        main_layout.addWidget(controls_box)

        # Image viewer
        right_layout = QVBoxLayout()
        self.image_label = QLabel("Image will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid #bbb;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        clayout = QVBoxLayout(container)
        clayout.addWidget(self.image_label)
        scroll.setWidget(container)
        right_layout.addWidget(scroll)
        main_layout.addLayout(right_layout, stretch=1)

        # Styling
        self.setStyleSheet("""
            QWidget { background-color: #fafafa; font-family: 'Segoe UI'; font-size: 10pt; }
            QPushButton { background-color: #1976d2; color: white; border-radius: 6px; padding: 6px 10px; }
            QPushButton:disabled { background-color: #9fbce6; color: #eee; }
        """)

        # Signals
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_auto_baseline.clicked.connect(self.auto_detect_baseline)
        self.btn_save.clicked.connect(self.save_result)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)
        self.btn_export.clicked.connect(self.export_csv)

    # ---------- Core Methods ----------
    def set_status(self, text):
        self.status_label.setText(text)
        QApplication.processEvents()

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", self.last_path,
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        self.last_path = os.path.dirname(path)
        self.image_path = path
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Unable to load image.")
            return
        self.original_image = img
        self.run_full_analysis(initial=True)
        self.set_status(f"Loaded: {os.path.basename(path)}")

    def preview_baseline_only(self):
        if self.overlay is None:
            return
        y = self.slider.value()
        temp = self.overlay.copy()
        cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
        self.display_image(temp)

    def auto_detect_baseline(self):
        if self.original_image is None:
            return
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        baseline_y = detect_auto_baseline(gray)
        h = self.original_image.shape[0]
        baseline_y = int(np.clip(baseline_y, 0, h - 1))
        self.slider.blockSignals(True)
        self.slider.setRange(0, h - 1)
        self.slider.setValue(baseline_y)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)
        self.run_full_analysis()

    def run_full_analysis(self, initial=False):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = adaptive_drop_mask(img)
        cnt = choose_droplet_contour(mask)
        if cnt is None:
            self.set_status("No contour found.")
            return
        auto_y = find_auto_baseline_from_contour(cnt)
        h = img.shape[0]
        if initial or not self.slider.isEnabled():
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            self.slider.setValue(int(auto_y))
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
        baseline_y = int(self.slider.value())
        l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray)
        left_ang = local_poly_contact_angle(cnt, l_pt)
        right_ang = local_poly_contact_angle(cnt, r_pt)
        apx = apex_of_contour(cnt)
        area = int(cv2.contourArea(cnt))
        base_w = abs(r_pt[0] - l_pt[0]) if l_pt and r_pt else None
        height = int(baseline_y - apx[1]) if apx else None
        self.left_angle_label.setText(f"{left_ang:.2f}°" if left_ang else "--")
        self.right_angle_label.setText(f"{right_ang:.2f}°" if right_ang else "--")
        self.base_width_label.setText(f"{base_w} px" if base_w else "-- px")
        self.drop_height_label.setText(f"{height} px" if height else "-- px")
        self.apex_pos_label.setText(f"{apx}" if apx else "(-, -)")
        self.contour_area_label.setText(f"{area} px^2")

        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)
        self.overlay = vis
        self.display_image(vis)
        self.set_status("Processing complete.")

    def display_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.current_pixmap = pix

    def save_result(self):
        if self.overlay is None:
            QMessageBox.warning(self, "No result", "No processed image to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", self.last_path, "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if not path:
            return
        cv2.imwrite(path, self.overlay)
        QMessageBox.information(self, "Saved", f"Result saved to:\n{path}")

    # ---------------- Export CSV ----------------
    def export_csv(self):
        if self.overlay is None or self.original_image is None:
            QMessageBox.warning(self, "No data", "No analysis data to export.")
            return

        left = self.left_angle_label.text()
        right = self.right_angle_label.text()
        base_w = self.base_width_label.text()
        height = self.drop_height_label.text()
        apex = self.apex_pos_label.text()
        area = self.contour_area_label.text()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = self.image_path if self.image_path else "Unknown"
        csv_path = os.path.join(os.getcwd(), "sessile_result.csv")

        headers = [
            "Timestamp",
            "Image Path",
            "Left Angle (deg)",
            "Right Angle (deg)",
            "Base Width (px)",
            "Drop Height (px)",
            "Apex (x,y)",
            "Contour Area (px^2)"
        ]
        row = [timestamp, image_path, left, right, base_w, height, apex, area]

        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)
            QMessageBox.information(self, "Export CSV", f"Results appended to:\n{csv_path}")
            self.set_status(f"Appended to sessile_result.csv at {timestamp}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
            self.set_status("Failed to export CSV.")

# ================================== Main ==================================

def main():
    app = QApplication(sys.argv)
    window = SessileDropAnalyzer()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()