#!/usr/bin/env python3
"""
Sessile Drop Goniometer (Improved)
 - Displays contact angles on image
 - Optional angle arcs (toggle)
 - Auto-export to sessile_result.csv (toggle)
 - Upload, Auto-baseline, Save result, Export CSV
"""

import sys
import os
import csv
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QSizePolicy, QScrollArea, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

# ---------------- Image / geometry utils ----------------

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

def local_poly_contact_angle_and_slope(cnt, contact_pt, window_px=25):
    """
    Returns (theta_deg, slope_m) where theta_deg is contact angle between tangent and horizontal.
    slope_m is the fitted polynomial derivative at contact x (can be +/-, useful for arc orientation).
    """
    if contact_pt is None:
        return None, None
    cx = contact_pt[0]
    pts = cnt[:, 0, :]
    nb = pts[np.abs(pts[:, 0] - cx) < window_px]
    if len(nb) < 8:
        nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if len(nb) < 8:
        return None, None
    x = nb[:, 0].astype(np.float64)
    y = nb[:, 1].astype(np.float64)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return None, None
    m = 2.0 * a * cx + b  # derivative dy/dx at cx
    theta = np.degrees(np.arctan(np.abs(m)))
    return float(np.clip(theta, 0.0, 175.0)), float(m)

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

# ---------------- Utility to draw angle arcs ----------------

def draw_angle_arc(img, center, tangent_slope, angle_deg, side='left', radius=30, color=(0, 0, 255), thickness=2):
    """
    Draw a small visual arc near 'center' representing the angle between baseline (horizontal)
    and the tangent. We approximate using cv2.ellipse by selecting start/end angles based on the slope.
    side: 'left' or 'right' controls which quadrant the arc sits in.
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    # angle of tangent in degrees (arctan(slope)) relative to +x
    tan_angle = np.degrees(np.arctan(tangent_slope))
    # We want the arc between horizontal (0 deg) and tangent angle, absolute
    # For left contact: arc typically above the baseline and to the left -> we draw from 180 to 180 - angle
    # For right contact: from 0 to angle
    r = max(12, int(radius))
    if side == 'left':
        start_angle = 180
        end_angle = 180 - int(round(angle_deg))
    else:
        start_angle = 0
        end_angle = int(round(angle_deg))
    # OpenCV uses angles in degrees measured clockwise from the x-axis.
    # Draw ellipse with small axes (r,r)
    try:
        cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    except Exception:
        pass

# ---------------- Main GUI ----------------

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Goniometer")
        self.setGeometry(60, 60, 1200, 820)

        # state
        self.original_image = None
        self.overlay = None
        self.current_pixmap = None
        self.image_path = None
        self.last_path = os.path.expanduser("~")

        # Main layout: controls left, image right
        main_layout = QHBoxLayout(self)

        # Controls
        controls_box = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_box.setLayout(controls_layout)
        controls_box.setMaximumWidth(380)

        # Buttons row
        row = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_auto = QPushButton("Auto Baseline")
        self.btn_save = QPushButton("Save Result")
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_auto)
        row.addWidget(self.btn_save)
        controls_layout.addLayout(row)

        # Slider row
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Baseline:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        slider_row.addWidget(self.slider)
        controls_layout.addLayout(slider_row)

        # Checkboxes
        self.chk_arcs = QCheckBox("Show angle arcs")
        self.chk_arcs.setChecked(True)
        self.chk_auto_export = QCheckBox("Auto export to sessile_result.csv")
        self.chk_auto_export.setChecked(True)
        controls_layout.addWidget(self.chk_arcs)
        controls_layout.addWidget(self.chk_auto_export)

        # Status
        self.status_label = QLabel("No image loaded.")
        controls_layout.addWidget(self.status_label)

        # Results info
        result_group = QGroupBox("Results")
        result_layout = QFormLayout()
        result_group.setLayout(result_layout)
        self.left_angle_label = QLabel("--")
        self.right_angle_label = QLabel("--")
        self.base_width_label = QLabel("-- px")
        self.drop_height_label = QLabel("-- px")
        self.apex_label = QLabel("(-, -)")
        self.area_label = QLabel("-- px^2")
        result_layout.addRow("Left Angle:", self.left_angle_label)
        result_layout.addRow("Right Angle:", self.right_angle_label)
        result_layout.addRow("Base Width:", self.base_width_label)
        result_layout.addRow("Drop Height:", self.drop_height_label)
        result_layout.addRow("Apex (x,y):", self.apex_label)
        result_layout.addRow("Contour Area:", self.area_label)
        controls_layout.addWidget(result_group)

        # Export button
        self.btn_export = QPushButton("Export CSV now")
        controls_layout.addWidget(self.btn_export)

        controls_layout.addStretch(1)
        main_layout.addWidget(controls_box)

        # Image area (right)
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

        # Footer (image path + timestamp)
        self.footer = QLabel("Ready.")
        font = QFont()
        font.setPointSize(9)
        self.footer.setFont(font)
        right_layout.addWidget(self.footer)

        # Styling
        self.setStyleSheet("""
            QWidget { background-color: #fafafa; font-family: 'Segoe UI'; font-size: 10pt; }
            QPushButton { background-color: #1976d2; color: white; border-radius: 6px; padding: 6px 10px; }
            QGroupBox { font-weight: bold; }
        """)

        # Signals
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_auto.clicked.connect(self.auto_baseline)
        self.btn_save.clicked.connect(self.save_result)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)
        self.btn_export.clicked.connect(self.export_csv)

        # Try auto-load default
        self.auto_load_default()

    def set_status(self, txt):
        self.status_label.setText(txt)
        QApplication.processEvents()

    def auto_load_default(self):
        p = os.path.expanduser("~/sessile.png")
        if os.path.exists(p):
            self.load_image(p)

    def load_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", self.last_path,
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        self.last_path = os.path.dirname(path)
        self.image_path = path
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Unable to read image.")
            return
        self.original_image = img
        self.run_full_analysis(initial=True)
        self.set_status(f"Loaded: {os.path.basename(path)}")
        self.update_footer()

    def preview_baseline_only(self):
        if self.overlay is None:
            return
        y = int(self.slider.value())
        temp = self.overlay.copy()
        cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
        self.display_image(temp)

    def auto_baseline(self):
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
        try:
            self.set_status("Processing...")
            QApplication.processEvents()

            img = self.original_image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = adaptive_drop_mask(img)
            cnt = choose_droplet_contour(mask)
            if cnt is None:
                self.set_status("No contour detected.")
                self.overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                self.display_image(self.overlay)
                return

            auto_y = find_auto_baseline_from_contour(cnt)
            if initial or not self.slider.isEnabled():
                h = img.shape[0]
                self.slider.blockSignals(True)
                self.slider.setRange(0, h - 1)
                self.slider.setValue(int(np.clip(auto_y, 0, h - 1)))
                self.slider.setEnabled(True)
                self.slider.blockSignals(False)
            baseline_y = int(self.slider.value())

            # contact points
            l_pt, r_pt = fit_circle_and_intersect(cnt, baseline_y, gray, refine=True)
            # fallback if None
            if (l_pt is None) or (r_pt is None):
                pts = cnt[:, 0, :]
                xs = pts[:, 0]
                medx = np.median(xs)
                left_half = pts[xs <= medx]
                right_half = pts[xs >= medx]
                if len(left_half):
                    l_pt = tuple(left_half[np.argmax(left_half[:, 1])])
                if len(right_half):
                    r_pt = tuple(right_half[np.argmax(right_half[:, 1])])
                l_pt = refine_contact_point(gray, l_pt, 5) if l_pt is not None else None
                r_pt = refine_contact_point(gray, r_pt, 5) if r_pt is not None else None

            left_ang, left_slope = local_poly_contact_angle_and_slope(cnt, l_pt, window_px=25)
            right_ang, right_slope = local_poly_contact_angle_and_slope(cnt, r_pt, window_px=25)

            apx = apex_of_contour(cnt)
            contour_area = int(cv2.contourArea(cnt))
            base_w = int(abs(r_pt[0] - l_pt[0])) if (l_pt is not None and r_pt is not None) else None
            drop_h = int(baseline_y - apx[1]) if apx is not None else None

            # Update result labels (formatted)
            self.left_angle_label.setText(f"{left_ang:.2f}°" if left_ang is not None else "--")
            self.right_angle_label.setText(f"{right_ang:.2f}°" if right_ang is not None else "--")
            self.base_width_label.setText(f"{base_w} px" if base_w is not None else "-- px")
            self.drop_height_label.setText(f"{drop_h} px" if drop_h is not None else "-- px")
            self.apex_label.setText(f"{int(apx[0])}, {int(apx[1])}" if apx is not None else "(-, -)")
            self.area_label.setText(f"{contour_area} px^2")

            # Visualization
            vis = img.copy()
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)  # baseline blue
            if apx is not None:
                cv2.circle(vis, (int(apx[0]), int(apx[1])), 6, (0, 0, 255), -1)  # apex red

            # Base width line
            if l_pt is not None and r_pt is not None:
                cv2.line(vis, (int(l_pt[0]), int(l_pt[1])), (int(r_pt[0]), int(r_pt[1])), (255, 255, 0), 2)

            # Draw contact points and angle labels (on-image)
            def draw_contact_and_label(img, pt, angle, slope, side_name):
                if pt is None:
                    return
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
                # label background box for readability
                label = f"θ{side_name} = {angle:.1f}°" if angle is not None else f"θ{side_name} = --"
                # text size
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                bx = px + 10 if px + 10 + tw < img.shape[1] else px - 10 - tw
                by = py - 10 if py - 10 - th > 0 else py + 10 + th
                cv2.rectangle(img, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (255, 255, 255), -1)
                
                # optional small arc
                if self.chk_arcs.isChecked() and angle is not None and slope is not None:
                    side = 'left' if side_name == 'L' else 'right'
                    # radius proportional to image size
                    r = max(16, int(min(img.shape[0], img.shape[1]) * 0.02))
                    draw_angle_arc(img, (px, py), slope, angle, side=side, radius=r, color=(0, 0, 255), thickness=2)

            draw_contact_and_label(vis, l_pt, left_ang, left_slope, 'L')
            draw_contact_and_label(vis, r_pt, right_ang, right_slope, 'R')

            self.overlay = vis
            self.display_image(vis)
            self.set_status("Processing complete.")
            self.update_footer()

            # Auto-export if enabled
            if self.chk_auto_export.isChecked():
                self.export_csv(auto=True, left=left_ang, right=right_ang, baseline=baseline_y,
                                apex=apx, base_width=base_w, height_px=drop_h, area_px=contour_area)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.set_status("Error during processing.")

    def display_image(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        self.set_status(f"Saved result to {os.path.basename(path)}")

    def update_footer(self):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        imgname = os.path.basename(self.image_path) if self.image_path else "N/A"
        self.footer.setText(f"Last analysis: {ts}    Image: {imgname}")

    def export_csv(self, auto=False, left=None, right=None, baseline=None, apex=None, base_width=None, height_px=None, area_px=None):
        """
        If called from auto-export, pass the computed values to avoid re-parsing labels.
        Otherwise, collect values from the UI labels.
        """
        if self.overlay is None or self.original_image is None:
            QMessageBox.warning(self, "No data", "No analysis data to export.")
            return

        # Use provided values if auto call; else parse from labels
        if auto:
            left_val = f"{left:.2f}" if left is not None else ""
            right_val = f"{right:.2f}" if right is not None else ""
            base_w = f"{base_width}" if base_width is not None else ""
            height_v = f"{height_px}" if height_px is not None else ""
            apex_s = f"{int(apex[0])},{int(apex[1])}" if apex is not None else ""
            area_s = f"{area_px}" if area_px is not None else ""
            baseline_y = f"{baseline}" if baseline is not None else ""
        else:
            # read from labels
            left_val = self.left_angle_label.text().replace("°", "").strip()
            right_val = self.right_angle_label.text().replace("°", "").strip()
            base_w = self.base_width_label.text().replace(" px", "").strip()
            height_v = self.drop_height_label.text().replace(" px", "").strip()
            apex_s = self.apex_label.text().strip()
            area_s = self.area_label.text().replace(" px^2", "").strip()
            baseline_y = str(self.slider.value()) if self.slider.isEnabled() else ""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_path = self.image_path if self.image_path else "Unknown"

        csv_path = os.path.join(os.getcwd(), "sessile_result.csv")
        headers = [
            "Timestamp", "Image Path", "Left Angle (deg)", "Right Angle (deg)",
            "Baseline (y)", "Base Width (px)", "Drop Height (px)",
            "Apex (x,y)", "Contour Area (px^2)"
        ]
        row = [timestamp, image_path, left_val, right_val, baseline_y, base_w, height_v, apex_s, area_s]

        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)
            if not auto:
                QMessageBox.information(self, "Export CSV", f"Results appended to:\n{csv_path}")
            self.set_status(f"Appended to sessile_result.csv at {timestamp}")
            self.update_footer()
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
            self.set_status("Failed to export CSV.")

# ---------------- Main ----------------

def main():
    app = QApplication(sys.argv)
    win = SessileDropAnalyzer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()