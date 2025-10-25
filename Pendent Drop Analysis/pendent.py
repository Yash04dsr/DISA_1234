import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QSizePolicy, QMessageBox, QFormLayout, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ============================ Image Processing Utilities ============================

def detect_drop_contour(bgr):
    """Detect pendant drop contour and needle location."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    # --- Detect top needle region ---
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.abs(sobel_y), axis=1)
    needle_row = int(np.argmax(edge_strength[:gray.shape[0] // 3]))

    # --- Define ROI below the needle ---
    roi_top = max(needle_row + 5, 0)
    roi = gray[roi_top:, :]

    otsu_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(5, int(0.5 * otsu_val))
    high = max(10, int(1.5 * otsu_val))

    edges = cv2.Canny(roi, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, needle_row, roi_top

    # --- Select best contour ---
    h, w = roi.shape[:2]
    baseline_y = int(0.95 * h)
    best, best_score = None, -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        ys = c[:, 0, 1]
        xs = c[:, 0, 0]
        frac_above = np.mean(ys < baseline_y)
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per + 1e-6)
        score = area * circ * frac_above
        if score > best_score:
            best, best_score = c, score

    if best is not None:
        best[:, 0, 1] += roi_top
    return best, needle_row, roi_top
# def detect_drop_contour(bgr):
#     """Contour detection robust to background and reflection noise."""
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)

#     # Invert image if droplet is darker than background
#     if np.mean(gray[:50, :]) > np.mean(gray[-50:, :]):
#         gray = cv2.bitwise_not(gray)

#     # --- Adaptive + Otsu combo threshold ---
#     _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                   cv2.THRESH_BINARY, 51, 2)
#     combined = cv2.bitwise_and(otsu, adapt)

#     # --- Morphology cleanup ---
#     kernel = np.ones((5, 5), np.uint8)
#     clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
#     clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

#     # --- Invert if needed (make droplet white) ---
#     white_ratio = np.sum(clean == 255) / clean.size
#     if white_ratio < 0.5:
#         clean = cv2.bitwise_not(clean)

#     # --- Contour extraction ---
#     contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, 0, 0

#     h, w = gray.shape
#     frame_margin = 10

#     best, best_score = None, -1
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < 300:
#             continue
#         x, y, cw, ch = cv2.boundingRect(c)
#         if x <= frame_margin or y <= frame_margin or x + cw >= w - frame_margin:
#             continue  # ignore touching edges (background)
#         ys = c[:, 0, 1]
#         y_mean = np.mean(ys)
#         per = cv2.arcLength(c, True)
#         circ = 4 * np.pi * area / (per * per + 1e-6)
#         # prefer large, compact, low droplet shapes
#         score = area * circ * (y_mean / h)
#         if score > best_score:
#             best, best_score = c, score

#     return best, 0, 0


# ============================ GUI Application ============================

class PendantDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pendant Drop Surface Tension Analyzer (Hansen–Rødsrud Corrected)")
        self.setGeometry(100, 100, 1000, 900)

        self.img = None
        self.current_pixmap = None

        layout = QVBoxLayout(self)
        self.image_label = QLabel("Load a pendant drop image.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #888;")
        self.image_label.setMinimumSize(720, 460)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # --- Input form for user parameters ---
        form_layout = QFormLayout()
        self.needle_diam_input = QLineEdit("1.65")  # mm, default 16G needle
        self.density_input = QLineEdit("997")       # kg/m³, water
        form_layout.addRow("<b>Needle Diameter (mm):</b>", self.needle_diam_input)
        form_layout.addRow("<b>Liquid Density (kg/m³):</b>", self.density_input)
        layout.addLayout(form_layout)

        # --- Controls ---
        controls = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_analyze = QPushButton("Analyze")
        controls.addWidget(self.btn_load)
        controls.addWidget(self.btn_analyze)
        layout.addLayout(controls)

        self.result_label = QLabel("Results will appear here.")
        self.result_label.setStyleSheet("font-weight:bold; font-size:16px;")
        layout.addWidget(self.result_label)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_analyze.clicked.connect(self.run_analysis)

    # ================== Image Handling ==================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", "Could not load image.")
            return
        h, w = img.shape[:2]
        if h > 700:
            img = cv2.resize(img, (int(w * 700 / h), 700), interpolation=cv2.INTER_AREA)
        self.img = img
        self.display_image(img)
        self.result_label.setText("Results will appear here.")

    def display_image(self, img_bgr):
        """Display CV2 image in QLabel safely."""
        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(
            self.current_pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        """Redraw image without zoom distortion."""
        super().resizeEvent(event)
        if self.current_pixmap:
            self.image_label.setPixmap(
                self.current_pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    # ================== Analysis Logic ==================
    def run_analysis(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return

        try:
            needle_diam_mm = float(self.needle_diam_input.text())
            density = float(self.density_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input values.")
            return

        cnt, needle_row, roi_top = detect_drop_contour(self.img)
        if cnt is None:
            QMessageBox.warning(self, "Error", "Could not detect droplet contour.")
            return

        # Compute bounding box and top zone (needle)
        x, y, w, h = cv2.boundingRect(cnt)
        top_y_cutoff = y + int(0.1 * h)
        needle_pts = [pt[0] for pt in cnt if y < pt[0][1] < top_y_cutoff]
        if not needle_pts:
            QMessageBox.warning(self, "Error", "Needle region not detected.")
            return
        needle_px = max(p[0] for p in needle_pts) - min(p[0] for p in needle_pts)
        if needle_px <= 0:
            QMessageBox.warning(self, "Error", "Needle width measurement invalid.")
            return

        mm_per_px = needle_diam_mm / needle_px

        # --- Find apex (lowest point) ---
        apex = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # --- Find equatorial diameter (D_E) ---
        de_px, de_y = 0, 0
        for y_scan in range(y, y + h):
            pts_y = [pt[0] for pt in cnt if pt[0][1] == y_scan]
            if len(pts_y) > 1:
                width = max(p[0] for p in pts_y) - min(p[0] for p in pts_y)
                if width > de_px:
                    de_px = width
                    de_y = y_scan

        if de_px == 0:
            QMessageBox.warning(self, "Error", "Failed to measure equatorial diameter (D_E).")
            return

        # --- Find D_S at one D_E above apex ---
        ds_y = int(apex[1] - de_px)
        pts_ds = [pt[0] for pt in cnt if abs(pt[0][1] - ds_y) <= 1]
        if len(pts_ds) < 2:
            QMessageBox.warning(self, "Error", "Failed to measure D_S (drop too small).")
            return
        ds_px = max(p[0] for p in pts_ds) - min(p[0] for p in pts_ds)

        # --- Convert to meters ---
        de_m = de_px * mm_per_px / 1000
        ds_m = ds_px * mm_per_px / 1000

        # --- Shape factor method (Hansen & Rødsrud, 1991) ---
        S = ds_m / de_m
        if not (0.7 < S < 1.0):
            QMessageBox.warning(self, "Warning",
                                f"Shape factor S={S:.3f} outside 0.7–1.0 → less accurate.")

        one_over_H = (0.9987 + 0.9037 * S + 1.9837 * (S ** 2) - 0.7754 * (S ** 3)
                      + 4.2384 * (S ** 4) - 3.7937 * (S ** 5))
        g = 9.80665
        gamma = (density * g * (de_m ** 2)) / one_over_H
        gamma_mNm = gamma * 1000

        # --- Visualization ---
        vis = self.img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        de_mid_x = int(np.mean([p[0] for p in cnt if p[0][1] == de_y]))
        cv2.line(vis, (de_mid_x - de_px // 2, de_y),
                 (de_mid_x + de_px // 2, de_y), (255, 0, 0), 2)
        cv2.putText(vis, "D_E", (de_mid_x + de_px // 2 + 10, de_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        ds_mid_x = int(np.mean([p[0] for p in pts_ds]))
        cv2.line(vis, (ds_mid_x - ds_px // 2, ds_y),
                 (ds_mid_x + ds_px // 2, ds_y), (0, 0, 255), 2)
        cv2.putText(vis, "D_S", (ds_mid_x + ds_px // 2 + 10, ds_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.circle(vis, apex, 6, (255, 255, 0), -1)
        cv2.putText(vis, "Apex", (apex[0] + 10, apex[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Display results ---
        self.result_label.setText(
            f"D_E = {de_px * mm_per_px:.2f} mm, "
            f"D_S = {ds_px * mm_per_px:.2f} mm, "
            f"S = {S:.3f}, γ ≈ {gamma_mNm:.2f} mN/m"
        )
        self.display_image(vis)


# ================================== Main ==================================

if __name__ == "__main__":
    import os, PyQt5.QtCore
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt5.QtCore.__file__), "Qt", "plugins", "platforms"
    )
    app = QApplication(sys.argv)
    w = PendantDropAnalyzer()
    w.show()
    sys.exit(app.exec_())
