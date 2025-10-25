import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QSizePolicy, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ============================ Image Processing Utilities ============================

def detect_drop_contour(bgr):
    """Robust color-aware contour detection for pendant drops."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s = cv2.GaussianBlur(s, (7, 7), 0)
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)

    otsu_val, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(5, int(0.5 * otsu_val))
    high = max(10, int(1.5 * otsu_val))

    edges = cv2.Canny(s, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0

    best, best_score = None, -1
    h, w = s.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        ys = c[:, 0, 1]
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per + 1e-6)
        frac_in_frame = np.mean(ys < h * 0.95)
        score = area * circ * frac_in_frame
        if score > best_score:
            best, best_score = c, score
    return best, 0, 0


def measure_drop_diameters(cnt):
    """Measure equatorial (D_E) and D_S (20% above equator) diameters."""
    if cnt is None:
        return None, None, None, None

    ys = cnt[:, 0, 1]
    xs = cnt[:, 0, 0]
    y_min, y_max = np.min(ys), np.max(ys)
    height = y_max - y_min

    # Compute width profile along height
    widths = []
    for y in range(int(y_min), int(y_max), 2):
        band = cnt[(ys >= y - 1) & (ys <= y + 1), 0, :]
        if len(band) > 1:
            width = np.max(band[:, 0]) - np.min(band[:, 0])
            widths.append((y, width))
    if not widths:
        return None, None, None, None

    # y where width is maximum -> equator
    y_eq, D_E = max(widths, key=lambda t: t[1])

    # y 20% above equator
    y_ds = int(y_eq - 0.2 * height)
    band = cnt[(ys >= y_ds - 1) & (ys <= y_ds + 1), 0, :]
    if len(band) > 1:
        D_S = np.max(band[:, 0]) - np.min(band[:, 0])
    else:
        D_S = None

    return D_E, D_S, y_eq, y_ds


def calculate_surface_tension(D_E, D_S, density, scale_mm_per_px):
    """Estimate surface tension using shape factor approximation."""
    if None in [D_E, D_S] or D_E == 0:
        return None
    g = 9.81
    D_E_mm = D_E * scale_mm_per_px
    D_S_mm = D_S * scale_mm_per_px
    ratio = D_S_mm / D_E_mm
    shape_factor = 0.5 + 1.5 * ratio
    gamma = density * g * (D_E_mm ** 2) / (shape_factor * 1000)  # mN/m
    return gamma


# ============================ GUI Application ============================

class PendantDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pendant Drop Analyzer (Automatic Equator Detection)")
        self.setGeometry(100, 100, 1150, 830)

        self.img = None
        self.current_pixmap = None
        self.overlay = None

        layout = QVBoxLayout(self)
        self.image_label = QLabel("Load a pendant drop image.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #888;")
        self.image_label.setMinimumSize(720, 460)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.btn_load = QPushButton("Load Image")
        self.btn_analyze = QPushButton("Analyze")
        controls.addWidget(self.btn_load)
        controls.addWidget(self.btn_analyze)

        self.result_label = QLabel("Results will appear here.")
        layout.addWidget(self.result_label)

        # Signals
        self.btn_load.clicked.connect(self.load_image)
        self.btn_analyze.clicked.connect(self.run_analysis)

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
        if w > 1400:
            img = cv2.resize(img, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
        self.img = img
        self.display_image(img)

    def run_analysis(self):
        if self.img is None:
            QMessageBox.warning(self, "Error", "No image loaded.")
            return
        try:
            cnt, _, _ = detect_drop_contour(self.img)
            if cnt is None:
                QMessageBox.warning(self, "Error", "No droplet detected.")
                return

            D_E, D_S, y_eq, y_ds = measure_drop_diameters(cnt)
            if D_E is None:
                QMessageBox.warning(self, "Error", "Could not measure diameters.")
                return

            # Calibration input
            needle_mm, ok = QInputDialog.getDouble(
                self, "Needle Diameter (mm)", "Enter actual needle diameter (mm):", 0.5, 0.1, 10, 2)
            if not ok:
                return
            needle_px, ok = QInputDialog.getDouble(
                self, "Needle Pixel Span", "Enter apparent needle width in pixels:", 50, 1, 1000, 1)
            if not ok:
                return
            scale_mm_per_px = needle_mm / needle_px

            density = 998  # water (kg/m³)
            gamma = calculate_surface_tension(D_E, D_S, density, scale_mm_per_px)

            D_E_mm = D_E * scale_mm_per_px
            D_S_mm = D_S * scale_mm_per_px
            text = f"D_E={D_E_mm:.2f} mm, D_S={D_S_mm:.2f} mm, γ≈{gamma:.2f} mN/m"
            self.result_label.setText(text)

            # Visualization
            vis = self.img.copy()
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

            xs = cnt[:, 0, 0]
            x_min, x_max = np.min(xs), np.max(xs)

            # Draw D_E and D_S lines at detected heights
            cv2.line(vis, (int(x_min), int(y_eq)), (int(x_max), int(y_eq)), (255, 0, 0), 2)
            cv2.putText(vis, "D_E", (int((x_min + x_max) / 2) - 30, int(y_eq) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.line(vis, (int(x_min), int(y_ds)), (int(x_max), int(y_ds)), (0, 0, 255), 2)
            cv2.putText(vis, "D_S", (int((x_min + x_max) / 2) - 30, int(y_ds) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.overlay = vis
            self.display_image(vis)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")

    def display_image(self, img_bgr):
        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, 3 * w, QImage.Format_BGR888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        scaled = self.current_pixmap.scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap is not None:
            scaled = self.current_pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)


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

