import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QHBoxLayout, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ============================ Image Processing Utilities ============================

def detect_drop_contour(bgr):
    """
    Detect pendant drop contour, baseline (needle row), and ROI top.
    Uses edge detection, Otsu thresholding, and contour scoring.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    # --- Detect the bright needle tip region (topmost strong edge) ---
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.abs(sobel_y), axis=1)
    needle_row = int(np.argmax(edge_strength[:gray.shape[0] // 3]))

    # --- Define ROI below the needle ---
    roi_top = max(needle_row + 5, 0)
    roi = gray[roi_top:, :]

    # --- Compute Otsu threshold (returns threshold_value, image) ---
    otsu_val, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(5, int(0.5 * otsu_val))
    high = max(10, int(1.5 * otsu_val))

    # --- Edge detection (Canny) ---
    edges = cv2.Canny(roi, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)

    # --- Contour extraction ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, needle_row, roi_top

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
        best[:, 0, 1] += roi_top  # offset y back to full image coords
    return best, needle_row, roi_top


def measure_drop_diameters(cnt):
    """Measure equatorial and apex-proximal diameters of pendant drop."""
    if cnt is None:
        return None, None

    ys = cnt[:, 0, 1]
    xs = cnt[:, 0, 0]
    y_min, y_max = np.min(ys), np.max(ys)
    height = y_max - y_min

    # Equatorial diameter (maximum horizontal span)
    D_E = np.max(xs) - np.min(xs)

    # Diameter at 1/4 height from apex (D_S)
    target_y = y_min + 0.25 * height
    band = cnt[(ys >= target_y - 2) & (ys <= target_y + 2), 0, :]
    if len(band) > 1:
        D_S = np.max(band[:, 0]) - np.min(band[:, 0])
    else:
        D_S = None
    return D_E, D_S


def calculate_surface_tension(D_E, D_S, density, scale_mm_per_px):
    """
    Estimate surface tension using shape factor method:
    γ = (Δρ * g * D_E^2) / (H)
    where H ≈ shape factor based on D_S/D_E (simplified empirical approach).
    """
    if None in [D_E, D_S] or D_E == 0:
        return None

    g = 9.81
    D_E_mm = D_E * scale_mm_per_px
    D_S_mm = D_S * scale_mm_per_px

    ratio = D_S_mm / D_E_mm
    shape_factor = 0.5 + 1.5 * ratio  # rough empirical correction
    gamma = density * g * (D_E_mm ** 2) / (shape_factor * 1000)  # mN/m
    return gamma


# ============================ PyQt5 GUI ============================

class PendantDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pendant Drop Analyzer (Surface Tension via Shape Method)")
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
            cnt, needle_row, roi_top = detect_drop_contour(self.img)
            if cnt is None:
                QMessageBox.warning(self, "Error", "Could not detect droplet contour.")
                return

            D_E, D_S = measure_drop_diameters(cnt)

            # Example constants (replace with user inputs)
            needle_diameter_mm = 0.5
            scale_mm_per_px = needle_diameter_mm / 50.0  # e.g. 50 px ~ 0.5 mm
            density = 998  # kg/m³ for water

            gamma = calculate_surface_tension(D_E, D_S, density, scale_mm_per_px)
            if gamma is None:
                text = "Measurement failed."
            else:
                text = f"D_E = {D_E:.1f}px, D_S = {D_S:.1f}px, γ ≈ {gamma:.2f} mN/m"

            self.result_label.setText(text)

            vis = self.img.copy()
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(vis, "Detected Contour", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
    # (macOS Cocoa fix)
    import os, PyQt5.QtCore
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt5.QtCore.__file__), "Qt", "plugins", "platforms"
    )

    app = QApplication(sys.argv)
    w = PendantDropAnalyzer()
    w.show()
    sys.exit(app.exec_())

# ============================ End of File ============================\
