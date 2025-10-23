import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


# ------------------------------ Core helpers ------------------------------

def adaptive_drop_mask(bgr):
    """Adaptive + morphological thresholding to get clean droplet contour."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
    )
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def choose_droplet_contour(mask):
    """Pick the contour with largest (area × circularity) score."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best, score = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per + 1e-6)
        s = area * circ
        if s > score:
            best, score = c, s
    return best


def find_auto_baseline_from_contour(cnt):
    ys = cnt[:, 0, 1]
    return int(np.percentile(ys, 95))  # near the bottom


def find_contact_points_by_slope(cnt, baseline_y, tol=6):
    """Detect where slope changes sharply near baseline."""
    pts = cnt[:, 0, :]
    base_pts = pts[np.abs(pts[:, 1] - baseline_y) < tol * 3]
    if len(base_pts) < 6:
        return None, None
    dx = np.gradient(base_pts[:, 0])
    dy = np.gradient(base_pts[:, 1])
    slopes = np.abs(np.degrees(np.arctan2(dy, dx)))
    rising = np.where(slopes > 30)[0]  # threshold angle where curvature rises
    if len(rising) < 2:
        return None, None
    left_pt = tuple(base_pts[rising[0]])
    right_pt = tuple(base_pts[rising[-1]])
    return left_pt, right_pt


def refine_contact_point(gray, point, search_radius=5):
    """Lock the contact on maximum local intensity gradient (edge)."""
    if point is None:
        return None
    x0, y0 = int(point[0]), int(point[1])
    best_p, best_g = (x0, y0), 0
    h, w = gray.shape
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            x, y = x0 + dx, y0 + dy
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                gx = int(gray[y, x + 1]) - int(gray[y, x - 1])
                gy = int(gray[y + 1, x]) - int(gray[y - 1, x])
                g2 = gx * gx + gy * gy
                if g2 > best_g:
                    best_g, best_p = g2, (x, y)
    return best_p


def local_poly_contact_angle(cnt, contact_pt, window_px=25):
    """Compute local tangent angle from quadratic fit."""
    if contact_pt is None:
        return None
    cx, cy = contact_pt
    pts = cnt[:, 0, :]
    nb = pts[np.abs(pts[:, 0] - cx) < window_px]
    if len(nb) < 8:
        nb = pts[np.abs(pts[:, 0] - cx) < window_px * 1.8]
    if len(nb) < 8:
        return None
    x, y = nb[:, 0].astype(float), nb[:, 1].astype(float)
    a, b, c = np.polyfit(x, y, 2)
    m = 2 * a * cx + b
    theta = np.degrees(np.arctan(abs(m)))
    return float(np.clip(theta, 0, 175))


def apex_of_contour(cnt):
    if cnt is None or len(cnt) == 0:
        return None
    return tuple(cnt[np.argmin(cnt[:, 0, 1]), 0, :])


# ------------------------------ UI Application ------------------------------

class SessileDropAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sessile Drop Contact Angle Analyzer (Final Version)")
        self.setGeometry(100, 100, 1100, 820)
        self.original_image = None
        self.current_pixmap = None
        self.draw_image = None

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.image_label = QLabel("Load a droplet image (side view).")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        self.image_label.setMinimumSize(700, 450)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, stretch=1)

        # Controls
        ctrl = QHBoxLayout()
        layout.addLayout(ctrl)
        self.btn_load = QPushButton("Load Image")
        ctrl.addWidget(self.btn_load)
        self.slider_label = QLabel("Baseline:")
        ctrl.addWidget(self.slider_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        ctrl.addWidget(self.slider)

        # Results
        res = QHBoxLayout()
        layout.addLayout(res)
        self.left_angle_label = QLabel("Left Angle: --°")
        self.right_angle_label = QLabel("Right Angle: --°")
        res.addWidget(self.left_angle_label)
        res.addWidget(self.right_angle_label)

        # Signals
        self.btn_load.clicked.connect(self.load_image)
        self.slider.valueChanged.connect(self.preview_baseline_only)
        self.slider.sliderReleased.connect(self.run_full_analysis)

    # ------------------------------------------------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.image_label.setText("Error: Could not load image.")
            return
        h, w = img.shape[:2]
        if w > 1400:
            img = cv2.resize(img, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
        self.original_image = img
        self.run_full_analysis(initial=True)

    def preview_baseline_only(self):
        if self.draw_image is None:
            return
        y = self.slider.value()
        temp = self.draw_image.copy()
        cv2.line(temp, (0, y), (temp.shape[1], y), (255, 0, 0), 2)
        self.display_image(temp)

    # ------------------------------------------------------------------

    def run_full_analysis(self, initial=False):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = adaptive_drop_mask(img)
        cnt = choose_droplet_contour(mask)
        if cnt is None:
            self.left_angle_label.setText("Left Angle: --°")
            self.right_angle_label.setText("Right Angle: --°")
            self.display_image(img)
            return

        auto_baseline = find_auto_baseline_from_contour(cnt)
        if initial or not self.slider.isEnabled():
            h = img.shape[0]
            self.slider.blockSignals(True)
            self.slider.setRange(0, h - 1)
            self.slider.setValue(int(auto_baseline))
            self.slider.setEnabled(True)
            self.slider.blockSignals(False)
        baseline_y = self.slider.value()

        # Contact detection + refinement
        l_pt, r_pt = find_contact_points_by_slope(cnt, baseline_y, tol=6)
        l_pt = refine_contact_point(gray, l_pt, 5)
        r_pt = refine_contact_point(gray, r_pt, 5)

        # Angles
        left_ang = local_poly_contact_angle(cnt, l_pt)
        right_ang = local_poly_contact_angle(cnt, r_pt)

        self.left_angle_label.setText(
            f"Left Angle: {left_ang:.2f}°" if left_ang is not None else "Left Angle: --°"
        )
        self.right_angle_label.setText(
            f"Right Angle: {right_ang:.2f}°" if right_ang is not None else "Right Angle: --°"
        )

        # Draw overlay
        vis = img.copy()
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        cv2.line(vis, (0, baseline_y), (vis.shape[1], baseline_y), (255, 0, 0), 2)
        if (apx := apex_of_contour(cnt)) is not None:
            cv2.circle(vis, apx, 6, (255, 255, 0), -1)
        for p in [l_pt, r_pt]:
            if p is not None:
                cv2.circle(vis, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1)
        self.draw_image = vis
        self.display_image(vis)

    # ------------------------------------------------------------------

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


# ------------------------------ main ------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SessileDropAnalyzer()
    w.show()
    sys.exit(app.exec_())
